//! The image model's VAE ops driven through the Rust bindings (spec image-model/02 acceptance 6
//! and 9, GitHub #319): the weight packing and the three convolution kinds (with the upsample, the
//! RGBA input, bias, residual, fp32 and BF16 output, and a band through the row windows) against
//! fp64 convolutions computed here; AvgDown and DupUp (plain and accumulating) against their
//! definitions restated here, bit for bit; the mid-block attention against fp64 attention. The
//! kernel leaf's CTest (`ignis_kernel_image_vae_test`) checks the ops in depth (every VAE shape at
//! 1024² and 2048² against a tight bound); this checks that the bindings carry every argument to
//! the right place, and that a refusal comes back with the leaf's message.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing GPU or a
//! kernel error is a skip, under it a hard failure (`ignis_core::gpu_profile`). Run via
//! `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

use ignis_artifact::{CudaDevice, Device, DeviceBuffer};
use ignis_core::gpu_profile;
use ignis_core::image_vae::{self, Conv, ConvKind, ShortcutKind};
use ignis_core::moe;

fn device() -> Option<CudaDevice> {
    match CudaDevice::create(0) {
        Ok(d) => Some(d),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            None
        }
    }
}

fn upload(dev: &mut CudaDevice, bytes: &[u8]) -> DeviceBuffer {
    let buf = dev.allocate(bytes.len() as u64).expect("device allocation");
    dev.copy_h2d(&buf, 0, bytes).expect("upload");
    dev.synchronize().expect("upload sync");
    buf
}

fn download(dev: &mut CudaDevice, buf: &DeviceBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buf.len() as usize];
    dev.copy_d2h(buf, 0, &mut out).expect("download");
    dev.synchronize().expect("download sync");
    out
}

/// A device buffer of `len` bytes, all 0xFF (NaN in fp32 and in BF16): an unwritten output shows.
fn poisoned(dev: &mut CudaDevice, len: usize) -> DeviceBuffer {
    upload(dev, &vec![0xFFu8; len])
}

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

/// Runs a binding and waits for it, under the profile's rule for a kernel error. The ops run on
/// the legacy default stream, which `CudaDevice::synchronize` (its own non-blocking stream) does
/// not wait for: without this, a download could read an output before the kernel wrote it.
fn ran(result: Result<(), String>, what: &str) -> bool {
    match result.and_then(|_| moe::stream_sync(std::ptr::null_mut())) {
        Ok(()) => true,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("{what}: {e}"));
            false
        }
    }
}

fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

/// A value in [-1, 1) from a counter hash.
fn hash_unit(stream: u32, i: u32) -> f32 {
    let h = lowbias32(i.wrapping_mul(0x9E37_79B9).wrapping_add(stream.wrapping_mul(0x85EB_CA6B)));
    (h >> 8) as f32 * (1.0f32 / 8_388_608.0) - 1.0
}

fn to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn bf16s(n: usize, stream: u32, amplitude: f32) -> Vec<u16> {
    (0..n).map(|i| to_bf16(amplitude * hash_unit(stream, i as u32))).collect()
}

fn bytes16(x: &[u16]) -> Vec<u8> {
    x.iter().flat_map(|e| e.to_le_bytes()).collect()
}

fn bytes32(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|e| e.to_le_bytes()).collect()
}

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn as_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// One convolution's geometry, as the header defines it.
#[derive(Clone, Copy)]
struct Geo {
    kind: ConvKind,
    upsample: bool,
    h: usize,
    w: usize,
    cin: usize,
    cout: usize,
}

impl Geo {
    fn k(&self) -> usize {
        if self.kind == ConvKind::K1x1 { 1 } else { 3 }
    }
    fn stride(&self) -> usize {
        if self.kind == ConvKind::K3x3S2 { 2 } else { 1 }
    }
    fn out_h(&self) -> usize {
        if self.kind == ConvKind::K3x3S2 { self.h / 2 } else { self.h << self.upsample as usize }
    }
    fn out_w(&self) -> usize {
        if self.kind == ConvKind::K3x3S2 { self.w / 2 } else { self.w << self.upsample as usize }
    }
}

/// fp64 conv2d of NHWC `x` with the PyTorch-layout weight `w [cout][cin][k][k]`, plus bias and
/// residual: `[out_h][out_w][cout]`.
fn conv_ref(g: Geo, x: &[u16], w: &[u16], bias: Option<&[u16]>, residual: Option<&[u16]>) -> Vec<f64> {
    let (k, st) = (g.k(), g.stride());
    let pad = if g.kind == ConvKind::K3x3 { 1isize } else { 0 };
    let up = g.upsample as usize;
    let (hl, wl) = ((g.h << up) as isize, (g.w << up) as isize);
    let mut y = vec![0.0f64; g.out_h() * g.out_w() * g.cout];
    for oy in 0..g.out_h() {
        for ox in 0..g.out_w() {
            for o in 0..g.cout {
                let mut acc = 0.0f64;
                for ky in 0..k {
                    for kx in 0..k {
                        let iy = (oy * st + ky) as isize - pad;
                        let ix = (ox * st + kx) as isize - pad;
                        if iy < 0 || iy >= hl || ix < 0 || ix >= wl {
                            continue;
                        }
                        let (sy, sx) = ((iy as usize) >> up, (ix as usize) >> up);
                        for c in 0..g.cin {
                            acc += bf16_to_f32(x[(sy * g.w + sx) * g.cin + c]) as f64
                                * bf16_to_f32(w[((o * g.cin + c) * k + ky) * k + kx]) as f64;
                        }
                    }
                }
                let at = (oy * g.out_w() + ox) * g.cout + o;
                acc += bias.map_or(0.0, |b| bf16_to_f32(b[o]) as f64);
                acc += residual.map_or(0.0, |r| bf16_to_f32(r[at]) as f64);
                y[at] = acc;
            }
        }
    }
    y
}

fn worst_rel(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| if (g - w).abs().is_nan() { f64::INFINITY } else { m.max((g - w).abs() / scale) })
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn vae_ops_through_the_bindings_match_fp64() {
    let Some(mut dev) = device() else { return };
    let null = std::ptr::null_mut();

    // ---- the convolution: four calls, each through the weight packing ----------------------
    struct Case {
        what: &'static str,
        g: Geo,
        bias: bool,
        residual: bool,
        f32_out: bool,
    }
    let cases = [
        Case { what: "3x3, bias and residual, fp32 out", g: Geo { kind: ConvKind::K3x3, upsample: false, h: 9, w: 7, cin: 16, cout: 32 }, bias: true, residual: true, f32_out: true },
        Case { what: "3x3 stride 2, BF16 out", g: Geo { kind: ConvKind::K3x3S2, upsample: false, h: 8, w: 6, cin: 8, cout: 16 }, bias: true, residual: false, f32_out: false },
        Case { what: "3x3 with the upsample, RGBA in", g: Geo { kind: ConvKind::K3x3, upsample: true, h: 5, w: 3, cin: 4, cout: 8 }, bias: true, residual: false, f32_out: true },
        Case { what: "1x1, no bias", g: Geo { kind: ConvKind::K1x1, upsample: false, h: 4, w: 4, cin: 24, cout: 8 }, bias: false, residual: false, f32_out: true },
    ];
    let mut first_out: Option<(Vec<u8>, Geo)> = None;
    let mut first_bufs = None;
    for (i, case) in cases.iter().enumerate() {
        let g = case.g;
        let k = g.k();
        let x = bf16s(g.h * g.w * g.cin, 10 + i as u32, 2.0);
        let w = bf16s(g.cout * g.cin * k * k, 20 + i as u32, 1.0 / ((g.cin * k * k) as f32).sqrt());
        let bias = bf16s(g.cout, 30 + i as u32, 0.25);
        let res = bf16s(g.out_h() * g.out_w() * g.cout, 40 + i as u32, 3.0);
        let (d_x, d_w, d_b, d_r) = (upload(&mut dev, &bytes16(&x)), upload(&mut dev, &bytes16(&w)), upload(&mut dev, &bytes16(&bias)), upload(&mut dev, &bytes16(&res)));
        let d_wp = poisoned(&mut dev, w.len() * 2);
        let packed = unsafe { image_vae::pack_conv_weight(ptr(&d_w), g.cout as u32, g.cin as u32, k as u32, ptr(&d_wp), null) };
        if !ran(packed, "ignis_image_vae_pack_conv_weight") {
            return;
        }
        let out_bytes = g.out_h() * g.out_w() * g.cout * if case.f32_out { 4 } else { 2 };
        let d_y = poisoned(&mut dev, out_bytes);
        let call = Conv {
            kind: g.kind,
            upsample: g.upsample,
            x: ptr(&d_x),
            height: g.h as u32,
            width: g.w as u32,
            in_channels: g.cin as u32,
            x_row0: 0,
            x_rows: g.h as u32,
            weight: ptr(&d_wp),
            bias: if case.bias { ptr(&d_b) } else { std::ptr::null() },
            out_channels: g.cout as u32,
            residual: if case.residual { ptr(&d_r) } else { std::ptr::null() },
            y: ptr(&d_y),
            y_row0: 0,
            y_rows: g.out_h() as u32,
            y_f32: case.f32_out,
        };
        if !ran(unsafe { image_vae::conv(&call, null) }, case.what) {
            return;
        }
        let out = download(&mut dev, &d_y);
        let got: Vec<f64> = if case.f32_out { as_f32(&out).iter().map(|&v| v as f64).collect() } else { as_u16(&out).iter().map(|&h| bf16_to_f32(h) as f64).collect() };
        let want = conv_ref(g, &x, &w, case.bias.then_some(&bias[..]), case.residual.then_some(&res[..]));
        // fp32 output sits within ~1e-6 of fp64 here, BF16 within its 2^-9; a misplaced argument is
        // an O(1) error.
        let rel = worst_rel(&got, &want);
        let tol = if case.f32_out { 1e-5 } else { 1e-2 };
        assert!(rel <= tol, "{}: worst error {rel} of max |y|", case.what);
        if i == 0 {
            first_out = Some((out, g));
            first_bufs = Some((d_x, d_wp, d_b, d_r));
            dev.deallocate(d_w).unwrap();
            dev.deallocate(d_y).unwrap();
        } else {
            for buf in [d_x, d_w, d_wp, d_b, d_r, d_y] {
                dev.deallocate(buf).unwrap();
            }
        }
    }

    // A band of the first call: output rows 3..6 from a copy of input rows 2..7 only, the same
    // values bit for bit.
    let (whole, g) = first_out.unwrap();
    let (d_x, d_wp, d_b, d_r) = first_bufs.unwrap();
    let x_all = download(&mut dev, &d_x);
    let in_row = g.w * g.cin * 2;
    let d_xb = upload(&mut dev, &x_all[2 * in_row..7 * in_row]);
    let out_row = g.out_w() * g.cout;
    let r_all = download(&mut dev, &d_r);
    let d_rb = upload(&mut dev, &r_all[3 * out_row * 2..6 * out_row * 2]);
    let d_yb = poisoned(&mut dev, 3 * out_row * 4);
    let band = Conv {
        kind: g.kind,
        upsample: false,
        x: ptr(&d_xb),
        height: g.h as u32,
        width: g.w as u32,
        in_channels: g.cin as u32,
        x_row0: 2,
        x_rows: 5,
        weight: ptr(&d_wp),
        bias: ptr(&d_b),
        out_channels: g.cout as u32,
        residual: ptr(&d_rb),
        y: ptr(&d_yb),
        y_row0: 3,
        y_rows: 3,
        y_f32: true,
    };
    if !ran(unsafe { image_vae::conv(&band, null) }, "a band of the 3x3 call") {
        return;
    }
    assert_eq!(download(&mut dev, &d_yb), whole[3 * out_row * 4..6 * out_row * 4].to_vec(), "the band equals the whole-image call's rows, bit for bit");
    // One input row short of the halo is refused, with the leaf's message.
    let short = Conv { x_row0: 3, x_rows: 4, x: unsafe { (ptr(&d_xb) as *const u8).add(in_row) as *const c_void }, ..band };
    let message = unsafe { image_vae::conv(&short, null) }.expect_err("a window missing a halo row must be refused");
    assert!(message.contains("ignis_image_vae_conv") && message.contains("x holds input rows"), "message: {message}");
    for buf in [d_x, d_wp, d_b, d_r, d_xb, d_rb, d_yb] {
        dev.deallocate(buf).unwrap();
    }

    // ---- the shortcuts: AvgDown in fp32, DupUp accumulating in BF16 -------------------------
    let (h, w, cin, cout) = (4usize, 6usize, 16usize, 32usize);
    let x: Vec<f32> = (0..h * w * cin).map(|i| 4.0 * hash_unit(50, i as u32)).collect();
    let d_x = upload(&mut dev, &bytes32(&x));
    let d_y = poisoned(&mut dev, (h / 2) * (w / 2) * cout * 4);
    if !ran(unsafe { image_vae::shortcut(ShortcutKind::AvgDown, ptr(&d_x), h as u32, w as u32, cin as u32, cout as u32, 2, 2, ptr(&d_y), false, true, null) }, "AvgDown") {
        return;
    }
    let got = as_f32(&download(&mut dev, &d_y));
    // group = 16 * 8 / 32 = 4: output channel o is channel o / 2's slot o % 2 (slot 0 the zero
    // padding frame) over the 2 x 2 window, summed from 0 in order and divided once.
    for (e, &v) in got.iter().enumerate() {
        let (o, px) = (e % cout, e / cout);
        let (oy, ox) = (px / (w / 2), px % (w / 2));
        let mut sum = 0.0f32;
        for j in 0..4 {
            sum += if o % 2 == 1 { x[((oy * 2 + j / 2) * w + ox * 2 + j % 2) * cin + o / 2] } else { 0.0 };
        }
        assert_eq!(v.to_bits(), (sum / 4.0).to_bits(), "AvgDown output {e}");
    }
    dev.deallocate(d_x).unwrap();
    dev.deallocate(d_y).unwrap();

    let (h, w, cin, cout) = (2usize, 3usize, 16usize, 8usize);
    let x = bf16s(h * w * cin, 51, 2.0);
    let y0 = bf16s(4 * h * w * cout, 52, 2.0);
    let (d_x, d_y) = (upload(&mut dev, &bytes16(&x)), upload(&mut dev, &bytes16(&y0)));
    if !ran(unsafe { image_vae::shortcut(ShortcutKind::DupUp, ptr(&d_x), h as u32, w as u32, cin as u32, cout as u32, 2, 2, ptr(&d_y), true, false, null) }, "DupUp") {
        return;
    }
    let got = as_u16(&download(&mut dev, &d_y));
    // repeats = 8 * 8 / 16 = 4: output channel o at sub-pixel (sh, sw) copies channel (o * 8 + 4
    // + sh * 2 + sw) / 4 (the frame's slot is the second), added into y in BF16.
    for (e, &v) in got.iter().enumerate() {
        let (o, px) = (e % cout, e / cout);
        let (yy, xx) = (px / (2 * w), px % (2 * w));
        let c = (o * 8 + 4 + (yy % 2) * 2 + xx % 2) / 4;
        let s = x[((yy / 2) * w + xx / 2) * cin + c];
        assert_eq!(v, to_bf16(bf16_to_f32(y0[e]) + bf16_to_f32(s)), "DupUp accumulate output {e}");
    }
    dev.deallocate(d_x).unwrap();
    dev.deallocate(d_y).unwrap();

    // ---- the mid-block attention: 37 tokens, one head of 64, fp32 and BF16 out ----------------
    let (tokens, dim) = (37usize, 64usize);
    let scale = 1.0f32 / (dim as f32).sqrt();
    let qkv = bf16s(tokens * 3 * dim, 60, 2.5);
    let d_qkv = upload(&mut dev, &bytes16(&qkv));
    let row_tile = 16u32;
    let ws_bytes = image_vae::attn_workspace_bytes(tokens as u32, dim as u32, row_tile);
    assert!(ws_bytes > 0, "the workspace query answers");
    let d_ws = poisoned(&mut dev, ws_bytes as usize);
    let at = |t: usize, off: usize, d: usize| bf16_to_f32(qkv[t * 3 * dim + off + d]) as f64;
    let want: Vec<f64> = (0..tokens)
        .flat_map(|i| {
            let s: Vec<f64> = (0..tokens).map(|j| (0..dim).map(|d| at(i, 0, d) * at(j, dim, d)).sum::<f64>() * scale as f64).collect();
            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
            let sum: f64 = p.iter().sum();
            (0..dim).map(move |d| (0..tokens).map(|j| p[j] * at(j, 2 * dim, d)).sum::<f64>() / sum).collect::<Vec<f64>>()
        })
        .collect();
    for f32_out in [true, false] {
        let d_o = poisoned(&mut dev, tokens * dim * if f32_out { 4 } else { 2 });
        if !ran(unsafe { image_vae::attn(ptr(&d_qkv), tokens as u32, dim as u32, scale, row_tile, ptr(&d_ws), ws_bytes, ptr(&d_o), f32_out, null) }, "the attention") {
            return;
        }
        let out = download(&mut dev, &d_o);
        let got: Vec<f64> = if f32_out { as_f32(&out).iter().map(|&v| v as f64).collect() } else { as_u16(&out).iter().map(|&h| bf16_to_f32(h) as f64).collect() };
        let (ee, ww) = got.iter().zip(&want).fold((0.0f64, 0.0f64), |(e, w), (g, v)| (e + (g - v) * (g - v), w + v * v));
        let rel = (ee / ww).sqrt();
        // The kernel's P is BF16 (~2e-3 per weight), so ~5e-4 relative RMS; a misplaced argument
        // is an O(1) error.
        assert!(rel <= 0.01 && !rel.is_nan(), "attention ({}): relative RMS error {rel}", if f32_out { "fp32" } else { "BF16" });
        dev.deallocate(d_o).unwrap();
    }
    let refused = unsafe { image_vae::attn(ptr(&d_qkv), tokens as u32, 12, scale, row_tile, ptr(&d_ws), ws_bytes, ptr(&d_ws), true, null) };
    let message = refused.expect_err("dim 12 must be refused");
    assert!(message.contains("ignis_image_vae_attn") && message.contains("dim"), "message: {message}");
    dev.deallocate(d_qkv).unwrap();
    dev.deallocate(d_ws).unwrap();
}
