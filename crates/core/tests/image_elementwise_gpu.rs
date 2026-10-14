//! The image model's elementwise ops driven through the Rust bindings (spec image-model/02
//! acceptance 7 and 9, GitHub #319): the RoPE positions (a layout worked by hand) and table, the
//! modulated LayerNorm with its per-token t / t=0 row, the zero-centred RMSNorm, the gated
//! residual, the QK RMSNorm with RoPE in place on a fused QKV row, SiLU, GELU-tanh, SwiGLU and the
//! VAE channel norm, each against fp64 computed here; the E4M3 outputs against
//! `image_gemm::quantize_tokens` of the BF16 output and the BF16 outputs against the fp32 ones
//! rounded, bit for bit. The kernel leaf's CTest (`ignis_kernel_image_elementwise_test`) checks
//! the ops in depth (the DiT's shapes against a bound that tracks every rounding); this checks that
//! the bindings carry every argument to the right place, and that a refusal comes back with the
//! leaf's message.
//!
//! The RoPE table of the oracle's t2i layout is compared with diffusers' own (`rope.npy`, spec
//! 01's recordings) when the oracle is there: `$IGNIS_IMAGE_ORACLE`, else
//! `F:/ai/models/ignis-image-2.1-turbo/oracle`; without it that part prints SKIP (a failure under the profile).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing GPU or a
//! kernel error is a skip, under it a hard failure (`ignis_core::gpu_profile`). Run via
//! `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

use ignis_artifact::{CudaDevice, Device, DeviceBuffer};
use ignis_core::gpu_profile;
use ignis_core::image_dit::{self, Act, Out};
use ignis_core::{image_gemm, image_vae, moe};

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

fn bf16s(n: usize, stream: u32, centre: f32, amplitude: f32) -> Vec<u16> {
    (0..n).map(|i| to_bf16(centre + amplitude * hash_unit(stream, i as u32))).collect()
}

fn f32s(n: usize, stream: u32, amplitude: f32) -> Vec<f32> {
    (0..n).map(|i| amplitude * hash_unit(stream, i as u32)).collect()
}

fn bytes16(x: &[u16]) -> Vec<u8> {
    x.iter().flat_map(|e| e.to_le_bytes()).collect()
}

fn bytes32(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|e| e.to_le_bytes()).collect()
}

fn bytes_i32(x: &[i32]) -> Vec<u8> {
    x.iter().flat_map(|e| e.to_le_bytes()).collect()
}

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn as_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn f64s(x: &[u16]) -> Vec<f64> {
    x.iter().map(|&h| bf16_to_f32(h) as f64).collect()
}

/// The worst |got - want| over max |want|; NaN counts as infinite.
fn worst_rel(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter().zip(want).fold(0.0f64, |m, (&g, w)| {
        let d = (g as f64 - w).abs();
        if d.is_nan() { f64::INFINITY } else { m.max(d / scale) }
    })
}

/// fp32 output sits within ~1e-6 of fp64 here; a misplaced argument is an O(1) error.
const F32_TOL: f64 = 1e-5;
const EPS: f32 = 1e-6;

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn gelu_tanh(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// `x * rsqrt(mean(x^2) + eps)` per row of `cols`.
fn rms_rows(x: &[f64], cols: usize) -> Vec<f64> {
    x.chunks(cols)
        .flat_map(|r| {
            let m = r.iter().map(|v| v * v).sum::<f64>() / cols as f64;
            let s = 1.0 / (m + EPS as f64).sqrt();
            r.iter().map(move |v| v * s).collect::<Vec<_>>()
        })
        .collect()
}

/// RoPE angle of position `pos` at pair `p` with diffusers' frequencies in fp64 (the fp32 rounding
/// of the frequencies and the angle is below the tolerance at these small positions).
fn rope_angle(pos: &[i32], p: usize) -> f64 {
    let (axis, i, d) = if p < 8 { (0, p, 16) } else { (1 + (p - 8) / 28, (p - 8) % 28, 56) };
    pos[axis] as f64 * 10000f64.powf(-2.0 * i as f64 / d as f64)
}

/// A little-endian C-order .npy's shape and payload; None when the file is missing.
fn load_npy(path: &std::path::Path) -> Option<(Vec<usize>, Vec<u8>)> {
    let raw = std::fs::read(path).ok()?;
    assert_eq!(&raw[..6], b"\x93NUMPY", "{} is not a .npy", path.display());
    let (len, at) = if raw[6] == 1 { (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10) } else { (u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize, 12) };
    let header = std::str::from_utf8(&raw[at..at + len]).unwrap();
    assert!(header.contains("'fortran_order': False"), "{} is not C order", path.display());
    let shape_at = header.find("'shape': (").unwrap() + 10;
    let shape: Vec<usize> = header[shape_at..header[shape_at..].find(')').unwrap() + shape_at].split(',').filter_map(|s| s.trim().parse().ok()).collect();
    Some((shape, raw[at + len..].to_vec()))
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn elementwise_ops_through_the_bindings_match_fp64() {
    let Some(mut dev) = device() else { return };
    let null = std::ptr::null_mut();

    // ---- RoPE: positions (host) worked by hand, then the table on the device ------------------
    // text 3, a 2 x 3 block, text 2, a 4 x 5 block, text 2.
    let pos = image_dit::rope_positions(33, &[3, 11], &[(2, 3), (4, 5)]).expect("rope_positions");
    let at = |t: usize| [pos[3 * t], pos[3 * t + 1], pos[3 * t + 2]];
    assert_eq!([at(2), at(3), at(8), at(9), at(11), at(30), at(31)], [[2, 2, 2], [3, -1, -2], [3, 0, 0], [6, 6, 6], [8, -2, -3], [8, 1, 1], [13, 13, 13]], "positions of the hand-worked layout");
    let message = image_dit::rope_positions(20, &[2, 5], &[(2, 2), (2, 2)]).expect_err("overlapping blocks must be refused");
    assert!(message.contains("ignis_image_dit_rope_positions") && message.contains("overlaps"), "message: {message}");
    let d_pos = upload(&mut dev, &bytes_i32(&pos));
    let d_table = poisoned(&mut dev, 33 * 64 * 8);
    if !ran(unsafe { image_dit::rope_table(ptr(&d_pos) as *const i32, 33, ptr(&d_table) as *mut f32, null) }, "rope_table") {
        return;
    }
    let table = as_f32(&download(&mut dev, &d_table));
    for t in 0..33 {
        for p in 0..64 {
            let a = rope_angle(&pos[3 * t..3 * t + 3], p);
            let (c, s) = (table[t * 128 + 2 * p] as f64, table[t * 128 + 2 * p + 1] as f64);
            assert!((c - a.cos()).abs() <= 1e-4 && (s - a.sin()).abs() <= 1e-4, "table token {t} pair {p}: ({c}, {s}) for angle {a}");
        }
    }
    dev.deallocate(d_pos).unwrap();
    dev.deallocate(d_table).unwrap();

    // The oracle's t2i table, diffusers' own, within one ulp: 2^-24 absolute is one ulp only for magnitudes in [0.5, 1); the CTest's ulp_of is the exact check.
    let oracle = std::env::var("IGNIS_IMAGE_ORACLE").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "F:/ai/models/ignis-image-2.1-turbo/oracle".to_string());
    let case = std::path::Path::new(&oracle).join("t2i");
    match (load_npy(&case.join("rope.npy")), load_npy(&case.join("joint_image_mask.npy"))) {
        (Some((shape, rope)), Some((_, mask))) => {
            let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(case.join("case.json")).unwrap()).unwrap();
            let shapes: Vec<(u32, u32)> = json["img_shapes"].as_array().unwrap().iter().map(|s| (s[1].as_u64().unwrap() as u32, s[2].as_u64().unwrap() as u32)).collect();
            let tokens = mask.len() as u32;
            assert_eq!(shape, vec![tokens as usize, 64, 2], "rope.npy's shape");
            let mut starts = Vec::new();
            let mut t = 0usize;
            for &(h, w) in &shapes {
                while mask[t] == 0 {
                    t += 1;
                }
                starts.push(t as u32);
                t += (h * w) as usize;
            }
            let pos = image_dit::rope_positions(tokens, &starts, &shapes).expect("the oracle's layout");
            let d_pos = upload(&mut dev, &bytes_i32(&pos));
            let d_table = poisoned(&mut dev, tokens as usize * 512);
            if !ran(unsafe { image_dit::rope_table(ptr(&d_pos) as *const i32, tokens, ptr(&d_table) as *mut f32, null) }, "rope_table (oracle)") {
                return;
            }
            let got = as_f32(&download(&mut dev, &d_table));
            let want = as_f32(&rope);
            let worst = got.iter().zip(&want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
            assert!(worst <= f32::EPSILON / 2.0, "the t2i table is {worst} from diffusers' rope.npy");
            println!("oracle t2i: {tokens} tokens, worst |diff| {worst:e} from rope.npy");
            dev.deallocate(d_pos).unwrap();
            dev.deallocate(d_table).unwrap();
        }
        _ => {
            gpu_profile::skip_or_fail(&format!("the oracle's RoPE comparison: no oracle at {oracle} (IGNIS_IMAGE_ORACLE)"));
        }
    }

    // ---- the modulated LayerNorm: 5 tokens of 64, the first 2 a prefix -----------------------
    let (tokens, cols, first_target) = (5usize, 64usize, 2usize);
    let x = bf16s(tokens * cols, 10, 0.5, 2.0);
    let (scale_t, scale_0) = (f32s(cols, 11, 1.5), f32s(cols, 12, 0.5));
    let (d_x, d_st, d_s0) = (upload(&mut dev, &bytes16(&x)), upload(&mut dev, &bytes32(&scale_t)), upload(&mut dev, &bytes32(&scale_0)));
    let d_y32 = poisoned(&mut dev, tokens * cols * 4);
    let d_y16 = poisoned(&mut dev, tokens * cols * 2);
    let d_codes = poisoned(&mut dev, tokens * cols);
    let d_scales = poisoned(&mut dev, tokens * 4);
    for (y, ys, out) in [(&d_y32, null, Out::F32), (&d_y16, null, Out::Bf16), (&d_codes, ptr(&d_scales), Out::E4m3)] {
        let r = unsafe { image_dit::modulated_norm(ptr(&d_x), tokens as u32, cols as u32, ptr(&d_st) as *const f32, ptr(&d_s0) as *const f32, first_target as u32, EPS, ptr(y), ys as *mut f32, out, null) };
        if !ran(r, "modulated_norm") {
            return;
        }
    }
    let xf = f64s(&x);
    let want: Vec<f64> = xf
        .chunks(cols)
        .enumerate()
        .flat_map(|(t, r)| {
            let mean = r.iter().sum::<f64>() / cols as f64;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / cols as f64;
            let s = if t < first_target { &scale_0 } else { &scale_t };
            r.iter().enumerate().map(move |(c, v)| (v - mean) / (var + EPS as f64).sqrt() * (1.0 + s[c] as f64)).collect::<Vec<_>>()
        })
        .collect();
    let y32 = as_f32(&download(&mut dev, &d_y32));
    let rel = worst_rel(&y32, &want);
    assert!(rel <= F32_TOL, "modulated_norm: worst error {rel} of max |y|");
    assert_eq!(as_u16(&download(&mut dev, &d_y16)), y32.iter().map(|&v| to_bf16(v)).collect::<Vec<_>>(), "modulated_norm: BF16 = the fp32 output rounded");
    // E4M3: the standalone quantization of the BF16 output, bit for bit.
    let d_qc = poisoned(&mut dev, tokens * cols);
    let d_qs = poisoned(&mut dev, tokens * 4);
    if !ran(unsafe { image_gemm::quantize_tokens(ptr(&d_y16), tokens as u32, cols as u32, ptr(&d_qc), ptr(&d_qs) as *mut f32, null) }, "quantize_tokens") {
        return;
    }
    assert_eq!(download(&mut dev, &d_codes), download(&mut dev, &d_qc), "modulated_norm: E4M3 codes = quantize_tokens of the BF16 output");
    assert_eq!(download(&mut dev, &d_scales), download(&mut dev, &d_qs), "modulated_norm: E4M3 scales = quantize_tokens of the BF16 output");
    let message = unsafe { image_dit::modulated_norm(ptr(&d_x), tokens as u32, cols as u32, ptr(&d_st) as *const f32, std::ptr::null(), 1, EPS, ptr(&d_y32), null as *mut f32, Out::F32, null) }.expect_err("a NULL t=0 row read must be refused");
    assert!(message.contains("ignis_image_dit_modulated_norm") && message.contains("modulation row"), "message: {message}");
    for buf in [d_st, d_s0, d_codes, d_scales, d_qc, d_qs] {
        dev.deallocate(buf).unwrap();
    }

    // ---- the zero-centred RMSNorm on the same x --------------------------------------------
    let w = bf16s(cols, 13, 0.0, 0.4);
    let d_w = upload(&mut dev, &bytes16(&w));
    if !ran(unsafe { image_dit::rms_norm(ptr(&d_x), tokens as u32, cols as u32, ptr(&d_w), EPS, ptr(&d_y32), null as *mut f32, Out::F32, null) }, "rms_norm") {
        return;
    }
    let want: Vec<f64> = rms_rows(&xf, cols).iter().enumerate().map(|(e, v)| v * (bf16_to_f32(w[e % cols]) as f64 + 1.0)).collect();
    let rel = worst_rel(&as_f32(&download(&mut dev, &d_y32)), &want);
    assert!(rel <= F32_TOL, "rms_norm: worst error {rel} of max |y|");
    dev.deallocate(d_w).unwrap();

    // ---- the gated residual: x + tanh(g) y, fp32 out, then in place in BF16 -------------------
    let yv = bf16s(tokens * cols, 14, 0.0, 3.0);
    let (gate_t, gate_0) = (f32s(cols, 15, 3.0), f32s(cols, 16, 1.0));
    let (d_yv, d_gt, d_g0) = (upload(&mut dev, &bytes16(&yv)), upload(&mut dev, &bytes32(&gate_t)), upload(&mut dev, &bytes32(&gate_0)));
    let residual = |out: &DeviceBuffer, x: &DeviceBuffer, f32_out: bool| unsafe {
        image_dit::gated_residual(ptr(x), ptr(&d_yv), tokens as u32, cols as u32, ptr(&d_gt) as *const f32, ptr(&d_g0) as *const f32, first_target as u32, ptr(out), f32_out, null)
    };
    if !ran(residual(&d_y32, &d_x, true), "gated_residual") || !ran(residual(&d_x, &d_x, false), "gated_residual in place") {
        return;
    }
    let want: Vec<f64> = (0..tokens * cols)
        .map(|e| {
            let g = if e / cols < first_target { gate_0[e % cols] } else { gate_t[e % cols] };
            xf[e] + (g as f64).tanh() * bf16_to_f32(yv[e]) as f64
        })
        .collect();
    let y32 = as_f32(&download(&mut dev, &d_y32));
    let rel = worst_rel(&y32, &want);
    assert!(rel <= F32_TOL, "gated_residual: worst error {rel} of max |y|");
    assert_eq!(as_u16(&download(&mut dev, &d_x)), y32.iter().map(|&v| to_bf16(v)).collect::<Vec<_>>(), "gated_residual: in place = the fp32 output rounded");
    for buf in [d_x, d_yv, d_gt, d_g0, d_y32, d_y16] {
        dev.deallocate(buf).unwrap();
    }

    // ---- the QK norm with RoPE in place on a fused QKV row [3][12288] ------------------------
    let (tokens, stride) = (3usize, 12288usize);
    let qkv = bf16s(tokens * stride, 20, 0.0, 2.0);
    let (qw, kw) = (bf16s(128, 21, 1.0, 0.3), bf16s(128, 22, 0.8, 0.5));
    let pos = image_dit::rope_positions(tokens as u32, &[], &[]).unwrap();
    let pos = pos.iter().map(|&p| p + 5).collect::<Vec<_>>(); // positions 5, 6, 7 on every axis
    let (d_qkv, d_qw, d_kw, d_pos) = (upload(&mut dev, &bytes16(&qkv)), upload(&mut dev, &bytes16(&qw)), upload(&mut dev, &bytes16(&kw)), upload(&mut dev, &bytes_i32(&pos)));
    let d_table = poisoned(&mut dev, tokens * 512);
    let d_out = poisoned(&mut dev, tokens * 8192 * 4);
    if !ran(unsafe { image_dit::rope_table(ptr(&d_pos) as *const i32, tokens as u32, ptr(&d_table) as *mut f32, null) }, "rope_table") {
        return;
    }
    let q = ptr(&d_qkv);
    let k = unsafe { (q as *mut u16).add(4096) as *mut c_void };
    let out_k = unsafe { (ptr(&d_out) as *mut f32).add(4096) as *mut c_void };
    let table = ptr(&d_table) as *const f32;
    if !ran(unsafe { image_dit::qk_norm_rope(q, k, tokens as u32, stride as u32, ptr(&d_qw), ptr(&d_kw), EPS, table, ptr(&d_out), out_k, 8192, true, null) }, "qk_norm_rope")
        || !ran(unsafe { image_dit::qk_norm_rope(q, k, tokens as u32, stride as u32, ptr(&d_qw), ptr(&d_kw), EPS, table, q, k, stride as u32, false, null) }, "qk_norm_rope in place")
    {
        return;
    }
    let qkvf = f64s(&qkv);
    let mut want = vec![0.0f64; tokens * 8192];
    for t in 0..tokens {
        for head in 0..64 {
            let w = if head < 32 { &qw } else { &kw };
            let x = &qkvf[t * stride + head * 128..t * stride + head * 128 + 128];
            let n: Vec<f64> = rms_rows(x, 128).iter().enumerate().map(|(c, v)| v * bf16_to_f32(w[c]) as f64).collect();
            for p in 0..64 {
                let a = rope_angle(&pos[3 * t..3 * t + 3], p);
                want[t * 8192 + head * 128 + 2 * p] = n[2 * p] * a.cos() - n[2 * p + 1] * a.sin();
                want[t * 8192 + head * 128 + 2 * p + 1] = n[2 * p] * a.sin() + n[2 * p + 1] * a.cos();
            }
        }
    }
    let got = as_f32(&download(&mut dev, &d_out));
    let rel = worst_rel(&got, &want);
    assert!(rel <= 1e-4, "qk_norm_rope: worst error {rel} of max |y|");
    let after = as_u16(&download(&mut dev, &d_qkv));
    for t in 0..tokens {
        let row = &after[t * stride..(t + 1) * stride];
        assert_eq!(row[..8192].to_vec(), got[t * 8192..(t + 1) * 8192].iter().map(|&v| to_bf16(v)).collect::<Vec<_>>(), "qk_norm_rope token {t}: in place = the fp32 output rounded");
        assert_eq!(row[8192..].to_vec(), qkv[t * stride + 8192..(t + 1) * stride].to_vec(), "qk_norm_rope token {t}: v untouched");
    }
    let message = unsafe { image_dit::qk_norm_rope(q, k, tokens as u32, 4000, ptr(&d_qw), ptr(&d_kw), EPS, table, q, k, 4000, false, null) }.expect_err("row_stride 4000 must be refused");
    assert!(message.contains("ignis_image_dit_qk_norm_rope") && message.contains("row_stride"), "message: {message}");
    for buf in [d_qkv, d_qw, d_kw, d_pos, d_table, d_out] {
        dev.deallocate(buf).unwrap();
    }

    // ---- the activations: SiLU and GELU-tanh, SwiGLU from a fused gate | up row --------------
    let (tokens, cols) = (3usize, 64usize);
    let gu = bf16s(tokens * 2 * cols, 30, 0.0, 6.0);
    let guf = f64s(&gu);
    let d_gu = upload(&mut dev, &bytes16(&gu));
    let d_y = poisoned(&mut dev, tokens * cols * 4);
    for act in [Act::Silu, Act::GeluTanh, Act::Swiglu] {
        let up = if act == Act::Swiglu { unsafe { (ptr(&d_gu) as *const u16).add(cols) as *const c_void } } else { std::ptr::null() };
        if !ran(unsafe { image_dit::act(act, ptr(&d_gu), up, tokens as u32, cols as u32, 2 * cols as u32, ptr(&d_y), null as *mut f32, Out::F32, null) }, "act") {
            return;
        }
        let want: Vec<f64> = (0..tokens * cols)
            .map(|e| {
                let g = guf[(e / cols) * 2 * cols + e % cols];
                match act {
                    Act::Silu => silu(g),
                    Act::GeluTanh => gelu_tanh(g),
                    Act::Swiglu => silu(g) * guf[(e / cols) * 2 * cols + cols + e % cols],
                }
            })
            .collect();
        let rel = worst_rel(&as_f32(&download(&mut dev, &d_y)), &want);
        assert!(rel <= F32_TOL, "act {act:?}: worst error {rel} of max |y|");
    }
    let message = unsafe { image_dit::act(Act::Swiglu, ptr(&d_gu), std::ptr::null(), tokens as u32, cols as u32, 2 * cols as u32, ptr(&d_y), null as *mut f32, Out::F32, null) }.expect_err("SwiGLU without up must be refused");
    assert!(message.contains("ignis_image_dit_act") && message.contains("up"), "message: {message}");
    dev.deallocate(d_gu).unwrap();
    dev.deallocate(d_y).unwrap();

    // ---- the VAE channel norm with SiLU: 5 pixels of 96, fp32 out, then in place ------------
    let (pixels, channels) = (5usize, 96usize);
    let x = bf16s(pixels * channels, 40, 0.3, 4.0);
    let gamma = bf16s(channels, 41, 1.0, 0.5);
    let (d_x, d_g) = (upload(&mut dev, &bytes16(&x)), upload(&mut dev, &bytes16(&gamma)));
    let d_y = poisoned(&mut dev, pixels * channels * 4);
    if !ran(unsafe { image_vae::rms_norm(ptr(&d_x), pixels as u32, channels as u32, ptr(&d_g), true, ptr(&d_y), true, null) }, "vae rms_norm")
        || !ran(unsafe { image_vae::rms_norm(ptr(&d_x), pixels as u32, channels as u32, ptr(&d_g), true, ptr(&d_x), false, null) }, "vae rms_norm in place")
    {
        return;
    }
    let gamma_f = f64s(&gamma);
    let want: Vec<f64> = f64s(&x)
        .chunks(channels)
        .flat_map(|r| {
            let norm = r.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            r.iter().zip(&gamma_f).map(|(v, g)| silu(v / norm * (channels as f64).sqrt() * g)).collect::<Vec<_>>()
        })
        .collect();
    let y32 = as_f32(&download(&mut dev, &d_y));
    let rel = worst_rel(&y32, &want);
    assert!(rel <= F32_TOL, "vae rms_norm: worst error {rel} of max |y|");
    assert_eq!(as_u16(&download(&mut dev, &d_x)), y32.iter().map(|&v| to_bf16(v)).collect::<Vec<_>>(), "vae rms_norm: in place = the fp32 output rounded");
    let message = unsafe { image_vae::rms_norm(ptr(&d_x), pixels as u32, 100, ptr(&d_g), true, ptr(&d_y), true, null) }.expect_err("100 channels must be refused");
    assert!(message.contains("ignis_image_vae_rms_norm") && message.contains("channels"), "message: {message}");
    for buf in [d_x, d_g, d_y] {
        dev.deallocate(buf).unwrap();
    }
}
