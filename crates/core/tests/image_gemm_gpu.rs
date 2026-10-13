//! The image model's FP8 W8A8 GEMM and per-token quantization driven through the Rust bindings
//! (spec image-model/02 acceptance 2, 3 and 9, GitHub #319): quantize BF16 activations, multiply
//! them by a row-scale-v1 FP8 weight, and check the outputs against an fp64 GEMM of the
//! dequantized operands computed here. The kernel leaf's CTest (`ignis_kernel_image_gemm_test`)
//! checks the ops in depth at every linear shape; this checks that the bindings carry every
//! argument to the right place, and that a refusal comes back with the leaf's message.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing GPU or a
//! kernel error is a skip, under it a hard failure (`ignis_core::gpu_profile`). Run via
//! `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

use ignis_artifact::{CudaDevice, Device, DeviceBuffer};
use ignis_core::gpu_profile;
use ignis_core::image_gemm;
use ignis_core::moe;

/// The device, prepared for the image GEMM, or `None` after the profile's verdict.
fn device() -> Option<CudaDevice> {
    let dev = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            return None;
        }
    };
    if let Err(e) = image_gemm::prepare() {
        gpu_profile::skip_or_fail(&format!("ignis_image_gemm_prepare: {e}"));
        return None;
    }
    Some(dev)
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

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

/// Runs a binding and waits for it, under the profile's rule for a kernel error. The ops run on
/// the legacy default stream, which `CudaDevice::synchronize` (its own non-blocking stream) does
/// not wait for: without this, a download could read the output before the kernel wrote it.
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

/// BF16 (round to nearest even) of a value in [-1, 1).
fn hash_bf16(stream: u32, i: u32) -> u16 {
    let h = lowbias32(i.wrapping_mul(0x9E37_79B9).wrapping_add(stream.wrapping_mul(0x85EB_CA6B)));
    let v = (h >> 8) as f32 * (1.0f32 / 8_388_608.0) - 1.0;
    let b = v.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_to_f64(h: u16) -> f64 {
    f32::from_bits((h as u32) << 16) as f64
}

/// E4M3FN to its value (no infinities; 0x7F/0xFF are NaN).
fn e4m3(c: u8) -> f64 {
    let sign = if c & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (((c >> 3) & 15) as i32, (c & 7) as f64);
    if e == 15 && m == 7.0 {
        return f64::NAN;
    }
    let v = if e == 0 { m * 2f64.powi(-9) } else { (1.0 + m / 8.0) * 2f64.powi(e - 7) };
    sign * v
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn quantized_gemm_through_the_bindings_matches_fp64_of_its_dequantized_operands() {
    let Some(mut dev) = device() else { return };
    let (tokens, rows, cols) = (37u32, 96u32, 128u32);
    // The weight: row-scale-v1 (layout.md §6.1), codes then BF16 row scales at 256 bytes.
    let codes_len = (rows * cols) as usize;
    let scale_at = codes_len.div_ceil(256) * 256;
    let mut weight = vec![0u8; scale_at + rows as usize * 2];
    for (i, w) in weight[..codes_len].iter_mut().enumerate() {
        let c = (lowbias32(i as u32 ^ 0x5151) & 0xFF) as u8;
        *w = if c & 0x7F == 0x7F { c ^ 1 } else { c };
    }
    // BF16 row scales near 0.002, each a different value; the reference reads them back as stored.
    let mut wscale = Vec::with_capacity(rows as usize);
    for r in 0..rows as usize {
        let bits = ((0.002f32 * (1.0 + (r % 7) as f32 / 8.0)).to_bits() >> 16) as u16;
        weight[scale_at + 2 * r..scale_at + 2 * r + 2].copy_from_slice(&bits.to_le_bytes());
        wscale.push(bf16_to_f64(bits));
    }
    let x: Vec<u16> = (0..tokens * cols).map(|i| hash_bf16(31, i)).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dw = upload(&mut dev, &weight);
    let dx = upload(&mut dev, &x_bytes);
    let dc = dev.allocate((tokens * cols) as u64).expect("device allocation");
    let ds = dev.allocate(tokens as u64 * 4).expect("device allocation");
    let quantize = unsafe { image_gemm::quantize_tokens(ptr(&dx), tokens, cols, ptr(&dc), ptr(&ds) as *mut f32, std::ptr::null_mut()) };
    if !ran(quantize, "ignis_image_quantize_tokens") {
        return;
    }
    let codes = download(&mut dev, &dc);
    let scales: Vec<f64> = download(&mut dev, &ds).chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64).collect();
    for t in 0..tokens as usize {
        let amax = (0..cols as usize).map(|c| bf16_to_f64(x[t * cols as usize + c]).abs()).fold(0.0, f64::max);
        assert_eq!(scales[t], (amax as f32 / image_gemm::E4M3_MAX) as f64, "token {t}: scale is amax / 448");
    }

    for y_f32 in [false, true] {
        let width = if y_f32 { 4 } else { 2 };
        let dy = dev.allocate((tokens * rows) as u64 * width).expect("device allocation");
        let gemm = unsafe {
            image_gemm::gemm_w8a8(ptr(&dw), rows, cols, ptr(&dc), ptr(&ds) as *const f32, tokens, ptr(&dy), y_f32, std::ptr::null_mut())
        };
        if !ran(gemm, "ignis_image_gemm_w8a8") {
            return;
        }
        let y: Vec<f64> = download(&mut dev, &dy)
            .chunks(width as usize)
            .map(|b| if y_f32 { f32::from_le_bytes(b.try_into().unwrap()) as f64 } else { bf16_to_f64(u16::from_le_bytes(b.try_into().unwrap())) })
            .collect();
        for t in 0..tokens as usize {
            for r in 0..rows as usize {
                let (mut s, mut mag) = (0.0f64, 0.0f64);
                for c in 0..cols as usize {
                    let p = e4m3(codes[t * cols as usize + c]) * e4m3(weight[r * cols as usize + c]);
                    s += p;
                    mag += p.abs();
                }
                let scale = scales[t] * wscale[r];
                let reference = s * scale;
                // The CTest's bound (fp32-class accumulation per instruction) and the output's rounding.
                let tol = (cols as f64 / 32.0) * 35.0 * 2f64.powi(-25) * mag * scale.abs()
                    + reference.abs() * (2f64.powi(-23) + if y_f32 { 0.0 } else { 2f64.powi(-8) })
                    + 1e-30;
                let got = y[t * rows as usize + r];
                assert!((got - reference).abs() <= tol, "y[{t}][{r}] = {got}, fp64 {reference}, bound {tol} ({})", if y_f32 { "fp32" } else { "bf16" });
            }
        }
        dev.deallocate(dy).unwrap();
    }

    // A refusal comes back through the binding with the leaf's message, naming the op.
    let refused = unsafe {
        image_gemm::gemm_w8a8(ptr(&dw), rows, 96, ptr(&dc), ptr(&ds) as *const f32, tokens, ptr(&dx), false, std::ptr::null_mut())
    };
    let message = refused.expect_err("cols off a multiple of 64 must be refused");
    assert!(message.contains("ignis_image_gemm_w8a8") && message.contains("64"), "message: {message}");

    for buf in [dw, dx, dc, ds] {
        dev.deallocate(buf).unwrap();
    }
}
