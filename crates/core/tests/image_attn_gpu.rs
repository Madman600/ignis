//! The image model's 8-bit DiT attention driven through the Rust bindings (spec image-model/02
//! acceptance 4 and 9, GitHub #319): quantize the queries, store a prefix run and a target run
//! (with the prefix's K mean), attend over both, over a truncated prefix, and causally, and check
//! the outputs against fp64 attention on the unquantized BF16 inputs computed here. The kernel
//! leaf's CTest (`ignis_kernel_image_attn_test`) checks the op in depth at the DiT's geometry
//! (mechanics against fp64 on the quantized operands, every call the DiT makes); this checks that
//! the bindings carry every argument to the right place, and that a refusal comes back with the
//! leaf's message.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing GPU or a
//! kernel error is a skip, under it a hard failure (`ignis_core::gpu_profile`). Run via
//! `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

use ignis_artifact::{CudaDevice, Device, DeviceBuffer};
use ignis_core::gpu_profile;
use ignis_core::image_attn::{self, RunA};
use ignis_core::moe;

const HD: usize = 4096;
const D: usize = 128;
const H: usize = 32;

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

/// A value in [-1, 1) from a counter hash.
fn hash_unit(stream: u32, i: u32) -> f32 {
    let h = lowbias32(i.wrapping_mul(0x9E37_79B9).wrapping_add(stream.wrapping_mul(0x85EB_CA6B)));
    (h >> 8) as f32 * (1.0f32 / 8_388_608.0) - 1.0
}

fn to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_to_f64(h: u16) -> f64 {
    f32::from_bits((h as u32) << 16) as f64
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// fp64 attention of query rows `q` over key rows `keys` (indices into the sequence) for head h.
fn reference(q: &[u16], k: &[u16], v: &[u16], qi: usize, keys: &[usize], h: usize, scale: f64) -> Vec<f64> {
    let scores: Vec<f64> = keys
        .iter()
        .map(|&j| (0..D).map(|d| bf16_to_f64(q[qi * HD + h * D + d]) * bf16_to_f64(k[j * HD + h * D + d])).sum::<f64>() * scale)
        .collect();
    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
    let sum: f64 = p.iter().sum();
    (0..D).map(|c| keys.iter().zip(&p).map(|(&j, pj)| pj * bf16_to_f64(v[j * HD + h * D + c])).sum::<f64>() / sum).collect()
}

/// The worst per-head cosine and relative RMS of `got` (token-major [rows][4096]) against the
/// fp64 references of `rows` x heads.
fn compare(got: &[f64], want: &[Vec<Vec<f64>>]) -> (f64, f64) {
    let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
    for h in 0..H {
        let (mut dot, mut gg, mut ww, mut ee) = (0.0, 0.0, 0.0, 0.0);
        for (i, rows) in want.iter().enumerate() {
            for c in 0..D {
                let (g, w) = (got[i * HD + h * D + c], rows[h][c]);
                dot += g * w;
                gg += g * g;
                ww += w * w;
                ee += (g - w) * (g - w);
            }
        }
        worst_cos = worst_cos.min(dot / (gg * ww).sqrt());
        worst_rel = worst_rel.max((ee / ww).sqrt());
    }
    (worst_cos, worst_rel)
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn attention_through_the_bindings_matches_fp64_of_its_inputs() {
    let Some(mut dev) = device() else { return };
    let (prefix, target) = (100usize, 50usize);
    let n = prefix + target;
    let scale = 1.0f32 / (D as f32).sqrt();
    // K with a per-channel offset (what smoothing removes), Q at a moderate temperature.
    let q: Vec<u16> = (0..n * HD).map(|i| to_bf16(2.0 * hash_unit(1, i as u32))).collect();
    let k: Vec<u16> = (0..n * HD).map(|i| to_bf16(hash_unit(2, i as u32) + ((i % D) % 7) as f32 * 0.5)).collect();
    let v: Vec<u16> = (0..n * HD).map(|i| to_bf16(hash_unit(3, i as u32) * (1.0 + (i % 5) as f32))).collect();
    let bytes = |x: &[u16]| x.iter().flat_map(|e| e.to_le_bytes()).collect::<Vec<u8>>();
    let (dq, dk, dv) = (upload(&mut dev, &bytes(&q)), upload(&mut dev, &bytes(&k)), upload(&mut dev, &bytes(&v)));
    let row = |buf: &DeviceBuffer, t: usize| unsafe { (ptr(buf) as *const u16).add(t * HD) as *const c_void };

    let q8 = dev.allocate(image_attn::q8_bytes(target as u32)).expect("device allocation");
    let kv_p = dev.allocate(image_attn::kv8_bytes(prefix as u32)).expect("device allocation");
    let kv_t = dev.allocate(image_attn::kv8_bytes(target as u32)).expect("device allocation");
    let o = dev.allocate((target * HD * 4) as u64).expect("device allocation");
    let null = std::ptr::null_mut();

    let stored = unsafe { image_attn::store_kv(row(&dk, 0), row(&dv, 0), prefix as u32, HD as u32, None, ptr(&kv_p), null) };
    if !ran(stored, "ignis_image_attn_store_kv (prefix)") {
        return;
    }
    let mean_from = Some((ptr(&kv_p) as *const c_void, prefix as u32));
    let stored = unsafe { image_attn::store_kv(row(&dk, prefix), row(&dv, prefix), target as u32, HD as u32, mean_from, ptr(&kv_t), null) };
    if !ran(stored, "ignis_image_attn_store_kv (target)") {
        return;
    }
    let quantized = unsafe { image_attn::quantize_q(row(&dq, prefix), target as u32, HD as u32, ptr(&q8), null) };
    if !ran(quantized, "ignis_image_attn_quantize_q") {
        return;
    }

    // The q8 buffer's first scale is head 0's first 32 target rows' amax / 127.
    let q8_bytes = download(&mut dev, &q8);
    let q_pad = target.div_ceil(128) * 128;
    let scale0 = f32::from_le_bytes(q8_bytes[H * q_pad * D..H * q_pad * D + 4].try_into().unwrap());
    let amax = (0..32).flat_map(|t| (0..D).map(move |d| (prefix + t) * HD + d)).map(|i| bf16_to_f64(q[i]).abs()).fold(0.0, f64::max);
    assert_eq!(scale0, (amax as f32) / 127.0, "q8: head 0's first scale is amax / 127");
    // The target run carries the prefix's K mean (header layout: K, V, K scales, V scales, mean).
    let mean_at = |tokens: usize| {
        let pad = tokens.div_ceil(64) * 64;
        2 * H * pad * D + H * (pad / 64) * 4 + HD * 4
    };
    let (rp, rt) = (download(&mut dev, &kv_p), download(&mut dev, &kv_t));
    assert_eq!(&rp[mean_at(prefix)..mean_at(prefix) + HD * 4], &rt[mean_at(target)..mean_at(target) + HD * 4], "the target run carries the prefix's mean");
    let mean = f32s(&rp[mean_at(prefix)..mean_at(prefix) + HD * 4]);
    let want_mean = (0..prefix).map(|t| bf16_to_f64(k[t * HD + 3])).sum::<f64>() / prefix as f64;
    assert!((mean[3] as f64 - want_mean).abs() < 1e-4, "the prefix's K mean, channel 3: {} vs {want_mean}", mean[3]);

    // The target over [prefix ; target], fp32 out; then causally (BF16 out); then the first 70
    // prefix keys alone, as a prefix segment's call reads them.
    struct Case {
        what: &'static str,
        a_keys: usize,
        with_b: bool,
        causal: bool,
        f32_out: bool,
    }
    let cases = [
        Case { what: "target over [prefix ; target]", a_keys: prefix, with_b: true, causal: false, f32_out: true },
        Case { what: "target causal over [prefix ; target]", a_keys: prefix, with_b: true, causal: true, f32_out: false },
        Case { what: "target over the first 70 prefix keys", a_keys: 70, with_b: false, causal: false, f32_out: true },
    ];
    for case in &cases {
        let a = RunA { kv8: ptr(&kv_p), tokens: prefix as u32, keys: case.a_keys as u32 };
        let b = case.with_b.then(|| (ptr(&kv_t) as *const c_void, target as u32));
        let attended = unsafe { image_attn::attention(ptr(&q8), target as u32, a, b, case.causal, scale, ptr(&o), HD as u32, case.f32_out, null) };
        if !ran(attended, case.what) {
            return;
        }
        let out = download(&mut dev, &o);
        let got: Vec<f64> = if case.f32_out {
            f32s(&out[..target * HD * 4]).into_iter().map(f64::from).collect()
        } else {
            out[..target * HD * 2].chunks(2).map(|b| bf16_to_f64(u16::from_le_bytes([b[0], b[1]]))).collect()
        };
        let total = case.a_keys + if case.with_b { target } else { 0 };
        let want: Vec<Vec<Vec<f64>>> = (0..target)
            .map(|i| {
                let visible = if case.causal { (i + total - target + 1).min(total) } else { total };
                let keys: Vec<usize> = (0..visible).map(|j| if j < case.a_keys { j } else { prefix + (j - case.a_keys) }).collect();
                (0..H).map(|h| reference(&q, &k, &v, prefix + i, &keys, h, scale as f64)).collect()
            })
            .collect();
        let (cos, rel) = compare(&got, &want);
        assert!(cos >= 0.99 && rel <= 0.1, "{}: worst per-head cosine {cos}, relative RMS {rel}", case.what);
    }

    // A refusal comes back through the binding with the leaf's message, naming the op.
    let a = RunA { kv8: ptr(&kv_p), tokens: prefix as u32, keys: 0 };
    let refused = unsafe { image_attn::attention(ptr(&q8), target as u32, a, None, false, scale, ptr(&o), HD as u32, true, null) };
    let message = refused.expect_err("a call reading no key of run A must be refused");
    assert!(message.contains("ignis_image_attn") && message.contains("a_keys"), "message: {message}");

    for buf in [dq, dk, dv, q8, kv_p, kv_t, o] {
        dev.deallocate(buf).unwrap();
    }
}
