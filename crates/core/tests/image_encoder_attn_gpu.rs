//! The image model's encoder attention driven through the Rust bindings (spec image-model/02
//! acceptance 5 and 9, GitHub #319): causal and non-causal calls, fp32 and BF16 output, Q/K/V read
//! from the fused QKV row and from separate buffers with their own strides, each checked against
//! fp64 GQA attention computed here. The kernel leaf's CTest (`ignis_kernel_image_encoder_attn_test`)
//! checks the op in depth (a tight mechanics bound from 1 to 41,000 tokens); this checks that the
//! bindings carry every argument to the right place, and that a refusal comes back with the leaf's
//! message.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing GPU or a
//! kernel error is a skip, under it a hard failure (`ignis_core::gpu_profile`). Run via
//! `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

use ignis_artifact::{CudaDevice, Device, DeviceBuffer};
use ignis_core::gpu_profile;
use ignis_core::image_encoder_attn::{self, Qkv};
use ignis_core::moe;

const HD: usize = 4096;
const KVD: usize = 1024;
const QKV: usize = 6144;
const D: usize = 128;
const H: usize = 32;
const GROUP: usize = 4;

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

/// Runs a binding and waits for it, under the profile's rule for a kernel error. The op runs on
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

fn bytes(x: &[u16]) -> Vec<u8> {
    x.iter().flat_map(|e| e.to_le_bytes()).collect()
}

/// fp64 attention of query `i`, head `h` (the fused row's values), over keys 0..visible.
fn reference(qkv: &[u16], i: usize, h: usize, visible: usize, scale: f64) -> Vec<f64> {
    let g = h / GROUP;
    let at = |t: usize, off: usize, d: usize| bf16_to_f64(qkv[t * QKV + off + d]);
    let scores: Vec<f64> = (0..visible).map(|j| (0..D).map(|d| at(i, h * D, d) * at(j, HD + g * D, d)).sum::<f64>() * scale).collect();
    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
    let sum: f64 = p.iter().sum();
    (0..D).map(|c| (0..visible).map(|j| p[j] * at(j, HD + KVD + g * D, c)).sum::<f64>() / sum).collect()
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn encoder_attention_through_the_bindings_matches_fp64() {
    let Some(mut dev) = device() else { return };
    let tokens = 150usize;
    let scale = 1.0f32 / (D as f32).sqrt();
    // q and k of about unit RMS (what the QK RMSNorm leaves), v with a per-channel offset.
    let qkv: Vec<u16> = (0..tokens * QKV)
        .map(|i| {
            let col = i % QKV;
            let x = 1.7 * hash_unit(1, i as u32);
            to_bf16(if col >= HD + KVD { x + ((col % D) % 5) as f32 * 0.5 } else { x })
        })
        .collect();
    // The same values as separate buffers: q [tokens][4096], k and v [tokens][1024].
    let split = |off: usize, width: usize| -> Vec<u16> { (0..tokens).flat_map(|t| qkv[t * QKV + off..t * QKV + off + width].to_vec()).collect() };
    let d_qkv = upload(&mut dev, &bytes(&qkv));
    let (d_q, d_k, d_v) = (upload(&mut dev, &bytes(&split(0, HD))), upload(&mut dev, &bytes(&split(HD, KVD))), upload(&mut dev, &bytes(&split(HD + KVD, KVD))));
    let o_stride = HD + 64;
    let o = dev.allocate((tokens * o_stride * 4) as u64).expect("device allocation");
    let elem = |buf: &DeviceBuffer, e: usize| unsafe { (ptr(buf) as *const u16).add(e) as *const c_void };
    let fused = Qkv { q: elem(&d_qkv, 0), k: elem(&d_qkv, HD), v: elem(&d_qkv, HD + KVD), q_row_stride: QKV as u32, kv_row_stride: QKV as u32 };
    let separate = Qkv { q: elem(&d_q, 0), k: elem(&d_k, 0), v: elem(&d_v, 0), q_row_stride: HD as u32, kv_row_stride: KVD as u32 };

    struct Case {
        what: &'static str,
        qkv: Qkv,
        causal: bool,
        f32_out: bool,
    }
    let cases = [
        Case { what: "causal, fused QKV row, fp32 out", qkv: fused, causal: true, f32_out: true },
        Case { what: "non-causal, fused QKV row, BF16 out", qkv: fused, causal: false, f32_out: false },
        Case { what: "causal, separate Q / K / V buffers, fp32 out", qkv: separate, causal: true, f32_out: true },
    ];
    for case in &cases {
        // Poison O (NaN in fp32 and in BF16) so an output the call fails to write is caught.
        dev.copy_h2d(&o, 0, &vec![0xFFu8; o.len() as usize]).expect("poison");
        dev.synchronize().expect("poison sync");
        let attended = unsafe { image_encoder_attn::attention(case.qkv, tokens as u32, case.causal, scale, ptr(&o), o_stride as u32, case.f32_out, std::ptr::null_mut()) };
        if !ran(attended, case.what) {
            return;
        }
        let out = download(&mut dev, &o);
        let got = |i: usize, h: usize, c: usize| -> f64 {
            let e = i * o_stride + h * D + c;
            if case.f32_out {
                f32::from_le_bytes(out[e * 4..e * 4 + 4].try_into().unwrap()) as f64
            } else {
                bf16_to_f64(u16::from_le_bytes([out[e * 2], out[e * 2 + 1]]))
            }
        };
        let mut worst_rel = 0.0f64;
        for h in 0..H {
            let (mut ee, mut ww) = (0.0f64, 0.0f64);
            for i in 0..tokens {
                let visible = if case.causal { i + 1 } else { tokens };
                let want = reference(&qkv, i, h, visible, scale as f64);
                for (c, w) in want.iter().enumerate() {
                    let g = got(i, h, c);
                    ee += (g - w) * (g - w);
                    ww += w * w;
                }
            }
            let rel = (ee / ww).sqrt();
            worst_rel = worst_rel.max(if rel.is_nan() { f64::INFINITY } else { rel });
        }
        // fp32 output sits ~5e-4 from fp64, BF16 output ~2e-3 (spec 02 "As built"); a misplaced
        // argument is an O(1) error.
        assert!(worst_rel <= 0.01, "{}: worst per-head relative RMS error {worst_rel}", case.what);
    }

    // A refusal comes back through the binding with the leaf's message, naming the op.
    let refused = unsafe { image_encoder_attn::attention(fused, tokens as u32, true, 0.0, ptr(&o), o_stride as u32, true, std::ptr::null_mut()) };
    let message = refused.expect_err("a zero scale must be refused");
    assert!(message.contains("ignis_image_encoder_attn") && message.contains("scale"), "message: {message}");

    for buf in [d_qkv, d_q, d_k, d_v, o] {
        dev.deallocate(buf).unwrap();
    }
}
