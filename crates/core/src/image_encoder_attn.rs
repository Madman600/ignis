//! The image model's encoder attention in the kernel leaf (spec image-model/02 acceptance 5,
//! GitHub #319): the flat C ABI of `kernel/include/ignis_image_encoder_attn.h`, 1:1, and a thin
//! wrapper that turns its return code into a `Result` carrying the leaf's own message.
//!
//! The op is the text encoder's (spec image-model/03), which calls it inside the kernel leaf: one
//! causal sequence, 32 query heads over 8 KV heads of 128, BF16, read from the fused QKV row after
//! the QK norm and mrope applied to it in place. These bindings exist for Rust-side tests and tools
//! that drive it directly. Every pointer argument is a device pointer the caller vouches for, so
//! the wrapper is `unsafe`; the shapes and alignments it takes are checked by the leaf.
//!
//! The op is asynchronous on the stream it is given (null: the legacy default stream) and carries
//! no synchronization. `CudaDevice`'s copies run on its own non-blocking load stream, which never
//! waits for the legacy stream, so a caller that uploads inputs or reads outputs through
//! `CudaDevice` orders the copies itself: `moe::stream_sync(null)` after the op, before the copy
//! that reads its output (as `tests/image_encoder_attn_gpu.rs` does).
//!
//! An adapted op (ADR 0051, tier 2): the kernel adapts FlashAttention-2's forward (BSD-3-Clause);
//! provenance in `kernel/NOTICE`.

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;

/// The encoder's query heads, KV heads and head size, the only geometry the op takes.
pub const HEADS: u32 = 32;
pub const KV_HEADS: u32 = 8;
pub const HEAD_DIM: u32 = 128;

/// 1:1 with `kernel/include/ignis_image_encoder_attn.h`.
pub mod ffi {
    use std::os::raw::{c_char, c_void};

    unsafe extern "C" {
        pub fn ignis_image_encoder_attn(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            tokens: u32,
            q_row_stride: u32,
            kv_row_stride: u32,
            causal: u32,
            scale: f32,
            o: *mut c_void,
            o_row_stride: u32,
            o_f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_encoder_attn_last_error() -> *const c_char;
    }
}

/// The leaf's message for the most recent failed encoder-attention call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(ffi::ignis_image_encoder_attn_last_error()) }.to_string_lossy().into_owned()
}

/// Where the call reads Q, K and V: token `t`'s query head `h` at element `t * q_row_stride + h *
/// 128` of `q`, its KV head `g` at `t * kv_row_stride + g * 128` of `k` and of `v`. The fused QKV
/// row is `q = qkv`, `k = qkv + 4096`, `v = qkv + 5120` elements, both strides 6144.
#[derive(Clone, Copy, Debug)]
pub struct Qkv {
    pub q: *const c_void,
    pub k: *const c_void,
    pub v: *const c_void,
    pub q_row_stride: u32,
    pub kv_row_stride: u32,
}

/// O = softmax(`scale` Q K^T) V over one sequence of `tokens`, causal when `causal`. Output
/// token-major with `o_row_stride` elements per token, fp32 when `o_f32`, else BF16.
///
/// # Safety
/// Every pointer is a device pointer of the size `tokens` and the strides imply; `o` does not
/// overlap Q, K or V.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention(
    qkv: Qkv,
    tokens: u32,
    causal: bool,
    scale: f32,
    o: *mut c_void,
    o_row_stride: u32,
    o_f32: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    let rc = unsafe {
        ffi::ignis_image_encoder_attn(
            qkv.q,
            qkv.k,
            qkv.v,
            tokens,
            qkv.q_row_stride,
            qkv.kv_row_stride,
            causal as u32,
            scale,
            o,
            o_row_stride,
            o_f32 as u32,
            stream,
        )
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}
