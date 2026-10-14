//! The image model's 8-bit DiT attention in the kernel leaf (spec image-model/02 acceptance 4,
//! GitHub #319): the flat C ABI of `kernel/include/ignis_image_attn.h`, 1:1, and thin wrappers
//! that turn its return codes into `Result`s carrying the leaf's own message.
//!
//! The ops are the image leaf's (spec image-model/04), which calls them inside the kernel leaf:
//! quantize the queries, store a K/V run (the condition prefix once, the target every step with
//! the prefix's K mean), and attend over up to two runs. These bindings exist for Rust-side
//! tests and tools that drive them directly. Every pointer argument is a device pointer the
//! caller vouches for, so the wrappers are `unsafe`; the shapes and alignments they take are
//! checked by the leaf.
//!
//! The ops are asynchronous on the stream they are given (null: the legacy default stream) and
//! carry no synchronization. `CudaDevice`'s copies run on its own non-blocking load stream, which
//! never waits for the legacy stream, so a caller that uploads inputs or reads outputs through
//! `CudaDevice` orders the copies itself: `moe::stream_sync(null)` after an op, before the copy
//! that reads its output (as `tests/image_attn_gpu.rs` does).
//!
//! An adapted op (ADR 0051, tier 2): the kernel adapts SageAttention2++ (Apache-2.0); provenance
//! in `kernel/NOTICE`.

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;

/// The DiT's heads and head size, the only geometry the op takes.
pub const HEADS: u32 = 32;
pub const HEAD_DIM: u32 = 128;

/// 1:1 with `kernel/include/ignis_image_attn.h`.
pub mod ffi {
    use std::os::raw::{c_char, c_void};

    unsafe extern "C" {
        pub fn ignis_image_attn_q8_bytes(tokens: u32) -> u64;
        pub fn ignis_image_attn_kv8_bytes(tokens: u32) -> u64;
        pub fn ignis_image_attn_quantize_q(
            q: *const c_void,
            tokens: u32,
            row_stride: u32,
            q8: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_attn_store_kv(
            k: *const c_void,
            v: *const c_void,
            tokens: u32,
            row_stride: u32,
            mean_from: *const c_void,
            mean_from_tokens: u32,
            kv8: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_attn(
            q8: *const c_void,
            q_tokens: u32,
            kv_a: *const c_void,
            a_tokens: u32,
            a_keys: u32,
            kv_b: *const c_void,
            b_tokens: u32,
            causal: u32,
            scale: f32,
            o: *mut c_void,
            o_row_stride: u32,
            o_f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_attn_last_error() -> *const c_char;
    }
}

/// The leaf's message for the most recent failed image-attention call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(ffi::ignis_image_attn_last_error()) }.to_string_lossy().into_owned()
}

fn check(rc: i32) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// Bytes of a q8 buffer for `tokens` queries.
pub fn q8_bytes(tokens: u32) -> u64 {
    unsafe { ffi::ignis_image_attn_q8_bytes(tokens) }
}

/// Bytes of a K/V run of `tokens` keys.
pub fn kv8_bytes(tokens: u32) -> u64 {
    unsafe { ffi::ignis_image_attn_kv8_bytes(tokens) }
}

/// BF16 queries (token `t`'s head `h` at element `t * row_stride + h * 128`) into a q8 buffer.
///
/// # Safety
/// Every pointer is a device pointer of the size `tokens` and `row_stride` imply.
pub unsafe fn quantize_q(q: *const c_void, tokens: u32, row_stride: u32, q8: *mut c_void, stream: *mut c_void) -> Result<(), String> {
    check(unsafe { ffi::ignis_image_attn_quantize_q(q, tokens, row_stride, q8, stream) })
}

/// The K/V run of `tokens` BF16 keys and values, smoothed with its own K mean (`mean_from`
/// `None`) or with the mean stored in another run of `mean_from.1` tokens (the prefix's, for the
/// target).
///
/// # Safety
/// Every pointer is a device pointer of the size the token counts and `row_stride` imply.
pub unsafe fn store_kv(
    k: *const c_void,
    v: *const c_void,
    tokens: u32,
    row_stride: u32,
    mean_from: Option<(*const c_void, u32)>,
    kv8: *mut c_void,
    stream: *mut c_void,
) -> Result<(), String> {
    let (from, from_tokens) = mean_from.unwrap_or((std::ptr::null(), 0));
    check(unsafe { ffi::ignis_image_attn_store_kv(k, v, tokens, row_stride, from, from_tokens, kv8, stream) })
}

/// Run A as an attention call reads it: the run, its token count, and how many of its first keys
/// the call reads (all of them for the target; a prefix segment's end for the prefix's calls).
#[derive(Clone, Copy, Debug)]
pub struct RunA {
    pub kv8: *const c_void,
    pub tokens: u32,
    pub keys: u32,
}

/// The attention of a q8 buffer's `q_tokens` queries over `a`'s first `a.keys` keys then all keys
/// of run `b` (the run and its token count); `causal` aligns the diagonal bottom-right. Output
/// token-major with `o_row_stride` elements per token, fp32 when `o_f32`, else BF16.
///
/// # Safety
/// Every pointer is a device pointer of the size the token counts and `o_row_stride` imply.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention(
    q8: *const c_void,
    q_tokens: u32,
    a: RunA,
    b: Option<(*const c_void, u32)>,
    causal: bool,
    scale: f32,
    o: *mut c_void,
    o_row_stride: u32,
    o_f32: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    let (b_ptr, b_tokens) = b.unwrap_or((std::ptr::null(), 0));
    check(unsafe {
        ffi::ignis_image_attn(q8, q_tokens, a.kv8, a.tokens, a.keys, b_ptr, b_tokens, causal as u32, scale, o, o_row_stride, o_f32 as u32, stream)
    })
}
