//! The image model's FP8 W8A8 GEMM and its per-token activation quantization in the kernel leaf
//! (spec image-model/02, GitHub #319): the flat C ABI of `kernel/include/ignis_image_gemm.h`,
//! 1:1, and thin wrappers that turn its return codes into `Result`s carrying the leaf's own
//! message.
//!
//! The ops are the image leaf's (spec image-model/04), which calls them inside the kernel leaf;
//! these bindings exist for Rust-side tests and tools that drive them directly. Every pointer
//! argument is a device pointer the caller vouches for, so the wrappers are `unsafe`; the shapes
//! and alignments they take are checked by the leaf.
//!
//! The ops are asynchronous on the stream they are given (null: the legacy default stream) and
//! carry no synchronization. `CudaDevice`'s copies run on its own non-blocking load stream, which
//! never waits for the legacy stream, so a caller that uploads inputs or reads outputs through
//! `CudaDevice` orders the copies itself: `moe::stream_sync(null)` after an op, before the copy
//! that reads its output (as `tests/image_gemm_gpu.rs` does).
//!
//! Our own implementation (ADR 0051, tier 1).

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;

/// E4M3FN's largest finite magnitude: a token's amax maps to it (`scales[t] = amax / 448`).
pub const E4M3_MAX: f32 = 448.0;

/// 1:1 with `kernel/include/ignis_image_gemm.h`.
pub mod ffi {
    use std::os::raw::{c_char, c_void};

    unsafe extern "C" {
        pub fn ignis_image_gemm_prepare() -> i32;
        pub fn ignis_image_quantize_tokens(
            x: *const c_void,
            tokens: u32,
            cols: u32,
            codes: *mut c_void,
            scales: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_gemm_w8a8(
            weight: *const c_void,
            rows: u32,
            cols: u32,
            codes: *const c_void,
            scales: *const f32,
            tokens: u32,
            y: *mut c_void,
            y_f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_gemm_last_error() -> *const c_char;
    }
}

/// The leaf's message for the most recent failed image GEMM call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(ffi::ignis_image_gemm_last_error()) }.to_string_lossy().into_owned()
}

fn check(rc: i32) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// Prepares the current device for the GEMM (kernel attributes); once per device at load,
/// outside any stream capture. The GEMM refuses to run on a device that was not prepared.
pub fn prepare() -> Result<(), String> {
    check(unsafe { ffi::ignis_image_gemm_prepare() })
}

/// BF16 `x [tokens][cols]` to E4M3 `codes [tokens][cols]` and fp32 `scales [tokens]`, per token.
///
/// # Safety
/// Every pointer is a device pointer of the size `tokens` and `cols` imply.
pub unsafe fn quantize_tokens(
    x: *const c_void,
    tokens: u32,
    cols: u32,
    codes: *mut c_void,
    scales: *mut f32,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_image_quantize_tokens(x, tokens, cols, codes, scales, stream) })
}

/// `y [tokens][rows] = (scales[t] * sum_c codes[t][c] * w[r][c]) * wscale[r]` for a row-scale-v1
/// FP8 weight `[rows][cols]`; `y_f32` selects fp32 output, else BF16.
///
/// # Safety
/// Every pointer is a device pointer of the size the shapes imply.
pub unsafe fn gemm_w8a8(
    weight: *const c_void,
    rows: u32,
    cols: u32,
    codes: *const c_void,
    scales: *const f32,
    tokens: u32,
    y: *mut c_void,
    y_f32: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_image_gemm_w8a8(weight, rows, cols, codes, scales, tokens, y, y_f32 as u32, stream) })
}
