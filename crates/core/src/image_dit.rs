//! The image model's DiT elementwise ops in the kernel leaf (spec image-model/02 acceptance 7,
//! GitHub #319): the flat C ABI of `kernel/include/ignis_image_dit.h`, 1:1, and thin wrappers
//! that turn its return codes into `Result`s carrying the leaf's own message.
//!
//! The ops are the image leaf's DiT (spec image-model/04), which calls them inside the kernel leaf:
//! the 3-axis RoPE positions (host) and table, the modulated LayerNorm with its per-token t / t=0
//! row, the zero-centred RMSNorm, the gated residual, the QK RMSNorm with RoPE, and SiLU,
//! GELU-tanh and SwiGLU. The row ops can emit the FP8 GEMM's E4M3 input directly. These bindings
//! exist for Rust-side tests and tools that drive them directly. Every device pointer argument is
//! one the caller vouches for, so the device wrappers are `unsafe`; the shapes and alignments they
//! take are checked by the leaf.
//!
//! The device ops are asynchronous on the stream they are given (null: the legacy default stream)
//! and carry no synchronization: a caller that moves data through `CudaDevice`'s own stream orders
//! the copies itself (`moe::stream_sync(null)` after an op, as `tests/image_elementwise_gpu.rs`
//! does).
//!
//! Our own implementation (ADR 0051, tier 1).

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;

/// 1:1 with `kernel/include/ignis_image_dit.h`.
pub mod ffi {
    use std::os::raw::{c_char, c_void};

    unsafe extern "C" {
        pub fn ignis_image_dit_rope_positions(
            tokens: u32,
            blocks: u32,
            block_start: *const u32,
            block_height: *const u32,
            block_width: *const u32,
            positions: *mut i32,
        ) -> i32;
        pub fn ignis_image_dit_rope_table(positions: *const i32, tokens: u32, table: *mut f32, stream: *mut c_void) -> i32;
        pub fn ignis_image_dit_modulated_norm(
            x: *const c_void,
            tokens: u32,
            cols: u32,
            scale_t: *const f32,
            scale_0: *const f32,
            first_target: u32,
            eps: f32,
            y: *mut c_void,
            y_scales: *mut f32,
            y_kind: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_dit_rms_norm(
            x: *const c_void,
            tokens: u32,
            cols: u32,
            weight: *const c_void,
            eps: f32,
            y: *mut c_void,
            y_scales: *mut f32,
            y_kind: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_dit_gated_residual(
            x: *const c_void,
            y: *const c_void,
            tokens: u32,
            cols: u32,
            gate_t: *const f32,
            gate_0: *const f32,
            first_target: u32,
            out: *mut c_void,
            out_f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_dit_qk_norm_rope(
            q: *const c_void,
            k: *const c_void,
            tokens: u32,
            row_stride: u32,
            q_weight: *const c_void,
            k_weight: *const c_void,
            eps: f32,
            rope: *const f32,
            q_out: *mut c_void,
            k_out: *mut c_void,
            out_row_stride: u32,
            out_f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_dit_act(
            kind: u32,
            x: *const c_void,
            up: *const c_void,
            tokens: u32,
            cols: u32,
            row_stride: u32,
            y: *mut c_void,
            y_scales: *mut f32,
            y_kind: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_dit_last_error() -> *const c_char;
    }
}

/// The leaf's message for the most recent failed DiT elementwise call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(ffi::ignis_image_dit_last_error()) }.to_string_lossy().into_owned()
}

fn check(rc: i32) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// A row op's output (the header's `IGNIS_IMAGE_DIT_OUT_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// BF16 `[tokens][cols]`.
    Bf16 = 0,
    /// fp32 `[tokens][cols]`.
    F32 = 1,
    /// E4M3 codes `[tokens][cols]` plus fp32 scales `[tokens]`: `ignis_image_quantize_tokens` of
    /// the BF16 output, bit for bit.
    E4m3 = 2,
}

/// The activations (the header's `IGNIS_IMAGE_DIT_SILU` / `_GELU_TANH` / `_SWIGLU`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Silu = 0,
    GeluTanh = 1,
    /// `silu(x) * up`.
    Swiglu = 2,
}

/// The RoPE positions `[tokens][3]` (frame, height, width) of a joint sequence whose image block
/// `b` is `block_hw[b]` = (height, width) latent tokens starting at token `block_start[b]`, blocks
/// in sequence order (diffusers' `QwenImage21Rope`).
pub fn rope_positions(tokens: u32, block_start: &[u32], block_hw: &[(u32, u32)]) -> Result<Vec<i32>, String> {
    if block_start.len() != block_hw.len() {
        return Err("rope_positions: one start per block".to_string());
    }
    let h: Vec<u32> = block_hw.iter().map(|b| b.0).collect();
    let w: Vec<u32> = block_hw.iter().map(|b| b.1).collect();
    let mut positions = vec![0i32; 3 * tokens as usize];
    check(unsafe {
        ffi::ignis_image_dit_rope_positions(
            tokens,
            block_start.len() as u32,
            block_start.as_ptr(),
            h.as_ptr(),
            w.as_ptr(),
            positions.as_mut_ptr(),
        )
    })?;
    Ok(positions)
}

/// The RoPE table `[tokens][64][2]` fp32 (cos, sin) from device `positions [tokens][3]`.
///
/// # Safety
/// Both pointers are device pointers of the size `tokens` implies.
pub unsafe fn rope_table(positions: *const i32, tokens: u32, table: *mut f32, stream: *mut c_void) -> Result<(), String> {
    check(unsafe { ffi::ignis_image_dit_rope_table(positions, tokens, table, stream) })
}

/// `y = LayerNorm(x) * (1 + s)`, `s = scale_0` for tokens below `first_target`, else `scale_t`;
/// `y_scales` is null unless `out` is [`Out::E4m3`].
///
/// # Safety
/// Every non-null pointer is a device pointer of the size the shape implies.
#[allow(clippy::too_many_arguments)]
pub unsafe fn modulated_norm(
    x: *const c_void,
    tokens: u32,
    cols: u32,
    scale_t: *const f32,
    scale_0: *const f32,
    first_target: u32,
    eps: f32,
    y: *mut c_void,
    y_scales: *mut f32,
    out: Out,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe {
        ffi::ignis_image_dit_modulated_norm(x, tokens, cols, scale_t, scale_0, first_target, eps, y, y_scales, out as u32, stream)
    })
}

/// `y = (x * rsqrt(mean(x^2) + eps)) * (w + 1)`, `weight` BF16 `[cols]`.
///
/// # Safety
/// Every non-null pointer is a device pointer of the size the shape implies.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rms_norm(
    x: *const c_void,
    tokens: u32,
    cols: u32,
    weight: *const c_void,
    eps: f32,
    y: *mut c_void,
    y_scales: *mut f32,
    out: Out,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_image_dit_rms_norm(x, tokens, cols, weight, eps, y, y_scales, out as u32, stream) })
}

/// `out = x + tanh(g) * y`, `g = gate_0` for tokens below `first_target`, else `gate_t`; `out`
/// may be `x` (BF16).
///
/// # Safety
/// Every non-null pointer is a device pointer of the size the shape implies.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gated_residual(
    x: *const c_void,
    y: *const c_void,
    tokens: u32,
    cols: u32,
    gate_t: *const f32,
    gate_0: *const f32,
    first_target: u32,
    out: *mut c_void,
    out_f32: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe {
        ffi::ignis_image_dit_gated_residual(x, y, tokens, cols, gate_t, gate_0, first_target, out, out_f32 as u32, stream)
    })
}

/// The QK RMSNorm with RoPE on 32 heads of 128 of `q` and `k` (rows at `row_stride` elements),
/// written at `q_out` / `k_out` (rows at `out_row_stride`); the in-place form passes `q`, `k` and
/// `row_stride` again with BF16 output.
///
/// # Safety
/// Every pointer is a device pointer of the size the shapes imply.
#[allow(clippy::too_many_arguments)]
pub unsafe fn qk_norm_rope(
    q: *const c_void,
    k: *const c_void,
    tokens: u32,
    row_stride: u32,
    q_weight: *const c_void,
    k_weight: *const c_void,
    eps: f32,
    rope: *const f32,
    q_out: *mut c_void,
    k_out: *mut c_void,
    out_row_stride: u32,
    out_f32: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe {
        ffi::ignis_image_dit_qk_norm_rope(
            q,
            k,
            tokens,
            row_stride,
            q_weight,
            k_weight,
            eps,
            rope,
            q_out,
            k_out,
            out_row_stride,
            out_f32 as u32,
            stream,
        )
    })
}

/// The activation `act` of `x` (and `up` for [`Act::Swiglu`], else null), rows at `row_stride`.
///
/// # Safety
/// Every non-null pointer is a device pointer of the size the shape implies.
#[allow(clippy::too_many_arguments)]
pub unsafe fn act(
    act: Act,
    x: *const c_void,
    up: *const c_void,
    tokens: u32,
    cols: u32,
    row_stride: u32,
    y: *mut c_void,
    y_scales: *mut f32,
    out: Out,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_image_dit_act(act as u32, x, up, tokens, cols, row_stride, y, y_scales, out as u32, stream) })
}
