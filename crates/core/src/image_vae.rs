//! The image model's VAE ops in the kernel leaf (spec image-model/02 acceptance 6 and 7, GitHub
//! #319): the flat C ABI of `kernel/include/ignis_image_vae.h`, 1:1, and thin wrappers that turn
//! its return codes into `Result`s carrying the leaf's own message.
//!
//! The ops are the image leaf's VAE (spec image-model/04), which calls them inside the kernel
//! leaf: the convolution (an implicit GEMM on BF16 tensor cores, with row windows for banded
//! decoding), the residual shortcuts (AvgDown / DupUp), the one-head mid-block attention and the
//! channel RMS norm with its SiLU. These
//! bindings exist for Rust-side tests and tools that drive them directly. Every pointer argument is
//! a device pointer the caller vouches for, so the wrappers are `unsafe`; the shapes and alignments
//! they take are checked by the leaf.
//!
//! The ops are asynchronous on the stream they are given (null: the legacy default stream) and
//! carry no synchronization. `CudaDevice`'s copies run on its own non-blocking load stream, which
//! never waits for the legacy stream, so a caller that uploads inputs or reads outputs through
//! `CudaDevice` orders the copies itself: `moe::stream_sync(null)` after an op, before the copy
//! that reads its output (as `tests/image_vae_gpu.rs` does).
//!
//! Our own implementation (ADR 0051, tier 1).

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;

/// 1:1 with `kernel/include/ignis_image_vae.h`.
pub mod ffi {
    use std::os::raw::{c_char, c_void};

    unsafe extern "C" {
        pub fn ignis_image_vae_pack_conv_weight(
            w: *const c_void,
            out_channels: u32,
            in_channels: u32,
            kernel: u32,
            w_packed: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_vae_conv(
            kind: u32,
            upsample: u32,
            x: *const c_void,
            height: u32,
            width: u32,
            in_channels: u32,
            x_row0: u32,
            x_rows: u32,
            weight: *const c_void,
            bias: *const c_void,
            out_channels: u32,
            residual: *const c_void,
            y: *mut c_void,
            y_row0: u32,
            y_rows: u32,
            y_f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_vae_shortcut(
            kind: u32,
            x: *const c_void,
            height: u32,
            width: u32,
            in_channels: u32,
            out_channels: u32,
            factor_t: u32,
            factor_s: u32,
            y: *mut c_void,
            accumulate: u32,
            f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_vae_attn_workspace_bytes(tokens: u32, dim: u32, row_tile: u32) -> u64;
        pub fn ignis_image_vae_attn(
            qkv: *const c_void,
            tokens: u32,
            dim: u32,
            scale: f32,
            row_tile: u32,
            workspace: *mut c_void,
            workspace_bytes: u64,
            o: *mut c_void,
            o_f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_vae_rms_norm(
            x: *const c_void,
            pixels: u32,
            channels: u32,
            gamma: *const c_void,
            silu: u32,
            y: *mut c_void,
            y_f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_image_vae_last_error() -> *const c_char;
    }
}

/// The leaf's message for the most recent failed VAE call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(ffi::ignis_image_vae_last_error()) }.to_string_lossy().into_owned()
}

fn check(rc: i32) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// The VAE's convolution kinds (the header's `IGNIS_IMAGE_VAE_CONV_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvKind {
    /// 3 x 3, stride 1, zero padding 1.
    K3x3 = 0,
    /// 3 x 3, stride 2, after zero padding (0, 1, 0, 1).
    K3x3S2 = 1,
    /// 1 x 1.
    K1x1 = 2,
}

/// The shortcut kinds (the header's `IGNIS_IMAGE_VAE_AVG_DOWN` / `IGNIS_IMAGE_VAE_DUP_UP`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortcutKind {
    AvgDown = 0,
    DupUp = 1,
}

/// One convolution call: BF16 NHWC `x` holds input rows `x_row0 .. x_row0 + x_rows` of a
/// `height` x `width` image (before the upsample), and the call writes output rows `y_row0 ..
/// y_row0 + y_rows` into `y` (fp32 when `y_f32`, else BF16), adding `bias` and `residual` (laid
/// out as `y`) when they are not null. `weight` is packed by [`pack_conv_weight`].
#[derive(Clone, Copy, Debug)]
pub struct Conv {
    pub kind: ConvKind,
    pub upsample: bool,
    pub x: *const c_void,
    pub height: u32,
    pub width: u32,
    pub in_channels: u32,
    pub x_row0: u32,
    pub x_rows: u32,
    pub weight: *const c_void,
    pub bias: *const c_void,
    pub out_channels: u32,
    pub residual: *const c_void,
    pub y: *mut c_void,
    pub y_row0: u32,
    pub y_rows: u32,
    pub y_f32: bool,
}

/// A BF16 weight `[out][in][kernel][kernel]` repacked to `[out][kernel][kernel][in]`, the layout
/// [`conv`] reads (kernel 1 or 3).
///
/// # Safety
/// Both pointers are device pointers of the size the shape implies, not overlapping.
pub unsafe fn pack_conv_weight(
    w: *const c_void,
    out_channels: u32,
    in_channels: u32,
    kernel: u32,
    w_packed: *mut c_void,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_image_vae_pack_conv_weight(w, out_channels, in_channels, kernel, w_packed, stream) })
}

/// `y = conv(x) + bias (+ residual)` over the call's windows (the header's semantics).
///
/// # Safety
/// Every non-null pointer is a device pointer of the size the call's shape and windows imply.
pub unsafe fn conv(c: &Conv, stream: *mut c_void) -> Result<(), String> {
    check(unsafe {
        ffi::ignis_image_vae_conv(
            c.kind as u32,
            c.upsample as u32,
            c.x,
            c.height,
            c.width,
            c.in_channels,
            c.x_row0,
            c.x_rows,
            c.weight,
            c.bias,
            c.out_channels,
            c.residual,
            c.y,
            c.y_row0,
            c.y_rows,
            c.y_f32 as u32,
            stream,
        )
    })
}

/// `y` (+)= the shortcut of NHWC `x [height][width][in_channels]`, BF16 or fp32 (`f32`): AvgDown
/// writes `[height / factor_s][width / factor_s][out_channels]`, DupUp `[height * factor_s][width
/// * factor_s][out_channels]`; `accumulate` adds it into `y`.
///
/// # Safety
/// Both pointers are device pointers of the size the shape implies, not overlapping.
#[allow(clippy::too_many_arguments)]
pub unsafe fn shortcut(
    kind: ShortcutKind,
    x: *const c_void,
    height: u32,
    width: u32,
    in_channels: u32,
    out_channels: u32,
    factor_t: u32,
    factor_s: u32,
    y: *mut c_void,
    accumulate: bool,
    f32: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe {
        ffi::ignis_image_vae_shortcut(
            kind as u32,
            x,
            height,
            width,
            in_channels,
            out_channels,
            factor_t,
            factor_s,
            y,
            accumulate as u32,
            f32 as u32,
            stream,
        )
    })
}

/// The mid-block attention's workspace in bytes for `tokens`, `dim` and `row_tile`; 0 when the
/// arguments are out of range.
pub fn attn_workspace_bytes(tokens: u32, dim: u32, row_tile: u32) -> u64 {
    unsafe { ffi::ignis_image_vae_attn_workspace_bytes(tokens, dim, row_tile) }
}

/// `O [tokens][dim] = softmax(scale Q K^T) V` with one head, from to_qkv's output `qkv
/// [tokens][3 dim]` BF16, in tiles of `row_tile` query rows through `workspace`; fp32 output when
/// `o_f32`, else BF16.
///
/// # Safety
/// Every pointer is a device pointer of the size the shapes imply (the workspace at least
/// [`attn_workspace_bytes`]); `o` overlaps neither `qkv` nor the workspace.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn(
    qkv: *const c_void,
    tokens: u32,
    dim: u32,
    scale: f32,
    row_tile: u32,
    workspace: *mut c_void,
    workspace_bytes: u64,
    o: *mut c_void,
    o_f32: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe {
        ffi::ignis_image_vae_attn(qkv, tokens, dim, scale, row_tile, workspace, workspace_bytes, o, o_f32 as u32, stream)
    })
}

/// The channel RMS norm of NHWC BF16 `x [pixels][channels]` (`gamma` BF16 `[channels]`), then SiLU
/// when `silu`; fp32 output when `y_f32`, else BF16, and `y` may be `x`.
///
/// # Safety
/// Every pointer is a device pointer of the size the shape implies.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rms_norm(
    x: *const c_void,
    pixels: u32,
    channels: u32,
    gamma: *const c_void,
    silu: bool,
    y: *mut c_void,
    y_f32: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_image_vae_rms_norm(x, pixels, channels, gamma, silu as u32, y, y_f32 as u32, stream) })
}
