//! The image leaf with ten condition images (spec image-model/04, GitHub #321): the prefix past
//! what one pass fits in the arena.
//!
//! Ten copies of the oracle's edit condition image (1024², 1,024 image pads each) go through
//! `EncoderFrontend` and `encode_condition`: ten VAE encodes, the tower on 40,960 patches, the
//! encoder on the 10,335-token sequence with its hooks, and a condition prefix of 40,960 latent
//! tokens plus the text rows, within the condition budget. The whole prefix pass does not fit the
//! arena, so it must run in chunks (`Condition::prefix_chunk`), which the transformer's CTest shows
//! equal to the whole pass bit for bit. Then one denoising step of a 64 x 64 target over it: a
//! finite velocity. The stages' wall times are printed.
//!
//! The artifact is `$IGNIS_IMAGE_ARTIFACT`, else
//! `F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer`; the oracle
//! `$IGNIS_IMAGE_ORACLE`, else `F:/ai/models/ignis-image-2.1-turbo/oracle`. Explicit GPU profile
//! (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing GPU, fixture or a kernel error is a skip,
//! under it a hard failure (`ignis_core::gpu_profile`). Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::{CudaDevice, Reader};
use ignis_core::gpu_profile;
use ignis_core::image_leaf::{ConditionImage as LeafImage, ConditionText, ImageLeaf, ImageLeafError, ImageLeafOptions};
use ignis_core::image_vision;
use ignis_core::vram::VramMode;
use ignis_server::image_encoder_input::{ConditionImage, EncoderFrontend};

/// The server's default VRAM headroom (`vram.headroom_bytes`).
const HEADROOM: u64 = 1536 << 20;

fn machine_local(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| default.to_string()))
}

/// The artifact and the oracle's edit case, or None after skip_or_fail.
fn fixtures() -> Option<(Reader, PathBuf)> {
    let artifact = machine_local("IGNIS_IMAGE_ARTIFACT", "F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer");
    let oracle = machine_local("IGNIS_IMAGE_ORACLE", "F:/ai/models/ignis-image-2.1-turbo/oracle");
    if !artifact.is_file() {
        gpu_profile::skip_or_fail(&format!("ten condition images: no artifact at {} (IGNIS_IMAGE_ARTIFACT)", artifact.display()));
        return None;
    }
    if !oracle.join("edit/condition.png").is_file() {
        gpu_profile::skip_or_fail(&format!("ten condition images: no oracle at {} (IGNIS_IMAGE_ORACLE)", oracle.display()));
        return None;
    }
    Some((Reader::open(&artifact).expect("the artifact opens"), oracle))
}

/// `condition.png` decoded to RGBA8.
fn condition_image(edit: &Path) -> (u32, u32, Vec<u8>) {
    let file = std::fs::read(edit.join("condition.png")).unwrap();
    let mut reader = png::Decoder::new(std::io::Cursor::new(file)).read_info().expect("condition.png decodes");
    let mut rgba = vec![0u8; reader.output_buffer_size().expect("a bounded frame")];
    let info = reader.next_frame(&mut rgba).expect("its frame");
    assert_eq!((info.bit_depth, info.color_type), (png::BitDepth::Eight, png::ColorType::Rgba), "condition.png is RGBA8");
    (info.width, info.height, rgba)
}

/// A leaf call, under the profile's rule for a failure (a refusal is a test failure either way).
fn ran<T>(result: Result<T, ImageLeafError>, what: &str) -> Option<T> {
    match result {
        Ok(v) => Some(v),
        Err(ImageLeafError::Failed(e)) => {
            gpu_profile::skip_or_fail(&format!("{what}: {e}"));
            None
        }
        Err(e) => panic!("{what}: {e}"),
    }
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn ten_condition_images_encode_in_chunks_and_a_step_runs_over_them() {
    let Some((reader, oracle)) = fixtures() else { return };
    let edit = oracle.join("edit");
    let case: serde_json::Value = serde_json::from_slice(&std::fs::read(edit.join("case.json")).unwrap()).unwrap();
    let (width, height, rgba) = condition_image(&edit);
    let frontend = EncoderFrontend::from_reader(&reader).expect("the artifact's front end");
    let count = image_vision::MAX_SEGMENTS as usize;
    let images: Vec<ConditionImage> = (0..count).map(|_| ConditionImage { width, height, rgba: &rgba }).collect();
    let input = frontend.input(case["case"]["prompt"].as_str().unwrap(), &images).expect("the front end");

    let dev = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            return;
        }
    };
    let Some(mut leaf) = ran(ImageLeaf::load(&reader, dev, &ImageLeafOptions::new(VramMode::Derived { headroom_bytes: HEADROOM })), "the load") else {
        return;
    };
    println!("{}", leaf.plan().line());
    let leaf_images: Vec<LeafImage> =
        input.images.iter().map(|v| LeafImage { grid: v.grid, pixels: &v.pixels, width, height, rgba: &rgba }).collect();
    let text = ConditionText {
        input_ids: &input.input_ids,
        positions: &input.positions,
        drop_idx: input.drop_idx,
        image_pad_rows: &input.image_pad_rows,
    };
    let start = Instant::now();
    let Some(condition) = ran(leaf.encode_condition(&text, &leaf_images), "encode_condition") else { return };
    let encode_ms = start.elapsed().as_secs_f64() * 1e3;
    let latent = (width / 16 * height / 16) as usize;
    let text_rows = input.input_ids.len() - input.drop_idx - input.image_pad_rows.len();
    assert_eq!(condition.tokens() as usize, text_rows + count * latent, "the prefix: the text rows and ten images' latents");
    assert_eq!(condition.blocks().len(), count);
    let chunk = condition.prefix_chunk();
    assert!(chunk.is_some(), "a {}-token prefix's whole pass fits the arena: the chunked pass is not exercised", condition.tokens());

    // One denoising step of a 64 x 64 target over the ten-image prefix.
    let target = 64 * 64;
    let noise: Vec<f32> = (0..target * 64).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
    if ran(leaf.start_image(&condition, 64, 64, &noise), "start_image").is_none() {
        return;
    }
    let start = Instant::now();
    if ran(leaf.denoise_step(&condition, 0), "denoise_step").is_none() {
        return;
    }
    let step_ms = start.elapsed().as_secs_f64() * 1e3;
    let Some(velocity) = ran(leaf.velocity(), "velocity") else { return };
    assert!(velocity.iter().all(|v| v.is_finite()), "a non-finite velocity over the ten-image prefix");
    println!(
        "ten images: {} encoder tokens, a {}-token prefix ({} GiB of runs) in chunks of {} tokens; encode_condition {encode_ms:.0} ms \
         (VAE x 10, tower, encoder, prefix pass), a 64 x 64 step over it {step_ms:.0} ms",
        input.input_ids.len(),
        condition.tokens(),
        format_args!("{:.2}", condition.bytes() as f64 / (1u64 << 30) as f64),
        chunk.unwrap()
    );
    ran(leaf.release_condition(condition), "release_condition");
}
