//! The image leaf's one timing run (spec image-model/04 acceptance 10, GitHub #321): information,
//! not a gate. Text to image at 1024² and 2048²; an edit with 1 and with 4 condition images and a
//! 2048² target. The condition image is the oracle's 1024² one, the size the server encodes a
//! condition image at (about 1024² of area), repeated.
//!
//! Each job is the server's sequence of leaf calls: `encode_condition_timed` (the condition's
//! phases, with the stream drained between them, which the untimed call does not do), the 8
//! denoising steps one by one, then `vae_decode`. Every job runs twice and the second is printed:
//! the first loads the kernels' modules (CUDA's lazy loading) and settles the clocks.
//!
//! It runs only with `IGNIS_IMAGE_TIMING=1`, with the card to itself (the GPU lock in timing
//! mode); otherwise it returns at once. The throughput per kernel class comes from the kernel
//! leaf's benches (`ignis_image_gemm_bench`, `ignis_image_attn_bench`, `ignis_image_vae_bench`).
//!
//! The artifact is `$IGNIS_IMAGE_ARTIFACT`, else
//! `F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer`; the oracle
//! `$IGNIS_IMAGE_ORACLE`, else `F:/ai/models/ignis-image-2.1-turbo/oracle`. Explicit GPU profile
//! (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing GPU, fixture or a kernel error is a skip,
//! under it a hard failure (`ignis_core::gpu_profile`).

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::{CudaDevice, Reader};
use ignis_core::gpu_profile;
use ignis_core::image::{ImageSize, TargetLatents};
use ignis_core::image_leaf::{ConditionImage as LeafImage, ConditionText, EncodeTimes, ImageLeaf, ImageLeafError, ImageLeafOptions};
use ignis_core::vram::VramMode;
use ignis_server::image_encoder_input::{ConditionImage, EncoderFrontend};

/// The server's default VRAM headroom (`vram.headroom_bytes`).
const HEADROOM: u64 = 1536 << 20;

fn machine_local(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| default.to_string()))
}

/// The artifact and the oracle, or None after skip_or_fail.
fn fixtures() -> Option<(Reader, PathBuf)> {
    let artifact = machine_local("IGNIS_IMAGE_ARTIFACT", "F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer");
    let oracle = machine_local("IGNIS_IMAGE_ORACLE", "F:/ai/models/ignis-image-2.1-turbo/oracle");
    if !artifact.is_file() {
        gpu_profile::skip_or_fail(&format!("the image leaf's timing: no artifact at {} (IGNIS_IMAGE_ARTIFACT)", artifact.display()));
        return None;
    }
    if !oracle.join("edit/condition.png").is_file() || !oracle.join("t2i/case.json").is_file() {
        gpu_profile::skip_or_fail(&format!("the image leaf's timing: no oracle at {} (IGNIS_IMAGE_ORACLE)", oracle.display()));
        return None;
    }
    Some((Reader::open(&artifact).expect("the artifact opens"), oracle))
}

/// A case's prompt.
fn prompt(dir: &Path) -> String {
    let case: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("case.json")).unwrap()).unwrap();
    case["case"]["prompt"].as_str().unwrap().to_owned()
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

/// One job's times, in seconds.
struct Timing {
    prefix_tokens: u32,
    chunk: Option<u32>,
    encode_s: f64,
    phases: EncodeTimes,
    start_s: f64,
    steps_s: Vec<f64>,
    decode_s: f64,
}

/// One job: a condition of `prompt` and `images` copies of the condition image, then one image of
/// `side`² from the server's noise for seed 7.
fn job(leaf: &mut ImageLeaf, frontend: &EncoderFrontend, prompt: &str, images: usize, condition: (u32, u32, &[u8]), side: u32) -> Option<Timing> {
    let (width, height, rgba) = condition;
    let sources: Vec<ConditionImage> = (0..images).map(|_| ConditionImage { width, height, rgba }).collect();
    let input = frontend.input(prompt, &sources).expect("the front end");
    let leaf_images: Vec<LeafImage> = input.images.iter().map(|v| LeafImage { grid: v.grid, pixels: &v.pixels, width, height, rgba }).collect();
    let text = ConditionText {
        input_ids: &input.input_ids,
        positions: &input.positions,
        drop_idx: input.drop_idx,
        image_pad_rows: &input.image_pad_rows,
    };
    let t = Instant::now();
    let (condition, phases) = ran(leaf.encode_condition_timed(&text, &leaf_images), "encode_condition")?;
    let encode_s = t.elapsed().as_secs_f64();
    let size = ImageSize::new(side, side);
    let noise = TargetLatents::noise(size, 7);
    let cells = size.latent_cells();
    let packed: Vec<f32> = (0..cells * 64).map(|i| noise.values[(i % 64) * cells + i / 64]).collect();
    let t = Instant::now();
    ran(leaf.start_image(&condition, side / 16, side / 16, &packed), "start_image")?;
    let start_s = t.elapsed().as_secs_f64();
    let mut steps_s = Vec::new();
    for step in 0..leaf.sampler().steps() {
        let t = Instant::now();
        ran(leaf.denoise_step(&condition, step), "denoise_step")?;
        steps_s.push(t.elapsed().as_secs_f64());
    }
    let t = Instant::now();
    let rgba = ran(leaf.vae_decode(), "vae_decode")?;
    let decode_s = t.elapsed().as_secs_f64();
    assert_eq!(rgba.len(), (side * side * 4) as usize);
    let (prefix_tokens, chunk) = (condition.tokens(), condition.prefix_chunk());
    ran(leaf.release_condition(condition), "release_condition")?;
    Some(Timing { prefix_tokens, chunk, encode_s, phases, start_s, steps_s, decode_s })
}

#[test]
#[ignore = "GPU timing: IGNIS_IMAGE_TIMING=1, with the card to itself"]
fn the_image_leafs_timing_run() {
    if std::env::var("IGNIS_IMAGE_TIMING").ok().as_deref() != Some("1") {
        println!("the image leaf's timing run: skipped (IGNIS_IMAGE_TIMING=1 runs it)");
        return;
    }
    let Some((reader, oracle)) = fixtures() else { return };
    let (t2i, edit) = (prompt(&oracle.join("t2i")), prompt(&oracle.join("edit")));
    let (width, height, rgba) = condition_image(&oracle.join("edit"));
    let frontend = EncoderFrontend::from_reader(&reader).expect("the artifact's front end");
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
    let ms = |s: f64| s * 1e3;
    for (name, prompt, images, side) in
        [("t2i 1024²", &t2i, 0, 1024), ("t2i 2048²", &t2i, 0, 2048), ("edit, 1 image, 2048²", &edit, 1, 2048), ("edit, 4 images, 2048²", &edit, 4, 2048)]
    {
        // The first run warms; the second is the one timed.
        if job(&mut leaf, &frontend, prompt, images, (width, height, &rgba), side).is_none() {
            return;
        }
        let Some(t) = job(&mut leaf, &frontend, prompt, images, (width, height, &rgba), side) else { return };
        let p = t.phases;
        let dit: f64 = t.steps_s.iter().sum();
        println!(
            "{name}: prefix {} tokens ({}); encode_condition {:.1} ms = VAE {:.1} + tower {:.1} + encoder {:.1} + txt_in/img_in {:.1} + prefix pass \
             {:.1} (+ {:.1} on the host); start_image {:.1} ms; steps {:?} ms, the DiT loop {:.0} ms; vae_decode {:.0} ms; the job {:.0} ms",
            t.prefix_tokens,
            t.chunk.map_or("whole".to_owned(), |c| format!("chunks of {c}")),
            ms(t.encode_s),
            ms(p.vae_s),
            ms(p.tower_s),
            ms(p.encoder_s),
            ms(p.projections_s),
            ms(p.prefix_s),
            ms(t.encode_s - p.vae_s - p.tower_s - p.encoder_s - p.projections_s - p.prefix_s),
            ms(t.start_s),
            t.steps_s.iter().map(|&s| (ms(s) * 10.0).round() / 10.0).collect::<Vec<_>>(),
            ms(dit),
            ms(t.decode_s),
            ms(t.encode_s + t.start_s + dit + t.decode_s)
        );
    }
}
