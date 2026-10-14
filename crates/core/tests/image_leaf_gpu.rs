//! The image leaf on the real artifact against spec 01's recordings (spec image-model/04, GitHub
//! #321): acceptance 1, 2, 3 at 1024² and 2048², 4, 5, 6, 7, 8 and 9, and the close-out's checks
//! (321e): the chunked prefix pass against the whole one, a non-square VAE encode, and the widest
//! preset run free and decoded.
//!
//! - **Load and plan (AC 1).** The family loads from the artifact (`ImageLeaf::load`: the full
//!   bind, the text encoder, the vision tower and the DiT's descriptors, the arena) and prints its
//!   plan line: weights, arena, condition budget. A load whose weights and arena do not fit an
//!   explicit budget is refused before anything is placed, naming the shortfall.
//! - **The sampler (AC 2).** Its sigmas and timesteps equal the recorded `sigmas.npy` and
//!   `timesteps.npy` bit for bit, and so do the timesteps the DiT reads (`dit_timesteps.npy`); the
//!   Euler update of each step's recorded input latents by the
//!   recorded velocity is the recorded output, bit for bit; and in every step below, the leaf's own
//!   update on the device equals the host's on the leaf's velocity, bit for bit.
//! - **t2i, step by step, at 1024² and 2048² (AC 3; the cases `t2i` and `t2i-2k`, one test each).** `encode_condition` runs ignis's own text encoder on the
//!   recorded `input_ids` and `mrope_position_ids` (never the recorded `prompt_embeds`), txt_in and
//!   the DiT's prefix pass; then each of the 8 steps is fed the recorded input latents
//!   (teacher-forced), and its velocity's relative L2 against the recorded velocity must be within
//!   `thresholds.json`'s per-step bound. The noise file packed is step 0's input, checked.
//! - **Edit, step by step (AC 4).** `encode_condition` with the condition image: the VAE encode,
//!   the tower and DeepStack through the encoder's hooks on the recorded sequence and tower input,
//!   txt_in and img_in laid out in the encoder's order; its segments must be the recorded ones; then
//!   the 8 steps as in AC 3, against the edit case's bound.
//! - **VAE encode (AC 7).** `vae_encode` of the condition image (its bytes from the recorded VAE
//!   input) against the recorded condition latents, within `thresholds.json`'s bound (5 x bf16's).
//!   A non-square crop of it (1024 x 768) encodes too: finite, and near the square's latents where
//!   it holds the same pixels.
//! - **The diagnosis (AC 5).** The residual stream after blocks 0, 15 and 31 beside the recorded
//!   ones, no threshold: the prefix's rows after `encode_condition` against step 0's first rows,
//!   step 0's target rows against the rest of step 0's dumps, step 1's against step 1's.
//! - **VAE decode (AC 6).** `vae_decode` of each case's recorded final latents (step 7's output)
//!   against the recorded `image.png` (diffusers' fp32 decode), at least 40 dB PSNR per channel,
//!   alpha included. At 1024² the arena's banded decode equals the unbanded one bit for bit; at
//!   2048² the decode runs in the arena, and two band heights with no common edge give the same
//!   image bit for bit.
//! - **The chunked prefix pass (review-321ab M1).** The edit's prefix pass cut in chunks of 1,024
//!   tokens (`ImageLeafOptions::prefix_chunk`, the image block cut in four) equals the whole pass
//!   bit for bit: the residual stream after blocks 0, 15 and 31, and step 0's velocity over it.
//! - **The widest preset (review-321c L2).** 2752 x 1536, a non-square target, runs free from the
//!   server's noise and decodes within the arena; the PNG goes beside the free runs.
//! - **n = 3, release and cancel (AC 8).** One condition, three images (the second from other
//!   noise, the third from the first's noise: the same bytes as the first, so the prefix outlived
//!   the image between them unchanged); then a job cancelled after three steps. Releasing each leaves
//!   the device's free memory where it was before the job, measured.
//! - **Free run, for the eye (AC 9).** Each case end to end from its noise file: encode_condition,
//!   the 8 steps on the leaf's own latents, vae_decode; the PNG goes to
//!   `$IGNIS_IMAGE_FREERUN/<case>/ignis.png` (else `F:/ai/models/ignis-image-2.1-turbo/freerun`),
//!   beside a copy of diffusers' fp32 `image.png`. Its PSNR and the final latents' distance from
//!   the recorded ones are printed, no threshold.
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
use ignis_core::image::{ImageSize, TargetLatents};
use ignis_core::image_leaf::{
    self, euler_step, Condition, ConditionImage, ConditionText, ImageLeaf, ImageLeafError, ImageLeafOptions, ImagePlanError, MAX_TARGET_TOKENS,
};
use ignis_core::vision::Grid;
use ignis_core::vram::VramMode;

const HIDDEN: usize = 4096;
const GIB: u64 = 1 << 30;
/// The server's default VRAM headroom (`vram.headroom_bytes`).
const HEADROOM: u64 = 1536 << 20;

fn machine_local(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| default.to_string()))
}

/// A little-endian C-order .npy's shape and payload.
fn load_npy(path: &Path) -> (Vec<usize>, Vec<u8>) {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&raw[..6], b"\x93NUMPY", "{} is not a .npy", path.display());
    let (len, at) = if raw[6] == 1 { (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10) } else { (u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize, 12) };
    let header = std::str::from_utf8(&raw[at..at + len]).unwrap();
    assert!(header.contains("'fortran_order': False"), "{} is not C order", path.display());
    let shape_at = header.find("'shape': (").unwrap() + 10;
    let shape: Vec<usize> = header[shape_at..header[shape_at..].find(')').unwrap() + shape_at].split(',').filter_map(|s| s.trim().parse().ok()).collect();
    (shape, raw[at + len..].to_vec())
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn i64s(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn npy_f32(path: &Path, shape: &[usize]) -> Vec<f32> {
    let (got, payload) = load_npy(path);
    assert_eq!(got, shape, "{}'s shape", path.display());
    f32s(&payload)
}

/// Relative L2 over the whole tensor, in f64 (spec 01's velocity metric).
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut e, mut n) = (0.0f64, 0.0f64);
    for (a, b) in got.iter().zip(want) {
        e += (*a as f64 - *b as f64).powi(2);
        n += (*b as f64).powi(2);
    }
    (e / n).sqrt()
}

/// The artifact, the oracle and its thresholds, or None after skip_or_fail.
fn fixtures() -> Option<(Reader, PathBuf, serde_json::Value)> {
    let artifact = machine_local("IGNIS_IMAGE_ARTIFACT", "F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer");
    let oracle = machine_local("IGNIS_IMAGE_ORACLE", "F:/ai/models/ignis-image-2.1-turbo/oracle");
    if !artifact.is_file() {
        gpu_profile::skip_or_fail(&format!("the image leaf: no artifact at {} (IGNIS_IMAGE_ARTIFACT)", artifact.display()));
        return None;
    }
    if !oracle.join("thresholds.json").is_file() || !oracle.join("t2i/case.json").is_file() {
        gpu_profile::skip_or_fail(&format!("the image leaf: no oracle at {} (IGNIS_IMAGE_ORACLE)", oracle.display()));
        return None;
    }
    let thresholds: serde_json::Value = serde_json::from_slice(&std::fs::read(oracle.join("thresholds.json")).unwrap()).unwrap();
    let reader = Reader::open(&artifact).expect("the artifact opens");
    // The bound was measured on this artifact. The sha256 thresholds.json records would read all 16.7
    // GB each run: the size and the header's identity (model id, weights id) say it is the same pack.
    let recorded = &thresholds["artifact"];
    let identity = reader.identity();
    if Some(reader.file_bytes()) != recorded["bytes"].as_u64()
        || Some(identity.model_id.as_str()) != recorded["identity"]["model_id"].as_str()
        || Some(identity.weights_id.as_str()) != recorded["identity"]["weights_id"].as_str()
    {
        gpu_profile::skip_or_fail(&format!("the image leaf: {} is not the artifact thresholds.json was measured on", artifact.display()));
        return None;
    }
    Some((reader, oracle, thresholds))
}

fn device() -> Option<CudaDevice> {
    match CudaDevice::create(0) {
        Ok(d) => Some(d),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            None
        }
    }
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
fn the_t2i_steps_at_1024_are_within_the_oracle_bound() {
    let Some((reader, oracle, thresholds)) = fixtures() else { return };

    // AC 1: a budget below the weights and the arena is refused before anything is placed.
    let Some(dev) = device() else { return };
    let small = ImageLeafOptions { vram: VramMode::Explicit { budget_bytes: 8 * GIB, allow_oversubscription: false }, ..ImageLeafOptions::new(VramMode::Derived { headroom_bytes: HEADROOM }) };
    match ImageLeaf::load(&reader, dev, &small) {
        Err(ImageLeafError::Plan(e @ ImagePlanError::DoesNotFit { .. })) => {
            let message = e.to_string();
            println!("an 8 GiB budget: {message}");
            assert!(message.contains("short by"), "the refusal names the shortfall: {message}");
        }
        Err(e) => panic!("an 8 GiB budget: refused for another reason: {e}"),
        Ok(_) => panic!("an 8 GiB budget: the load fits"),
    }

    let Some(mut leaf) = load(&reader) else { return };
    t2i_case(&mut leaf, &oracle, &thresholds, "t2i");
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_t2i_steps_at_2048_are_within_the_oracle_bound() {
    let Some((reader, oracle, thresholds)) = fixtures() else { return };
    let Some(mut leaf) = load(&reader) else { return };
    t2i_case(&mut leaf, &oracle, &thresholds, "t2i-2k");
}

/// AC 1: the load and its plan line, under the server's default headroom.
fn load(reader: &Reader) -> Option<ImageLeaf> {
    let dev = device()?;
    let start = Instant::now();
    let leaf = ran(ImageLeaf::load(reader, dev, &ImageLeafOptions::new(VramMode::Derived { headroom_bytes: HEADROOM })), "the load")?;
    println!("{} (loaded in {:.1} s)", leaf.plan().line(), start.elapsed().as_secs_f64());
    Some(leaf)
}

/// ACs 2, 3 and 5 on one of the oracle's text-to-image cases (`t2i` at 1024², `t2i-2k` at 2048²).
fn t2i_case(leaf: &mut ImageLeaf, oracle: &Path, thresholds: &serde_json::Value, name: &str) {
    let dir = oracle.join(name);
    let case: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("case.json")).unwrap()).unwrap();
    // The condition: ignis's own encoder on the recorded ids and positions.
    let drop = case["drop_idx"].as_u64().unwrap() as usize;
    let (ids, positions) = recorded_sequence(&dir);
    let start = Instant::now();
    let text = ConditionText { input_ids: &ids, positions: &positions, drop_idx: drop, image_pad_rows: &[] };
    let Some((condition, prefix_taps)) = ran(leaf.encode_condition_tapped(&text, &[], &TAPS), "encode_condition") else { return };
    let encode_s = start.elapsed().as_secs_f64();
    assert_eq!(condition.tokens() as usize, ids.len() - drop);
    teacher_forced(leaf, &dir, thresholds, name, &case, condition, prefix_taps, encode_s);
}

/// The blocks AC 5 reads the residual stream after.
const TAPS: [u32; 3] = [0, 15, 31];

/// The recorded `input_ids` and `mrope_position_ids` (`[3][1][tokens]`).
fn recorded_sequence(dir: &Path) -> (Vec<u32>, Vec<i32>) {
    let (shape, ids) = load_npy(&dir.join("input_ids.npy"));
    let tokens = shape[1];
    let ids: Vec<u32> = i64s(&ids).into_iter().map(|i| i as u32).collect();
    let (pshape, pos) = load_npy(&dir.join("mrope_position_ids.npy"));
    assert_eq!(pshape, vec![3, 1, tokens]);
    (ids, i64s(&pos).into_iter().map(|p| p as i32).collect())
}

/// ACs 2, 3 or 4, and 5 over `condition` (from `encode_condition_tapped` with [`TAPS`]): the
/// prefix's segments against the recorded ones; the prefix's rows at blocks 0, 15, 31 against step
/// 0's first rows; the 8 steps, teacher-forced, each velocity against the case's bound; the target's
/// rows at those blocks in steps 0 and 1.
#[allow(clippy::too_many_arguments)]
fn teacher_forced(
    leaf: &mut ImageLeaf,
    dir: &Path,
    thresholds: &serde_json::Value,
    name: &str,
    case: &serde_json::Value,
    condition: Condition,
    prefix_taps: Vec<Vec<f32>>,
    encode_s: f64,
) {
    // AC 2: the schedule.
    let sigmas = npy_f32(&dir.join("sigmas.npy"), &[9]);
    let timesteps = npy_f32(&dir.join("timesteps.npy"), &[8]);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&leaf.sampler().sigmas), bits(&sigmas), "the sigmas are the recorded ones");
    assert_eq!(bits(&leaf.sampler().timesteps()), bits(&timesteps), "the timesteps are the recorded ones");
    let steps = leaf.sampler().steps();
    assert_eq!(steps, 8);
    let dit_timesteps = npy_f32(&dir.join("dit_timesteps.npy"), &[8]);
    let ours: Vec<f32> = (0..steps).map(|i| leaf.sampler().dit_timestep(i)).collect();
    assert_eq!(bits(&ours), bits(&dit_timesteps), "the timesteps the DiT reads are the recorded ones");

    // The target is the last of img_shapes; the joint sequence is [prefix ; target].
    let last = case["img_shapes"].as_array().unwrap().last().unwrap().clone();
    let (h, w) = (last[1].as_u64().unwrap() as u32, last[2].as_u64().unwrap() as u32);
    let target = (h * w) as usize;
    let prefix = condition.tokens() as usize;
    let joint = case["sequence"]["dit"].as_u64().unwrap() as usize;
    assert_eq!(joint, prefix + target, "the joint sequence is [prefix ; target]");
    let recorded_segments: Vec<(u32, u32, bool)> = case["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| (g[0].as_u64().unwrap() as u32, g[1].as_u64().unwrap() as u32, g[2].as_bool().unwrap()))
        .collect();
    assert_eq!(condition.segments(), recorded_segments, "the prefix's segments are diffusers'");

    // AC 5: the prefix's rows at blocks 0, 15, 31 against step 0's first rows (t = 0, the same at every step).
    let block_dump = |step: usize, b: u32, rows: usize| npy_f32(&dir.join(format!("step{step}/block{b:02}.npy")), &[1, rows, HIDDEN]);
    let mut report = Vec::new();
    // With condition images, also the image blocks' rows and the text rows apart.
    let image_rows: Vec<bool> = (0..prefix as u32).map(|t| condition.blocks().iter().any(|&(s, (bh, bw))| t >= s && t < s + bh * bw)).collect();
    let rows_of = |x: &[f32], image: bool| -> Vec<f32> {
        x.chunks_exact(HIDDEN).zip(&image_rows).filter(|(_, i)| **i == image).flat_map(|(r, _)| r.iter().copied()).collect()
    };
    for (i, &b) in TAPS.iter().enumerate() {
        let want = block_dump(0, b, joint);
        let want = &want[..prefix * HIDDEN];
        let mut line = format!("prefix block {b:2}: {:.3}%", rel_l2(&prefix_taps[i], want) * 100.0);
        if !condition.blocks().is_empty() {
            let part = |image: bool| rel_l2(&rows_of(&prefix_taps[i], image), &rows_of(want, image)) * 100.0;
            line += &format!(" (image rows {:.3}%, text rows {:.3}%)", part(true), part(false));
        }
        report.push(line);
    }
    drop(prefix_taps);

    // AC 3 / 4: the 8 steps, teacher-forced.
    let noise = npy_f32(&dir.join("noise.npy"), &[1, 64, h as usize, w as usize]);
    let packed: Vec<f32> = (0..target * 64).map(|i| noise[(i % 64) * target + i / 64]).collect();
    let bound = thresholds["cases"][name]["velocity"]["bound"].as_array().expect("the velocity bound").iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
    let yard = &thresholds["cases"][name]["velocity"];
    if ran(leaf.start_image(&condition, h, w, &packed), "start_image").is_none() {
        return;
    }
    let mut over = Vec::new();
    let mut step_ms = Vec::new();
    for step in 0..steps {
        let latents_in = npy_f32(&dir.join(format!("step{step}/latents_in.npy")), &[1, target, 64]);
        let velocity_want = npy_f32(&dir.join(format!("step{step}/velocity.npy")), &[1, target, 64]);
        let latents_out = npy_f32(&dir.join(format!("step{step}/latents_out.npy")), &[1, target, 64]);
        if step == 0 {
            assert_eq!(bits(&packed), bits(&latents_in), "the noise file, packed, is step 0's input");
        }
        // AC 2: the recorded velocity steps the recorded input to the recorded output, bit for bit.
        let (sigma, sigma_next) = leaf.sampler().sigma(step);
        let mut host = latents_in.clone();
        euler_step(&mut host, &velocity_want, sigma, sigma_next);
        assert_eq!(bits(&host), bits(&latents_out), "step {step}: the Euler update of the recording is the recorded output");

        if ran(leaf.set_latents(&latents_in), "set_latents").is_none() {
            return;
        }
        let step_taps: &[u32] = if step < 2 { &TAPS } else { &[] };
        let t = Instant::now();
        let Some(tapped) = ran(leaf.denoise_step_tapped(&condition, step, step_taps), "denoise_step") else { return };
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let (Some(velocity), Some(stepped)) = (ran(leaf.velocity(), "velocity"), ran(leaf.latents(), "latents")) else { return };
        assert!(velocity.iter().all(|v| v.is_finite()), "step {step}: a non-finite velocity");
        let mut want_stepped = latents_in.clone();
        euler_step(&mut want_stepped, &velocity, sigma, sigma_next);
        assert_eq!(bits(&stepped), bits(&want_stepped), "step {step}: the device's update is the host's, bit for bit");
        let err = rel_l2(&velocity, &velocity_want);
        println!(
            "{name} step {step} (sigma {sigma:.6}): velocity {:.3}% (bound {:.3}%; bf16 {:.3}%, sim {:.3}%)",
            err * 100.0,
            bound[step] * 100.0,
            yard["bf16_error"][step].as_f64().unwrap() * 100.0,
            yard["sim_error"][step].as_f64().unwrap() * 100.0
        );
        if err > bound[step] {
            over.push(step);
        }
        for (i, &b) in step_taps.iter().enumerate() {
            let want = block_dump(step, b, if step == 0 { joint } else { target });
            let want = if step == 0 { &want[prefix * HIDDEN..] } else { &want[..] };
            report.push(format!("step {step} target block {b:2}: {:.3}%", rel_l2(&tapped[i], want) * 100.0));
        }
    }
    println!("{name}: AC 5, the residual stream against the recorded block outputs (relative L2, no threshold):");
    for line in &report {
        println!("  {line}");
    }
    println!(
        "{name}: encode_condition {:.0} ms (with 3 taps); denoise steps {:?} ms (steps 0 and 1 with 3 taps each)",
        encode_s * 1e3,
        step_ms.iter().map(|m| m.round() as i64).collect::<Vec<_>>()
    );
    ran(leaf.release_condition(condition), "release_condition");
    assert!(over.is_empty(), "{name}: steps {over:?} are over their velocity bound");
}

/// The edit case's condition image as RGBA8 bytes, from the recorded VAE input (`[1][4][1][h][w]`,
/// the pipeline's `2 (u / 255) - 1` of the bytes): each value's byte, checked to map back exactly.
fn condition_rgba(dir: &Path) -> (u32, u32, Vec<u8>) {
    let (shape, planes) = load_npy(&dir.join("condition_vae_input.npy"));
    assert_eq!((shape.len(), shape[1], shape[2]), (5, 4, 1), "condition_vae_input is [1][4][1][h][w]");
    let (h, w) = (shape[3], shape[4]);
    let planes = f32s(&planes);
    let rgba: Vec<u8> = (0..h * w * 4)
        .map(|i| {
            let v = planes[(i % 4) * h * w + i / 4];
            let u = ((v + 1.0) / 2.0 * 255.0).round() as u8;
            assert_eq!(2.0f32 * (u as f32 / 255.0f32) - 1.0f32, v, "the VAE input's value {v} is not a byte's");
            u
        })
        .collect();
    (w as u32, h as u32, rgba)
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_edit_condition_image_encodes_within_the_oracle_bound() {
    let Some((reader, oracle, thresholds)) = fixtures() else { return };
    let dir = oracle.join("edit");
    let Some(mut leaf) = load(&reader) else { return };
    // AC 7: the VAE encode of the condition image against the recorded (fp32) condition latents.
    let (width, height, rgba) = condition_rgba(&dir);
    let tokens = (width / 16 * height / 16) as usize;
    let want = npy_f32(&dir.join("condition_latents.npy"), &[1, tokens, 64]);
    let start = Instant::now();
    let Some(got) = ran(leaf.vae_encode(width, height, &rgba), "vae_encode") else { return };
    let ms = start.elapsed().as_secs_f64() * 1e3;
    assert!(got.iter().all(|v| v.is_finite()), "a non-finite latent");
    let yard = &thresholds["cases"]["edit"]["condition_latents"];
    let (bound, bf16) = (yard["bound"].as_f64().unwrap(), yard["bf16_error"].as_f64().unwrap());
    let err = rel_l2(&got, &want);
    println!(
        "edit: vae_encode of the {width} x {height} condition image: {:.3}% against the recorded condition latents (bound {:.3}% = 5 x bf16's \
         {:.3}%; {:.2} x bf16) in {ms:.1} ms (first call)",
        err * 100.0,
        bound * 100.0,
        bf16 * 100.0,
        err / bf16
    );
    assert!(err <= bound, "edit: the condition latents' error {:.3}% is over the bound {:.3}%", err * 100.0, bound * 100.0);

    // Review-321ab L4: a non-square image, the condition's top 768 rows (1024 x 768, the alpha
    // band below them cut off). Its latents are finite; in the top half of its rows, away from
    // the cut, they are the square's but for what the VAE's mid-block attention, which sees the
    // whole image, makes of the missing band.
    let rows = 768;
    let crop = &rgba[..(width * rows * 4) as usize];
    let Some(cropped) = ran(leaf.vae_encode(width, rows, crop), "vae_encode of a 1024 x 768 crop") else { return };
    assert_eq!(cropped.len(), (width / 16 * rows / 16) as usize * 64, "a 64 x 48 latent grid");
    assert!(cropped.iter().all(|v| v.is_finite()), "a non-finite latent of the 1024 x 768 crop");
    let row = (width / 16) as usize * 64;
    let top = (rows / 32) as usize * row;
    let near = rel_l2(&cropped[..top], &got[..top]);
    // A control: the same rows against the square's one latent row lower, what a crop laid out a
    // row off would match.
    let shifted = rel_l2(&cropped[..top], &got[row..row + top]);
    println!(
        "edit: vae_encode of its top {width} x {rows}: the top 24 latent rows {:.3}% from the square's (a row lower: {:.3}%)",
        near * 100.0,
        shifted * 100.0
    );
    assert!(near < shifted, "the 1024 x 768 crop's latents are no nearer the square's own rows than the rows below them");
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_edit_steps_are_within_the_oracle_bound() {
    let Some((reader, oracle, thresholds)) = fixtures() else { return };
    let dir = oracle.join("edit");
    let case: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("case.json")).unwrap()).unwrap();
    let Some(mut leaf) = load(&reader) else { return };
    // The condition: the recorded sequence and the tower's recorded input (spec 03: the front end
    // gives both bit for bit), the VAE's input from the recorded VAE input's bytes.
    let drop = case["drop_idx"].as_u64().unwrap() as usize;
    let (ids, positions) = recorded_sequence(&dir);
    let types = i64s(&load_npy(&dir.join("mm_token_type_ids.npy")).1);
    let pad_rows: Vec<u32> = (0..).zip(types).filter(|&(_, t)| t == 1).map(|(r, _)| r).collect();
    let grid_thw = i64s(&load_npy(&dir.join("image_grid_thw.npy")).1);
    let grid = Grid { t: grid_thw[0] as u32, h: grid_thw[1] as u32, w: grid_thw[2] as u32 };
    let pixels = f32s(&load_npy(&dir.join("pixel_values.npy")).1);
    let (width, height, rgba) = condition_rgba(&dir);
    let image = ConditionImage { grid, pixels: &pixels, width, height, rgba: &rgba };
    let text = ConditionText { input_ids: &ids, positions: &positions, drop_idx: drop, image_pad_rows: &pad_rows };
    let start = Instant::now();
    let Some((condition, prefix_taps)) = ran(leaf.encode_condition_tapped(&text, &[image], &TAPS), "encode_condition") else { return };
    let encode_s = start.elapsed().as_secs_f64();
    assert_eq!(condition.blocks(), &[(8, (64, 64))], "the condition image's block");
    assert_eq!(condition.prefix_chunk(), None, "a 4122-token prefix runs whole");
    teacher_forced(&mut leaf, &dir, &thresholds, "edit", &case, condition, prefix_taps, encode_s);
}

fn read_case(dir: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(dir.join("case.json")).unwrap()).unwrap()
}

/// A case's target, (height, width) in latent tokens: the last of `img_shapes`.
fn target_shape(case: &serde_json::Value) -> (u32, u32) {
    let last = case["img_shapes"].as_array().unwrap().last().unwrap();
    (last[1].as_u64().unwrap() as u32, last[2].as_u64().unwrap() as u32)
}

/// A case's noise file, packed (`packed[y * w + x][c] = noise[c][y][x]`).
fn packed_noise(dir: &Path, h: u32, w: u32) -> Vec<f32> {
    let target = (h * w) as usize;
    let noise = npy_f32(&dir.join("noise.npy"), &[1, 64, h as usize, w as usize]);
    (0..target * 64).map(|i| noise[(i % 64) * target + i / 64]).collect()
}

/// A case's condition through `encode_condition`: ignis's encoder on the recorded sequence, and
/// for the edit case the recorded tower input and the VAE input's bytes, as AC 4 builds it.
fn encode_case(leaf: &mut ImageLeaf, dir: &Path) -> Option<Condition> {
    encode_case_tapped(leaf, dir, &[]).map(|(condition, _)| condition)
}

/// [`encode_case`], with the prefix's residual stream after the blocks of `taps`.
fn encode_case_tapped(leaf: &mut ImageLeaf, dir: &Path, taps: &[u32]) -> Option<(Condition, Vec<Vec<f32>>)> {
    let case = read_case(dir);
    let drop = case["drop_idx"].as_u64().unwrap() as usize;
    let (ids, positions) = recorded_sequence(dir);
    if !dir.join("condition_vae_input.npy").is_file() {
        let text = ConditionText { input_ids: &ids, positions: &positions, drop_idx: drop, image_pad_rows: &[] };
        return ran(leaf.encode_condition_tapped(&text, &[], taps), "encode_condition");
    }
    let types = i64s(&load_npy(&dir.join("mm_token_type_ids.npy")).1);
    let pad_rows: Vec<u32> = (0..).zip(types).filter(|&(_, t)| t == 1).map(|(r, _)| r).collect();
    let grid_thw = i64s(&load_npy(&dir.join("image_grid_thw.npy")).1);
    let grid = Grid { t: grid_thw[0] as u32, h: grid_thw[1] as u32, w: grid_thw[2] as u32 };
    let pixels = f32s(&load_npy(&dir.join("pixel_values.npy")).1);
    let (width, height, rgba) = condition_rgba(dir);
    let image = ConditionImage { grid, pixels: &pixels, width, height, rgba: &rgba };
    let text = ConditionText { input_ids: &ids, positions: &positions, drop_idx: drop, image_pad_rows: &pad_rows };
    ran(leaf.encode_condition_tapped(&text, &[image], taps), "encode_condition")
}

/// An 8-bit RGBA PNG's width, height and pixels.
fn read_png(path: &Path) -> (u32, u32, Vec<u8>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut pixels = vec![0u8; reader.output_buffer_size().expect("a PNG frame that fits")];
    let info = reader.next_frame(&mut pixels).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!((info.color_type, info.bit_depth), (png::ColorType::Rgba, png::BitDepth::Eight), "{} is 8-bit RGBA", path.display());
    pixels.truncate(info.buffer_size());
    (info.width, info.height, pixels)
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header().unwrap().write_image_data(rgba).unwrap();
    }
    std::fs::write(path, out).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The PSNR of RGBA8 `got` against `want` per channel (R, G, B, A), in dB: infinite where they are
/// equal.
fn psnr(got: &[u8], want: &[u8]) -> [f64; 4] {
    assert_eq!(got.len(), want.len());
    let mut se = [0f64; 4];
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        se[i % 4] += (*a as f64 - *b as f64).powi(2);
    }
    let pixels = (got.len() / 4) as f64;
    se.map(|s| if s == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64.powi(2) / (s / pixels)).log10() })
}

/// Spec image-model/04 acceptance 6: the decode's PSNR per channel against diffusers' fp32 image.
const AC6_MIN_PSNR_DB: f64 = 40.0;

/// The bytes where two decodes differ.
fn mismatches(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_vae_decode_of_the_final_latents_is_the_recorded_png() {
    let Some((reader, oracle, _)) = fixtures() else { return };
    let Some(mut leaf) = load(&reader) else { return };
    // The decode reads the image slot alone: one text condition serves every case's shape.
    let Some(condition) = encode_case(&mut leaf, &oracle.join("t2i")) else { return };
    let mut low = Vec::new();
    for name in ["t2i", "edit", "t2i-2k"] {
        let dir = oracle.join(name);
        let (h, w) = target_shape(&read_case(&dir));
        let last = leaf.sampler().steps() - 1;
        let latents = npy_f32(&dir.join(format!("step{last}/latents_out.npy")), &[1, (h * w) as usize, 64]);
        if ran(leaf.start_image(&condition, h, w, &packed_noise(&dir, h, w)), "start_image").is_none()
            || ran(leaf.set_latents(&latents), "set_latents").is_none()
        {
            return;
        }
        let Some(free) = ran(leaf.device_free_bytes(), "the free memory") else { return };
        let t = Instant::now();
        let Some(rgba) = ran(leaf.vae_decode(), "vae_decode") else { return };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let Some(free_after) = ran(leaf.device_free_bytes(), "the free memory") else { return };
        assert_eq!(free_after, free, "{name}: vae_decode allocates nothing (it runs in the arena)");
        let (pw, ph, want) = read_png(&dir.join("image.png"));
        assert_eq!((pw, ph), (16 * w, 16 * h), "{name}: the recorded image's size");
        let db = psnr(&rgba, &want);
        let Some(planned) = ran(leaf.vae_decode_band_rows(), "the band plan") else { return };
        println!(
            "{name}: vae_decode {}x{} in bands of {planned} rows, {ms:.0} ms (the first at this size): PSNR R {:.2} G {:.2} B {:.2} A {:.2} dB against \
             diffusers' fp32 image.png",
            16 * w,
            16 * h,
            db[0],
            db[1],
            db[2],
            db[3]
        );
        if db.iter().any(|&d| d < AC6_MIN_PSNR_DB) {
            low.push(name);
        }
        if name == "t2i" {
            // The unbanded decode needs ~2.4 GB, more than the arena: its workspace is allocated for the call.
            let Some(unbanded) = ran(leaf.vae_decode_in_bands(16 * h), "the unbanded decode") else { return };
            assert!(planned < 16 * h, "the arena's 1024² decode runs in bands ({planned} rows)");
            assert_eq!(mismatches(&rgba, &unbanded), 0, "t2i: the decode in {planned}-row bands equals the unbanded one bit for bit");
        }
        if name == "t2i-2k" {
            // Review-321c L1: bands whose edges never meet below the image's height, so each
            // decode's band edges are checked against the other's interiors.
            let other = 208;
            let lcm = planned / gcd(planned, other) * other;
            assert!(lcm >= 16 * h, "t2i-2k: {planned}- and {other}-row bands share an edge at row {lcm}");
            let Some(half) = ran(leaf.vae_decode_in_bands(other), "the decode in other bands") else { return };
            assert_eq!(mismatches(&rgba, &half), 0, "t2i-2k: the decode in {planned}-row bands equals the one in {other}-row bands bit for bit");
        }
    }
    ran(leaf.release_condition(condition), "release_condition");
    assert!(low.is_empty(), "the cases {low:?} decode below {AC6_MIN_PSNR_DB} dB in a channel");
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn a_job_of_three_images_and_a_cancelled_job_leave_the_device_memory_where_it_was() {
    let Some((reader, oracle, _)) = fixtures() else { return };
    let Some(mut leaf) = load(&reader) else { return };
    let (t2i, edit) = (oracle.join("t2i"), oracle.join("edit"));
    let (h, w) = target_shape(&read_case(&t2i));
    assert_eq!(target_shape(&read_case(&edit)), (h, w), "the edit case's noise fits the t2i target");
    let (a, b) = (packed_noise(&t2i, h, w), packed_noise(&edit, h, w));
    let steps = leaf.sampler().steps();

    // A first job, one step and a decode: the process's first launch of a kernel loads its module
    // (CUDA's lazy loading), device memory that stays. Measured: 2 MiB here, none with
    // CUDA_MODULE_LOADING=EAGER. The jobs after it launch no new kernel.
    let Some(first) = ran(leaf.device_free_bytes(), "the free memory") else { return };
    // Review-321c L3: no image started, nothing to decode: a refusal (a 400), not a failure.
    assert!(matches!(leaf.vae_decode(), Err(ImageLeafError::Refused(_))), "vae_decode before start_image is refused");
    let Some(condition) = encode_case(&mut leaf, &t2i) else { return };
    if ran(leaf.start_image(&condition, h, w, &a), "start_image").is_none()
        || ran(leaf.denoise_step(&condition, 0), "denoise_step").is_none()
        || ran(leaf.vae_decode(), "vae_decode").is_none()
    {
        return;
    }
    ran(leaf.release_condition(condition), "release_condition");
    // The started image goes with its prefix: its RoPE table was laid out after it.
    assert!(matches!(leaf.vae_decode(), Err(ImageLeafError::Refused(_))), "release_condition clears the image slot");
    assert!(matches!(leaf.latents(), Err(ImageLeafError::Refused(_))), "release_condition clears the image slot");
    let Some(before) = ran(leaf.device_free_bytes(), "the free memory") else { return };
    println!(
        "the first job: {} bytes less free after its release (kernel modules loaded at first launch; CUDA_MODULE_LOADING={})",
        first as i64 - before as i64,
        std::env::var("CUDA_MODULE_LOADING").unwrap_or_else(|_| "unset, lazy".into())
    );

    // n = 3: one prefix, three images, the third from the first's noise.
    let Some(condition) = encode_case(&mut leaf, &t2i) else { return };
    let Some(held) = ran(leaf.device_free_bytes(), "the free memory") else { return };
    let mut images = Vec::new();
    for noise in [&a, &b, &a] {
        if ran(leaf.start_image(&condition, h, w, noise), "start_image").is_none() {
            return;
        }
        for step in 0..steps {
            if ran(leaf.denoise_step(&condition, step), "denoise_step").is_none() {
                return;
            }
        }
        let Some(rgba) = ran(leaf.vae_decode(), "vae_decode") else { return };
        images.push(rgba);
    }
    assert_eq!(
        mismatches(&images[0], &images[2]),
        0,
        "the third image, from the first's noise, is the first's bytes: the prefix outlived the image between unchanged"
    );
    assert!(mismatches(&images[0], &images[1]) > 0, "the second image, from other noise, differs");
    ran(leaf.release_condition(condition), "release_condition");
    let Some(after) = ran(leaf.device_free_bytes(), "the free memory") else { return };
    println!(
        "n = 3: free {before} bytes before the job, {held} with its prefix ({:.1} MiB), {after} after release_condition",
        (before - held) as f64 / (1u64 << 20) as f64
    );
    assert_eq!(after, before, "n = 3: release_condition leaves the device's memory where it was before the job");

    // A job cancelled between denoising steps 2 and 3: its prefix released there.
    let Some(condition) = encode_case(&mut leaf, &t2i) else { return };
    if ran(leaf.start_image(&condition, h, w, &a), "start_image").is_none() {
        return;
    }
    for step in 0..3 {
        if ran(leaf.denoise_step(&condition, step), "denoise_step").is_none() {
            return;
        }
    }
    ran(leaf.release_condition(condition), "release_condition");
    let Some(cancelled) = ran(leaf.device_free_bytes(), "the free memory") else { return };
    println!("cancelled after 3 steps: free {cancelled} bytes after release_condition");
    assert_eq!(cancelled, before, "a cancelled job's release leaves the device's memory where it was before the job");
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn each_case_runs_free_from_its_noise_file() {
    let Some((reader, oracle, _)) = fixtures() else { return };
    let Some(mut leaf) = load(&reader) else { return };
    let out = machine_local("IGNIS_IMAGE_FREERUN", "F:/ai/models/ignis-image-2.1-turbo/freerun");
    let steps = leaf.sampler().steps();
    for name in ["t2i", "t2i-2k", "edit"] {
        let dir = oracle.join(name);
        let (h, w) = target_shape(&read_case(&dir));
        let t = Instant::now();
        let Some(condition) = encode_case(&mut leaf, &dir) else { return };
        let encode_ms = t.elapsed().as_secs_f64() * 1e3;
        if ran(leaf.start_image(&condition, h, w, &packed_noise(&dir, h, w)), "start_image").is_none() {
            return;
        }
        let t = Instant::now();
        for step in 0..steps {
            if ran(leaf.denoise_step(&condition, step), "denoise_step").is_none() {
                return;
            }
        }
        let steps_ms = t.elapsed().as_secs_f64() * 1e3;
        let Some(latents) = ran(leaf.latents(), "latents") else { return };
        assert!(latents.iter().all(|v| v.is_finite()), "{name}: a non-finite latent");
        let recorded = npy_f32(&dir.join(format!("step{}/latents_out.npy", steps - 1)), &[1, (h * w) as usize, 64]);
        let t = Instant::now();
        let Some(rgba) = ran(leaf.vae_decode(), "vae_decode") else { return };
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;
        ran(leaf.release_condition(condition), "release_condition");
        let case_out = out.join(name);
        write_png(&case_out.join("ignis.png"), 16 * w, 16 * h, &rgba);
        std::fs::copy(dir.join("image.png"), case_out.join("diffusers-fp32.png")).unwrap();
        let (_, _, want) = read_png(&dir.join("image.png"));
        let db = psnr(&rgba, &want);
        println!(
            "{name} free run: {} ({encode_ms:.0} ms encode_condition, {steps_ms:.0} ms for {steps} steps, {decode_ms:.0} ms vae_decode); final latents \
             {:.2}% from diffusers' fp32 run; PSNR against its image R {:.2} G {:.2} B {:.2} A {:.2} dB (information)",
            case_out.join("ignis.png").display(),
            rel_l2(&latents, &recorded) * 100.0,
            db[0],
            db[1],
            db[2],
            db[3]
        );
    }
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_chunked_prefix_pass_equals_the_whole_one_on_the_edit() {
    let Some((reader, oracle, _)) = fixtures() else { return };
    let dir = oracle.join("edit");
    let (h, w) = target_shape(&read_case(&dir));
    let noise = packed_noise(&dir, h, w);
    // Each leaf in turn (two do not fit the card): the edit's prefix with taps, then step 0 over it.
    let run = |chunk: Option<u32>| -> Option<(Option<u32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>)> {
        let options = ImageLeafOptions { prefix_chunk: chunk, ..ImageLeafOptions::new(VramMode::Derived { headroom_bytes: HEADROOM }) };
        let mut leaf = ran(ImageLeaf::load(&reader, device()?, &options), "the load")?;
        let (condition, prefix_taps) = encode_case_tapped(&mut leaf, &dir, &TAPS)?;
        let ran_chunk = condition.prefix_chunk();
        ran(leaf.start_image(&condition, h, w, &noise), "start_image")?;
        let target_taps = ran(leaf.denoise_step_tapped(&condition, 0, &TAPS), "denoise_step")?;
        let velocity = ran(leaf.velocity(), "velocity")?;
        ran(leaf.release_condition(condition), "release_condition")?;
        Some((ran_chunk, prefix_taps, target_taps, velocity))
    };
    let Some(whole) = run(None) else { return };
    let Some(chunked) = run(Some(1024)) else { return };
    assert_eq!(whole.0, None, "the edit's 4122-token prefix runs whole by default");
    assert_eq!(chunked.0, Some(1024), "the knob cuts the prefix pass in chunks of 1024 (the image block in four)");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (i, &b) in TAPS.iter().enumerate() {
        assert!(bits(&chunked.1[i]) == bits(&whole.1[i]), "the prefix's rows after block {b}: the chunked pass differs from the whole one");
        assert!(bits(&chunked.2[i]) == bits(&whole.2[i]), "step 0's target rows after block {b}: the chunked prefix differs from the whole one");
    }
    assert!(bits(&chunked.3) == bits(&whole.3), "step 0's velocity over the chunked prefix differs from the whole one's");
    println!("edit: the prefix pass in chunks of 1024 equals the whole pass bit for bit (blocks {TAPS:?}, step 0's target rows and velocity)");
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_widest_preset_runs_free_and_decodes_within_the_arena() {
    let Some((reader, oracle, _)) = fixtures() else { return };
    let Some(mut leaf) = load(&reader) else { return };
    let size = ImageSize::new(2752, 1536);
    let (h, w) = (size.height / 16, size.width / 16);
    // Every spec 05 preset fits this arena (the server's test checks each); this one is the widest.
    let Ok(band_rows) = image_leaf::decode_band_rows(MAX_TARGET_TOKENS, size.width, size.height) else {
        panic!("{size}: its decode does not fit the arena")
    };
    let Some(condition) = encode_case(&mut leaf, &oracle.join("t2i")) else { return };
    // The server's noise for seed 7, packed.
    let noise = TargetLatents::noise(size, 7);
    let cells = size.latent_cells();
    let packed: Vec<f32> = (0..cells * 64).map(|i| noise.values[(i % 64) * cells + i / 64]).collect();
    if ran(leaf.start_image(&condition, h, w, &packed), "start_image").is_none() {
        return;
    }
    let t = Instant::now();
    for step in 0..leaf.sampler().steps() {
        if ran(leaf.denoise_step(&condition, step), "denoise_step").is_none() {
            return;
        }
    }
    let steps_ms = t.elapsed().as_secs_f64() * 1e3;
    let Some(latents) = ran(leaf.latents(), "latents") else { return };
    assert!(latents.iter().all(|v| v.is_finite()), "{size}: a non-finite latent");
    let Some(free) = ran(leaf.device_free_bytes(), "the free memory") else { return };
    let t = Instant::now();
    let Some(rgba) = ran(leaf.vae_decode(), "vae_decode") else { return };
    let decode_ms = t.elapsed().as_secs_f64() * 1e3;
    let Some(free_after) = ran(leaf.device_free_bytes(), "the free memory") else { return };
    ran(leaf.release_condition(condition), "release_condition");
    assert_eq!(free_after, free, "{size}: vae_decode allocates nothing (it runs in the arena)");
    assert_eq!(rgba.len(), (size.width * size.height * 4) as usize);
    // Not a blank or saturated frame: the RGB channels spread.
    let pixels = (size.width * size.height) as f64;
    let spread = |c: usize| {
        let mean = rgba.iter().skip(c).step_by(4).map(|&v| v as f64).sum::<f64>() / pixels;
        (rgba.iter().skip(c).step_by(4).map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / pixels).sqrt()
    };
    assert!((0..3).all(|c| spread(c) > 10.0), "{size}: a flat image (RGB spread {:.1} {:.1} {:.1})", spread(0), spread(1), spread(2));
    let out = machine_local("IGNIS_IMAGE_FREERUN", "F:/ai/models/ignis-image-2.1-turbo/freerun").join("t2i-2752x1536").join("ignis.png");
    write_png(&out, size.width, size.height, &rgba);
    println!(
        "{size}: 8 steps {steps_ms:.0} ms, vae_decode {decode_ms:.0} ms in bands of {band_rows} rows; {} (the t2i prompt, seed 7)",
        out.display()
    );
}
