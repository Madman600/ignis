//! The image leaf on the real artifact against spec 01's recordings (spec image-model/04, GitHub
//! #321): acceptance 1, 2, 3 at 1024² and 2048², and 5.
//!
//! - **Load and plan (AC 1).** The family loads from the artifact (`ImageLeaf::load`: the full
//!   bind, the text encoder, the vision tower and the DiT's descriptors, the arena) and prints its
//!   plan line: weights, arena, condition budget. A load whose weights and arena do not fit an
//!   explicit budget is refused before anything is placed, naming the shortfall.
//! - **The sampler (AC 2).** Its sigmas and timesteps equal the recorded `sigmas.npy` and
//!   `timesteps.npy` bit for bit; the Euler update of each step's recorded input latents by the
//!   recorded velocity is the recorded output, bit for bit; and in every step below, the leaf's own
//!   update on the device equals the host's on the leaf's velocity, bit for bit.
//! - **t2i, step by step, at 1024² and 2048² (AC 3; the cases `t2i` and `t2i-2k`, one test each).** `encode_condition` runs ignis's own text encoder on the
//!   recorded `input_ids` and `mrope_position_ids` (never the recorded `prompt_embeds`), txt_in and
//!   the DiT's prefix pass; then each of the 8 steps is fed the recorded input latents
//!   (teacher-forced), and its velocity's relative L2 against the recorded velocity must be within
//!   `thresholds.json`'s per-step bound. The noise file packed is step 0's input, checked.
//! - **The diagnosis (AC 5).** The residual stream after blocks 0, 15 and 31 beside the recorded
//!   ones, no threshold: the prefix's rows after `encode_condition` against step 0's first rows,
//!   step 0's target rows against the rest of step 0's dumps, step 1's against step 1's.
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
use ignis_core::image_leaf::{euler_step, ConditionText, ImageLeaf, ImageLeafError, ImageLeafOptions, ImagePlanError};
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
    // AC 2: the schedule.
    let sigmas = npy_f32(&dir.join("sigmas.npy"), &[9]);
    let timesteps = npy_f32(&dir.join("timesteps.npy"), &[8]);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&leaf.sampler().sigmas), bits(&sigmas), "the sigmas are the recorded ones");
    assert_eq!(bits(&leaf.sampler().timesteps()), bits(&timesteps), "the timesteps are the recorded ones");
    let steps = leaf.sampler().steps();
    assert_eq!(steps, 8);

    // The condition: ignis's own encoder on the recorded ids and positions.
    let drop = case["drop_idx"].as_u64().unwrap() as usize;
    let (shape, ids) = load_npy(&dir.join("input_ids.npy"));
    let tokens = shape[1];
    let ids: Vec<u32> = i64s(&ids).into_iter().map(|i| i as u32).collect();
    let (pshape, pos) = load_npy(&dir.join("mrope_position_ids.npy"));
    assert_eq!(pshape, vec![3, 1, tokens]);
    let positions: Vec<i32> = i64s(&pos).into_iter().map(|p| p as i32).collect();
    let prefix = tokens - drop;
    let joint = case["sequence"]["dit"].as_u64().unwrap() as usize;
    let (h, w) = (case["img_shapes"][0][1].as_u64().unwrap() as u32, case["img_shapes"][0][2].as_u64().unwrap() as u32);
    let target = (h * w) as usize;
    assert_eq!(joint, prefix + target, "the joint sequence is [prefix ; target]");
    let taps = [0u32, 15, 31];
    let start = Instant::now();
    let Some((condition, prefix_taps)) =
        ran(leaf.encode_condition_tapped(&ConditionText { input_ids: &ids, positions: &positions, drop_idx: drop }, &taps), "encode_condition")
    else {
        return;
    };
    let encode_s = start.elapsed().as_secs_f64();
    assert_eq!(condition.tokens() as usize, prefix);

    // AC 5: the prefix's rows at blocks 0, 15, 31 against step 0's first rows (t = 0, the same at every step).
    let block_dump = |step: usize, b: u32, rows: usize| npy_f32(&dir.join(format!("step{step}/block{b:02}.npy")), &[1, rows, HIDDEN]);
    let mut report = Vec::new();
    for (i, &b) in taps.iter().enumerate() {
        let want = block_dump(0, b, joint);
        report.push(format!("prefix block {b:2}: {:.3}%", rel_l2(&prefix_taps[i], &want[..prefix * HIDDEN]) * 100.0));
    }

    // AC 3: the 8 steps, teacher-forced.
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
        let step_taps: &[u32] = if step < 2 { &taps } else { &[] };
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
}
