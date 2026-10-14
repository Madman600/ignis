//! The image model's encoder with its vision tower and DeepStack (spec image-model/03 acceptance 4
//! and 7, GitHub #320), driven through the Rust bindings on the real artifact.
//!
//! - **The edit case against the oracle (AC 4).** The tower (`qwen_image::bind_vision`) and the
//!   text encoder (`bind_text_encoder`), each materialized on the device, run spec 01's edit case:
//!   the tower on the condition image's `pixel_values` (f32, rounded to BF16 at upload), its
//!   merger's rows written over the image-pad rows and the three DeepStack mergers' rows added
//!   after text layers 0, 1 and 2 (the text encoder's hooks). The output against the recorded fp32
//!   `encoder_hidden_states`, with spec 01's encoder metric (each row's relative L2, then their
//!   RMS): over the text rows after the first image pad, within `thresholds.json`'s
//!   `encoder_after_vision` bound; over every text row from row 1 (`encoder`, printed with its
//!   bound); row 0 (`encoder_row0`, printed). With DeepStack off (the same tower, no adds) the
//!   numbers are printed beside, and the first must be over the bound: spec 03's weak check that
//!   DeepStack matters to the test, resting on 2 of the 18 rows.
//!   It runs twice: from the recorded inputs (`input_ids`, `mrope_position_ids`, `pixel_values`
//!   and `image_grid_thw`, the pad rows where `mm_token_type_ids` is 1), and through
//!   `EncoderFrontend` from the real condition image (`condition.png`, decoded, which must equal
//!   the VAE input read back). The two must agree bit for bit.
//! - **Ten condition images (AC 7).** The edit image ten times (10 x 1,024 merged tokens) through
//!   the front end, the tower and the text encoder: each stage's wall time, printed.
//!
//! With `$IGNIS_IMAGE_VISION_REF` naming a directory of transformers' tower outputs on the edit
//! case (`merger.npy` and `deepstack{0,1,2}.npy` from the checkpoint's weights in fp32, `w8_*.npy`
//! from the artifact's W8 weights decoded, in fp32; [1024][4096] each, from
//! `tools/image-converter/vision_probe.py`'s `outputs`), each tower output's error against both is
//! printed: the stage diagnosis spec 03 asks for when AC 4 is over its bound.
//!
//! The artifact is `$IGNIS_IMAGE_ARTIFACT`, else
//! `F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer`; the oracle
//! `$IGNIS_IMAGE_ORACLE`, else `F:/ai/models/ignis-image-2.1-turbo/oracle`. Explicit GPU profile
//! (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing GPU, fixture or a kernel error is a skip,
//! under it a hard failure (`ignis_core::gpu_profile`). Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::qwen_image::{self, QwenImageGeometry};
use ignis_artifact::{materialize, CudaDevice, Device, DeviceBuffer, MaterializedArtifact, Reader};
use ignis_core::gpu_profile;
use ignis_core::image_encoder::{self, embed_rows, TextEncoderConfig, TextEncoderWeights};
use ignis_core::image_vision::{self, vision_call, VisionTowerConfig, VisionWeights};
use ignis_core::vision::Grid;
use ignis_core::{image_gemm, moe};
use ignis_server::image_encoder_input::{ConditionImage, EncoderFrontend};

const HIDDEN: usize = 4096;

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

fn i64s(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

/// Spec 01's encoder metric over `rows` of `[tokens][4096]`: each row's relative L2 against
/// `want`, then their RMS, in f64.
fn rms_rel(got: &[f32], want: &[f32], rows: &[usize]) -> f64 {
    let mean: f64 = rows
        .iter()
        .map(|&r| {
            let (g, w) = (&got[r * HIDDEN..(r + 1) * HIDDEN], &want[r * HIDDEN..(r + 1) * HIDDEN]);
            let e: f64 = g.iter().zip(w).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
            let n: f64 = w.iter().map(|b| (*b as f64).powi(2)).sum();
            e / n
        })
        .sum::<f64>()
        / rows.len() as f64;
    mean.sqrt()
}

/// The relative L2 over a whole tensor.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let e: f64 = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
    let n: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    (e / n).sqrt()
}

/// A binding's result after a sync of the legacy stream (where the ops run), under the profile's
/// rule for a kernel error.
fn ran(result: Result<(), String>, what: &str) -> bool {
    match result.and_then(|_| moe::stream_sync(std::ptr::null_mut())) {
        Ok(()) => true,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("{what}: {e}"));
            false
        }
    }
}

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

/// What one encode reads: the sequence and its condition images.
struct Inputs {
    ids: Vec<u32>,
    positions: Vec<i32>,
    pad_rows: Vec<u32>,
    images: Vec<(Grid, Vec<f32>)>,
}

/// What one encode gives back: x after the 36 layers with DeepStack on and off, the tower's outputs,
/// and the stages' wall times (ms).
struct Encoded {
    on: Vec<f32>,
    off: Vec<f32>,
    embeds: Vec<u8>,
    deep: [Vec<u8>; 3],
    front_ms: f64,
    tower_ms: f64,
    text_ms: f64,
}

/// The tower and the text encoder on the device, ready to run.
struct Encoder {
    text: TextEncoderWeights,
    vision: VisionWeights,
    table: Vec<u8>,
    _text: MaterializedArtifact,
    _vision: MaterializedArtifact,
    dev: CudaDevice,
}

impl Encoder {
    /// Materializes both scoped binds; None (after skip_or_fail) when the GPU or the artifact is
    /// not there.
    fn load(reader: &Reader) -> Option<Self> {
        let mut dev = match CudaDevice::create(0) {
            Ok(d) => d,
            Err(e) => {
                gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
                return None;
            }
        };
        let g = QwenImageGeometry::qwen_image_21_turbo();
        let text_bound = qwen_image::bind_text_encoder(reader, &g.text).expect("the text encoder binds");
        let vision_bound = qwen_image::bind_vision(reader, &g.vision).expect("the vision tower binds");
        let start = Instant::now();
        let (text_m, vision_m) = match (
            materialize(reader, &text_bound.plan, &mut dev, None),
            materialize(reader, &vision_bound.plan, &mut dev, None),
        ) {
            (Ok(t), Ok(v)) => (t, v),
            (Err(e), _) | (_, Err(e)) => {
                gpu_profile::skip_or_fail(&format!("the encoder does not materialize: {e}"));
                return None;
            }
        };
        let config = text_m.resource_bytes(text_bound.handles[qwen_image::TEXT_ENCODER_CONFIG]).unwrap();
        let text_config = TextEncoderConfig::parse(config, &g.text).expect("the artifact's text config");
        let tower_config = VisionTowerConfig::parse(config, &g.vision).expect("the artifact's vision config");
        println!(
            "encoder: text {:.2} GiB, tower {:.2} GiB on the device in {:.1} s; DeepStack after tower blocks {:?}",
            text_bound.plan.device_capacity_bytes as f64 / (1u64 << 30) as f64,
            vision_bound.plan.device_capacity_bytes as f64 / (1u64 << 30) as f64,
            start.elapsed().as_secs_f64(),
            tower_config.deepstack_layers
        );
        let text = TextEncoderWeights::new(&text_m, &text_bound.handles, &g.text, text_config).expect("the text descriptors");
        let vision = VisionWeights::new(&vision_m, &vision_bound.handles, &g.vision, &tower_config).expect("the tower descriptors");
        let table = text_m.host_tensor_bytes(text_bound.handles["text_encoder/embed_tokens.weight"]).unwrap().to_vec();
        if !ran(image_gemm::prepare(), "image_gemm::prepare") {
            return None;
        }
        Some(Self { text, vision, table, _text: text_m, _vision: vision_m, dev })
    }

    fn upload(&mut self, bytes: &[u8]) -> DeviceBuffer {
        let buf = self.dev.allocate(bytes.len() as u64).expect("device allocation");
        self.dev.copy_h2d(&buf, 0, bytes).expect("upload");
        self.dev.synchronize().expect("upload sync");
        buf
    }

    fn download(&mut self, buf: &DeviceBuffer) -> Vec<u8> {
        let mut out = vec![0u8; buf.len() as usize];
        self.dev.copy_d2h(buf, 0, &mut out).expect("download");
        self.dev.synchronize().expect("download sync");
        out
    }

    /// The tower on `inputs`' images, then the 36 text layers from the embedding rows with the
    /// vision hooks, once with DeepStack and once without. `runs` timed repeats of each stage
    /// after a first run (the median is reported). None after skip_or_fail.
    fn encode(&mut self, inputs: &Inputs, runs: usize) -> Option<Encoded> {
        let null = std::ptr::null_mut();
        let images: Vec<(Grid, &[f32])> = inputs.images.iter().map(|(g, p)| (*g, &p[..])).collect();
        let call = vision_call(&images).expect("the tower call");
        let rows = call.tokens() as usize;
        assert_eq!(rows, inputs.pad_rows.len(), "one merged token per image-pad row");
        let out_bytes = (rows * HIDDEN * 2) as u64;
        let d_embeds = self.dev.allocate(out_bytes).expect("the merger's output");
        let d_deep: Vec<DeviceBuffer> = (0..3).map(|_| self.dev.allocate(out_bytes).expect("a DeepStack output")).collect();
        let deep_ptrs = [ptr(&d_deep[0]), ptr(&d_deep[1]), ptr(&d_deep[2])];
        let vws = image_vision::workspace_bytes(call.tokens(), call.segments());
        let d_vws = self.dev.allocate(vws).expect("the tower's workspace");
        let vdesc = self.vision.descriptor();
        let mut tower_times = Vec::new();
        for _ in 0..=runs {
            let t = Instant::now();
            if !ran(unsafe { image_vision::forward(&vdesc, &call, ptr(&d_embeds), Some(&deep_ptrs), ptr(&d_vws), vws, null) }, "the tower") {
                return None;
            }
            tower_times.push(t.elapsed().as_secs_f64() * 1e3);
        }

        let tokens = inputs.ids.len();
        let x0: Vec<u8> = embed_rows(&self.table, HIDDEN, &inputs.ids).expect("the embedding rows").iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_x = self.upload(&x0);
        let d_pos = self.upload(&inputs.positions.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
        let d_rows = self.upload(&inputs.pad_rows.iter().flat_map(|r| (*r as i32).to_le_bytes()).collect::<Vec<u8>>());
        let tws = image_encoder::workspace_bytes(tokens as u32);
        let d_tws = self.dev.allocate(tws).expect("the text encoder's workspace");
        let tdesc = self.text.descriptor();
        let mut text_times = Vec::new();
        let mut outs = Vec::new();
        for deepstack in [true, false] {
            let vision = image_encoder::ffi::ignis_image_encoder_vision {
                rows: rows as u32,
                row_index: ptr(&d_rows) as *const i32,
                embeds: ptr(&d_embeds),
                deepstack: if deepstack { deep_ptrs.map(|p| p as *const c_void) } else { [std::ptr::null(); 3] },
            };
            for i in 0..=if deepstack { runs } else { 0 } {
                if i > 0 || !deepstack {
                    self.dev.copy_h2d(&d_x, 0, &x0).expect("upload");
                    self.dev.synchronize().expect("upload sync");
                }
                let t = Instant::now();
                let rc = unsafe {
                    image_encoder::forward(&tdesc, 0, tdesc.layer_count, ptr(&d_x) as *mut f32, tokens as u32, ptr(&d_pos) as *const i32, Some(&vision), ptr(&d_tws), tws, null)
                };
                if !ran(rc, "the text encoder") {
                    return None;
                }
                if deepstack {
                    text_times.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            outs.push(f32s(&self.download(&d_x)));
        }
        let embeds = self.download(&d_embeds);
        let deep = [self.download(&d_deep[0]), self.download(&d_deep[1]), self.download(&d_deep[2])];
        for b in [d_embeds, d_vws, d_x, d_pos, d_rows, d_tws].into_iter().chain(d_deep) {
            self.dev.deallocate(b).unwrap();
        }
        let median = |mut v: Vec<f64>| {
            let warm = v.split_off(1);
            let mut warm = if warm.is_empty() { v } else { warm };
            warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
            warm[warm.len() / 2]
        };
        let off = outs.pop().unwrap();
        let on = outs.pop().unwrap();
        Some(Encoded { on, off, embeds, deep, front_ms: 0.0, tower_ms: median(tower_times), text_ms: median(text_times) })
    }
}

/// The artifact and the oracle's edit case, or None after skip_or_fail.
fn fixtures() -> Option<(Reader, PathBuf, serde_json::Value)> {
    let artifact = machine_local("IGNIS_IMAGE_ARTIFACT", "F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer");
    let oracle = machine_local("IGNIS_IMAGE_ORACLE", "F:/ai/models/ignis-image-2.1-turbo/oracle");
    if !artifact.is_file() {
        gpu_profile::skip_or_fail(&format!("the edit encoder: no artifact at {} (IGNIS_IMAGE_ARTIFACT)", artifact.display()));
        return None;
    }
    if !oracle.join("thresholds.json").is_file() || !oracle.join("edit/case.json").is_file() {
        gpu_profile::skip_or_fail(&format!("the edit encoder: no oracle at {} (IGNIS_IMAGE_ORACLE)", oracle.display()));
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
        gpu_profile::skip_or_fail(&format!("the edit encoder: {} is not the artifact thresholds.json was measured on", artifact.display()));
        return None;
    }
    Some((reader, oracle, thresholds))
}

/// The real condition image: `condition.png` decoded to RGBA8, checked against the VAE's input
/// read back (`2 (u / 255) - 1` of the bytes is the recording, image_encoder_input.rs).
fn condition_image(edit: &Path) -> (u32, u32, Vec<u8>) {
    let file = std::fs::read(edit.join("condition.png")).unwrap();
    let mut reader = png::Decoder::new(std::io::Cursor::new(file)).read_info().expect("condition.png decodes");
    let mut rgba = vec![0u8; reader.output_buffer_size().expect("a bounded frame")];
    let info = reader.next_frame(&mut rgba).expect("its frame");
    assert_eq!((info.bit_depth, info.color_type), (png::BitDepth::Eight, png::ColorType::Rgba), "condition.png is RGBA8");
    let (w, h) = (info.width as usize, info.height as usize);
    let (shape, planes) = load_npy(&edit.join("condition_vae_input.npy"));
    assert_eq!(shape, vec![1, 4, 1, h, w], "condition_vae_input's shape");
    let planes = f32s(&planes);
    let mismatched = (0..h * w * 4).filter(|&i| 2.0f32 * (rgba[i] as f32 / 255.0f32) - 1.0f32 != planes[(i % 4) * h * w + i / 4]).count();
    assert_eq!(mismatched, 0, "condition.png is the VAE's input, byte for byte");
    (info.width, info.height, rgba)
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_edit_encoder_with_the_tower_and_deepstack_is_within_the_oracle_bound() {
    let Some((reader, oracle, thresholds)) = fixtures() else { return };
    let edit = oracle.join("edit");
    let case: serde_json::Value = serde_json::from_slice(&std::fs::read(edit.join("case.json")).unwrap()).unwrap();
    let drop = case["drop_idx"].as_u64().expect("drop_idx") as usize;
    let prompt = case["case"]["prompt"].as_str().expect("the prompt");
    let npy = |field: &str| load_npy(&edit.join(format!("{field}.npy")));

    // The recorded inputs.
    let (shape, ids) = npy("input_ids");
    let tokens = shape[1];
    let ids: Vec<u32> = i64s(&ids).into_iter().map(|i| i as u32).collect();
    let positions: Vec<i32> = i64s(&npy("mrope_position_ids").1).into_iter().map(|p| p as i32).collect();
    assert_eq!(positions.len(), 3 * tokens, "mrope_position_ids is [3][1][tokens]");
    let pad_rows: Vec<u32> = (0..).zip(i64s(&npy("mm_token_type_ids").1)).filter(|&(_, t)| t == 1).map(|(r, _)| r).collect();
    let grid_thw = i64s(&npy("image_grid_thw").1);
    let grid = Grid { t: grid_thw[0] as u32, h: grid_thw[1] as u32, w: grid_thw[2] as u32 };
    let recorded = Inputs { ids, positions, pad_rows, images: vec![(grid, f32s(&npy("pixel_values").1))] };
    let (wshape, want) = npy("encoder_hidden_states");
    assert_eq!(wshape, vec![1, tokens, HIDDEN], "encoder_hidden_states' shape");
    let want = f32s(&want);

    // The front end's, from the real condition image.
    let (width, height, rgba) = condition_image(&edit);
    let frontend = EncoderFrontend::from_reader(&reader).expect("the artifact's front end");
    let start = Instant::now();
    let input = frontend.input(prompt, &[ConditionImage { width, height, rgba: &rgba }]).expect("the front end");
    let front_ms = start.elapsed().as_secs_f64() * 1e3;
    assert_eq!(input.drop_idx, drop, "drop_idx");
    let front = Inputs {
        ids: input.input_ids,
        positions: input.positions,
        pad_rows: input.image_pad_rows,
        images: input.images.into_iter().map(|v| (v.grid, v.pixels)).collect(),
    };
    assert!(front.ids == recorded.ids && front.positions == recorded.positions && front.pad_rows == recorded.pad_rows, "the front end's sequence is the recording's");
    assert_eq!(front.images[0].0, grid, "the front end's grid");
    assert!(front.images[0].1.iter().zip(&recorded.images[0].1).all(|(a, b)| a.to_bits() == b.to_bits()), "the front end's pixel_values are the recording's");

    let Some(mut encoder) = Encoder::load(&reader) else { return };
    let Some(from_recording) = encoder.encode(&recorded, 3) else { return };
    let Some(from_front) = encoder.encode(&front, 0) else { return };

    // The metric's rows, in the sequence's numbering.
    let pads: std::collections::BTreeSet<usize> = recorded.pad_rows.iter().map(|&r| r as usize).collect();
    let first_pad = *pads.first().expect("an image pad");
    let after: Vec<usize> = (first_pad + 1..tokens).filter(|r| !pads.contains(r)).collect();
    let text: Vec<usize> = (drop + 1..tokens).filter(|r| !pads.contains(r)).collect();
    let yard = &thresholds["cases"]["edit"];
    let bound = yard["encoder_after_vision"]["bound"].as_f64().expect("the encoder_after_vision bound");
    let pct = |v: f64| v * 100.0;
    let mut errors = Vec::new();
    for (label, x) in [("DeepStack on", &from_recording.on), ("DeepStack off", &from_recording.off)] {
        assert!(x.iter().all(|v| v.is_finite()), "{label}: a non-finite output");
        let (a, t, r0) = (rms_rel(x, &want, &after), rms_rel(x, &want, &text), rms_rel(x, &want, &[drop]));
        println!(
            "edit, {label}: {:.2}% over the {} text rows after the first image pad (row {first_pad}; bound {:.2}%; bf16 {:.2}%, sim {:.2}%); \
             encoder {:.2}% over {} text rows from row 1 (bound {:.2}%; bf16 {:.2}%, sim {:.2}%); encoder_row0 {:.2}% (bf16 {:.2}%, sim {:.2}%)",
            pct(a),
            after.len(),
            pct(bound),
            pct(yard["encoder_after_vision"]["bf16_error"].as_f64().unwrap()),
            pct(yard["encoder_after_vision"]["sim_error"].as_f64().unwrap()),
            pct(t),
            text.len(),
            pct(yard["encoder"]["bound"].as_f64().unwrap()),
            pct(yard["encoder"]["bf16_error"].as_f64().unwrap()),
            pct(yard["encoder"]["sim_error"].as_f64().unwrap()),
            pct(r0),
            pct(yard["encoder_row0"]["bf16_error"].as_f64().unwrap()),
            pct(yard["encoder_row0"]["sim_error"].as_f64().unwrap()),
        );
        let per_row: Vec<String> = after.iter().map(|&r| format!("{r}:{:.1}", pct(rms_rel(x, &want, &[r])))).collect();
        println!("  per row after the first pad (row:%): {}", per_row.join(" "));
        errors.push(a);
    }
    println!(
        "edit: {tokens} tokens, {} image pads; front end {front_ms:.1} ms (CPU), tower {:.2} ms, text encoder {:.2} ms (warm medians of 3)",
        pads.len(),
        from_recording.tower_ms,
        from_recording.text_ms
    );

    if let Some(refs) = std::env::var("IGNIS_IMAGE_VISION_REF").ok().filter(|s| !s.is_empty()) {
        // The stage diagnosis: the tower's outputs against transformers' (fp32 weights; W8 weights).
        let rows = recorded.pad_rows.len();
        let all: Vec<usize> = (0..rows).collect();
        for (name, got) in [("merger", &from_recording.embeds), ("deepstack0", &from_recording.deep[0]), ("deepstack1", &from_recording.deep[1]), ("deepstack2", &from_recording.deep[2])] {
            let got = bf16s(got);
            let mut line = format!("  tower {name:>10}:");
            for prefix in ["", "w8_"] {
                let (_, reference) = load_npy(&Path::new(&refs).join(format!("{prefix}{name}.npy")));
                let reference = f32s(&reference);
                let worst = (0..rows).map(|r| rms_rel(&got, &reference, &[r])).fold(0.0f64, f64::max);
                line += &format!(
                    " vs {}: {:.2}% whole, {:.2}% per-row RMS, worst row {:.2}%;",
                    if prefix.is_empty() { "fp32" } else { "W8 fp32" },
                    pct(rel_l2(&got, &reference)),
                    pct(rms_rel(&got, &reference, &all)),
                    pct(worst)
                );
            }
            println!("{line}");
        }
    }

    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(
        from_front.embeds == from_recording.embeds && from_front.deep == from_recording.deep,
        "the tower's outputs through the front end are the recording's, bit for bit"
    );
    assert!(same(&from_front.on, &from_recording.on) && same(&from_front.off, &from_recording.off), "the encoder through the front end is the recording's, bit for bit");
    assert!(errors[0] <= bound, "edit: the error after the first image pad {:.3}% is over the bound {:.3}%", pct(errors[0]), pct(bound));
    // Spec 03's check that DeepStack matters to the test: without it the same metric is over the
    // bound. A weak check (fp32 itself exceeds the bound by 4% without DeepStack, through two rows).
    assert!(errors[1] > bound, "edit: DeepStack off ({:.3}%) is within the bound {:.3}%: the test cannot tell DeepStack's absence", pct(errors[1]), pct(bound));
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn a_ten_image_edit_encodes_and_its_wall_time_is_printed() {
    let Some((reader, oracle, _)) = fixtures() else { return };
    let edit = oracle.join("edit");
    let case: serde_json::Value = serde_json::from_slice(&std::fs::read(edit.join("case.json")).unwrap()).unwrap();
    let (width, height, rgba) = condition_image(&edit);
    let frontend = EncoderFrontend::from_reader(&reader).expect("the artifact's front end");
    let images: Vec<ConditionImage> = (0..image_vision::MAX_SEGMENTS).map(|_| ConditionImage { width, height, rgba: &rgba }).collect();
    let start = Instant::now();
    let input = frontend.input(case["case"]["prompt"].as_str().unwrap(), &images).expect("the front end");
    let front_ms = start.elapsed().as_secs_f64() * 1e3;
    let inputs = Inputs {
        ids: input.input_ids,
        positions: input.positions,
        pad_rows: input.image_pad_rows,
        images: input.images.into_iter().map(|v| (v.grid, v.pixels)).collect(),
    };
    let Some(mut encoder) = Encoder::load(&reader) else { return };
    let Some(mut encoded) = encoder.encode(&inputs, 3) else { return };
    encoded.front_ms = front_ms;
    assert!(encoded.on.iter().all(|v| v.is_finite()), "a non-finite output");
    // The segments are independent: the ten identical images' tower rows are the one-image run's, ten
    // times over, bit for bit (real weights at T 40,960, past where the vendor cases stop at 4,096).
    let first = frontend.input(case["case"]["prompt"].as_str().unwrap(), &images[..1]).expect("the front end, one image");
    let first = Inputs {
        ids: first.input_ids,
        positions: first.positions,
        pad_rows: first.image_pad_rows,
        images: first.images.into_iter().map(|v| (v.grid, v.pixels)).collect(),
    };
    let Some(single) = encoder.encode(&first, 1) else { return };
    for (name, ten, one) in [
        ("merger", &encoded.embeds, &single.embeds),
        ("deepstack0", &encoded.deep[0], &single.deep[0]),
        ("deepstack1", &encoded.deep[1], &single.deep[1]),
        ("deepstack2", &encoded.deep[2], &single.deep[2]),
    ] {
        assert_eq!(ten.len(), one.len() * image_vision::MAX_SEGMENTS as usize, "{name}: ten segments' rows");
        for (segment, rows) in ten.chunks(one.len()).enumerate() {
            assert!(rows == &one[..], "{name}: segment {segment}'s rows are not the one-image run's, bit for bit");
        }
    }
    let patches: u64 = inputs.images.iter().map(|(g, _)| g.raw_patches()).sum();
    println!(
        "ten images: {} tokens ({} image pads, {patches} patches); front end {:.1} ms (CPU), tower {:.2} ms, text encoder {:.2} ms (warm medians of 3); \
         tower + text {:.2} ms",
        inputs.ids.len(),
        inputs.pad_rows.len(),
        encoded.front_ms,
        encoded.tower_ms,
        encoded.text_ms,
        encoded.tower_ms + encoded.text_ms
    );
}
