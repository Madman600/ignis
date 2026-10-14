//! The image model's text encoder driven through the Rust bindings (spec image-model/03
//! acceptance 3 and 7, GitHub #320).
//!
//! - **The t2i cases against the oracle (AC 3).** The real artifact's text encoder (bound alone,
//!   `qwen_image::bind_text_encoder`, and materialized on the device) runs the 36 layers on the
//!   recorded `input_ids` and `mrope_position_ids` of spec 01's t2i and t2i-2k cases. Its output
//!   against the recorded fp32 `encoder_hidden_states` must stay within `thresholds.json`'s
//!   `encoder` bound, with spec 01's encoder metric: each text row's relative L2, from the row
//!   after `drop_idx` + 1 on (the rows whose `image_pad_mask` is 0), then their RMS. The first
//!   row after `drop_idx` (`encoder_row0`) is printed beside it, in no bound. The forward's wall
//!   time (AC 7) is printed too.
//! - **The bindings.** The three ops on small synthetic inputs, each against the host, and a
//!   refusal that comes back with the leaf's message. The kernel leaf's CTest
//!   (`ignis_kernel_image_encoder_test`) checks the ops in depth and the vision hooks.
//!
//! The artifact is `$IGNIS_IMAGE_ARTIFACT`, else
//! `F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer`; the oracle
//! `$IGNIS_IMAGE_ORACLE`, else `F:/ai/models/ignis-image-2.1-turbo/oracle`. Without either the
//! AC 3 test prints SKIP, a failure under the profile. With `$IGNIS_IMAGE_ENCODER_LAYERS` naming a
//! directory of per-layer references (`layer00.npy` .. `layer35.npy`, fp32 [tokens][4096], from
//! transformers' text model on the t2i case, `tools/image-converter/ref_layers.py`), it also runs the
//! t2i case one layer at a time, each from the reference's previous layer, and prints that layer's
//! own error against its reference: the diagnosis spec 03 asks for when AC 3 is over its bound.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing GPU, a
//! missing fixture or a kernel error is a skip, under it a hard failure (`ignis_core::gpu_profile`).
//! Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::os::raw::c_void;
use std::path::Path;
use std::time::Instant;

use ignis_artifact::qwen_image::{self, QwenImageGeometry};
use ignis_artifact::{materialize, CudaDevice, Device, DeviceBuffer, Reader};
use ignis_core::gpu_profile;
use ignis_core::image_encoder::{self, embed_rows, TextEncoderConfig, TextEncoderWeights};
use ignis_core::{image_gemm, moe};

const HIDDEN: usize = 4096;

fn device() -> Option<CudaDevice> {
    match CudaDevice::create(0) {
        Ok(d) => Some(d),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            None
        }
    }
}

fn upload(dev: &mut CudaDevice, bytes: &[u8]) -> DeviceBuffer {
    let buf = dev.allocate(bytes.len() as u64).expect("device allocation");
    dev.copy_h2d(&buf, 0, bytes).expect("upload");
    dev.synchronize().expect("upload sync");
    buf
}

fn download(dev: &mut CudaDevice, buf: &DeviceBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buf.len() as usize];
    dev.copy_d2h(buf, 0, &mut out).expect("download");
    dev.synchronize().expect("download sync");
    out
}

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

/// Runs a binding and waits for it, under the profile's rule for a kernel error. The op runs on
/// the legacy default stream, which `CudaDevice::synchronize` (its own non-blocking stream) does
/// not wait for: without this, a download could read the output before the kernel wrote it.
fn ran(result: Result<(), String>, what: &str) -> bool {
    match result.and_then(|_| moe::stream_sync(std::ptr::null_mut())) {
        Ok(()) => true,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("{what}: {e}"));
            false
        }
    }
}

fn f32_bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_to_f64(h: u16) -> f64 {
    f32::from_bits((h as u32) << 16) as f64
}

/// A little-endian C-order .npy's shape and payload (`image_elementwise_gpu.rs`'s reader).
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

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| default.to_string())
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_t2i_encoder_output_is_within_the_oracle_bound() {
    let artifact = env_or("IGNIS_IMAGE_ARTIFACT", "F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer");
    let oracle = env_or("IGNIS_IMAGE_ORACLE", "F:/ai/models/ignis-image-2.1-turbo/oracle");
    if !Path::new(&artifact).is_file() {
        gpu_profile::skip_or_fail(&format!("the encoder against the oracle: no artifact at {artifact} (IGNIS_IMAGE_ARTIFACT)"));
        return;
    }
    let thresholds_path = Path::new(&oracle).join("thresholds.json");
    if !thresholds_path.is_file() {
        gpu_profile::skip_or_fail(&format!("the encoder against the oracle: no oracle at {oracle} (IGNIS_IMAGE_ORACLE)"));
        return;
    }
    let thresholds: serde_json::Value = serde_json::from_slice(&std::fs::read(&thresholds_path).unwrap()).unwrap();
    let Some(mut dev) = device() else { return };

    let reader = Reader::open(Path::new(&artifact)).expect("the artifact opens");
    // The bound was measured on this artifact. The sha256 thresholds.json records would read all 16.7
    // GB each run: the size and the header's identity (model id, weights id) say it is the same pack.
    let recorded = &thresholds["artifact"];
    assert_eq!(Some(reader.file_bytes()), recorded["bytes"].as_u64(), "{artifact} is not the artifact thresholds.json was measured on");
    let identity = reader.identity();
    assert_eq!(Some(identity.model_id.as_str()), recorded["identity"]["model_id"].as_str(), "{artifact}'s model id");
    assert_eq!(Some(identity.weights_id.as_str()), recorded["identity"]["weights_id"].as_str(), "{artifact}'s weights id");
    let geometry = QwenImageGeometry::qwen_image_21_turbo().text;
    let bound = qwen_image::bind_text_encoder(&reader, &geometry).expect("the text encoder binds");
    let start = Instant::now();
    let materialized = match materialize(&reader, &bound.plan, &mut dev, None) {
        Ok(m) => m,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("materialize the text encoder on the device: {e}"));
            return;
        }
    };
    let load_s = start.elapsed().as_secs_f64();
    let config = TextEncoderConfig::parse(materialized.resource_bytes(bound.handles[qwen_image::TEXT_ENCODER_CONFIG]).unwrap(), &geometry)
        .expect("the artifact's text config");
    println!(
        "text encoder: {:.2} GiB on the device in {load_s:.1} s; eps {:e}, theta {}, mrope section {:?}",
        bound.plan.device_capacity_bytes as f64 / (1u64 << 30) as f64,
        config.rms_norm_eps,
        config.rope_theta,
        config.mrope_section
    );
    let weights = TextEncoderWeights::new(&materialized, &bound.handles, &geometry, config).expect("the layers' descriptors");
    let desc = weights.descriptor();
    if !ran(image_gemm::prepare(), "image_gemm::prepare") {
        return;
    }
    let table = materialized.host_tensor_bytes(bound.handles["text_encoder/embed_tokens.weight"]).unwrap();
    let layers = desc.layer_count;
    let null = std::ptr::null_mut();

    for case in ["t2i", "t2i-2k"] {
        let dir = Path::new(&oracle).join(case);
        let case_json: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("case.json")).unwrap()).unwrap();
        let drop = case_json["drop_idx"].as_u64().expect("drop_idx") as usize;
        let (shape, ids) = load_npy(&dir.join("input_ids.npy"));
        let tokens = shape[1];
        let ids: Vec<u32> = i64s(&ids).into_iter().map(|i| i as u32).collect();
        let (pshape, pos) = load_npy(&dir.join("mrope_position_ids.npy"));
        assert_eq!(pshape, vec![3, 1, tokens], "{case}: mrope_position_ids' shape");
        let positions: Vec<i32> = i64s(&pos).into_iter().map(|p| p as i32).collect();
        let (pad_shape, pad) = load_npy(&dir.join("image_pad_mask.npy"));
        assert_eq!(pad_shape, vec![1, tokens - drop], "{case}: image_pad_mask's shape");
        let (wshape, want) = load_npy(&dir.join("encoder_hidden_states.npy"));
        assert_eq!(wshape, vec![1, tokens, HIDDEN], "{case}: encoder_hidden_states' shape");
        let want = as_f32(&want);
        // The text rows from row 1 on, and row 0, in the sequence's numbering.
        let text_rows: Vec<usize> = (drop + 1..tokens).filter(|&r| pad[r - drop] == 0).collect();
        assert!(pad.iter().all(|&p| p == 0), "{case}: a t2i prompt has no image pad");

        let host_start = Instant::now();
        let x0 = embed_rows(table, HIDDEN, &ids).expect("the prompt's embedding rows");
        let x0_bytes = f32_bytes(&x0);
        let d_x = upload(&mut dev, &x0_bytes);
        let d_pos = upload(&mut dev, &positions.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
        let ws_bytes = image_encoder::workspace_bytes(tokens as u32);
        let d_ws = match dev.allocate(ws_bytes) {
            Ok(b) => b,
            Err(e) => {
                gpu_profile::skip_or_fail(&format!("{case}: the {ws_bytes}-byte workspace: {e}"));
                return;
            }
        };
        let host_s = host_start.elapsed().as_secs_f64();
        let run = |first: u32, end: u32| unsafe {
            image_encoder::forward(&desc, first, end, ptr(&d_x) as *mut f32, tokens as u32, ptr(&d_pos) as *const i32, None, ptr(&d_ws), ws_bytes, null)
        };

        // The first run, then five timed ones, each from the embedding rows.
        let mut times = Vec::new();
        for i in 0..6 {
            if i > 0 {
                dev.copy_h2d(&d_x, 0, &x0_bytes).expect("upload");
                dev.synchronize().expect("upload sync");
            }
            let t = Instant::now();
            if !ran(run(0, layers), &format!("{case}: the forward")) {
                return;
            }
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let got = as_f32(&download(&mut dev, &d_x));
        assert!(got.iter().all(|v| v.is_finite()), "{case}: a non-finite output");
        let err = rms_rel(&got, &want, &text_rows);
        let row0 = rms_rel(&got, &want, &[drop]);
        let limit = thresholds["cases"][case]["encoder"]["bound"].as_f64().expect("the encoder bound");
        let yard = &thresholds["cases"][case];
        let mut warm = times[1..].to_vec();
        warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{case}: {tokens} tokens (drop {drop}): encoder {:.2}% over {} text rows from row 1 (bound {:.2}%; bf16 {:.2}%, sim {:.2}%), encoder_row0 {:.2}% (bf16 {:.2}%, sim {:.2}%)",
            err * 100.0,
            text_rows.len(),
            limit * 100.0,
            yard["encoder"]["bf16_error"].as_f64().unwrap() * 100.0,
            yard["encoder"]["sim_error"].as_f64().unwrap() * 100.0,
            row0 * 100.0,
            yard["encoder_row0"]["bf16_error"].as_f64().unwrap() * 100.0,
            yard["encoder_row0"]["sim_error"].as_f64().unwrap() * 100.0,
        );
        println!(
            "{case}: forward of {layers} layers {:.2} ms warm (median of 5; min {:.2}), {:.2} ms the first run; host prep (embedding rows, uploads) {:.1} ms",
            warm[2], warm[0], times[0], host_s * 1e3
        );

        if case == "t2i" {
            if let Some(refs) = std::env::var("IGNIS_IMAGE_ENCODER_LAYERS").ok().filter(|s| !s.is_empty()) {
                // The diagnosis: layer by layer against transformers' text model, teacher-forced. Each
                // layer starts from the reference's input (the embedding rows, then the reference's
                // layer l-1) and is compared with the reference's output, so a row's error is that
                // layer's own: the errors of a free-running stack accumulate, cancel and amplify.
                for l in 0..layers {
                    if l == 0 {
                        dev.copy_h2d(&d_x, 0, &x0_bytes).expect("upload");
                    } else {
                        let (_, input) = load_npy(&Path::new(&refs).join(format!("layer{:02}.npy", l - 1)));
                        dev.copy_h2d(&d_x, 0, &input).expect("upload");
                    }
                    dev.synchronize().expect("upload sync");
                    if !ran(run(l, l + 1), &format!("layer {l}")) {
                        return;
                    }
                    let got = as_f32(&download(&mut dev, &d_x));
                    let (_, reference) = load_npy(&Path::new(&refs).join(format!("layer{l:02}.npy")));
                    let reference = as_f32(&reference);
                    let all: Vec<usize> = (0..tokens).collect();
                    println!(
                        "  layer {l:2} alone: text rows from row 1 {:.3}%, row 0 {:.3}%, every row {:.3}%",
                        rms_rel(&got, &reference, &text_rows) * 100.0,
                        rms_rel(&got, &reference, &[drop]) * 100.0,
                        rms_rel(&got, &reference, &all) * 100.0
                    );
                }
            }
        }
        dev.deallocate(d_x).unwrap();
        dev.deallocate(d_pos).unwrap();
        dev.deallocate(d_ws).unwrap();
        assert!(err <= limit, "{case}: the encoder's error {:.3}% is over the bound {:.3}%", err * 100.0, limit * 100.0);
    }
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn encoder_ops_through_the_bindings_match_the_host() {
    let Some(mut dev) = device() else { return };
    let null = std::ptr::null_mut();
    let (tokens, cols) = (4usize, HIDDEN);

    // rows: written over, then added, at rows 2 and 0.
    let x0: Vec<f32> = (0..tokens * cols).map(|i| (i % 13) as f32 - 6.0).collect();
    let src: Vec<u16> = (0..2 * cols).map(|i| to_bf16((i % 7) as f32 * 0.25)).collect();
    let d_x = upload(&mut dev, &f32_bytes(&x0));
    let d_src = upload(&mut dev, &src.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>());
    let d_idx = upload(&mut dev, &[2i32, 0].iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
    for add in [false, true] {
        if !ran(unsafe { image_encoder::rows(ptr(&d_x) as *mut f32, tokens as u32, cols as u32, ptr(&d_idx) as *const i32, 2, ptr(&d_src), add, null) }, "rows") {
            return;
        }
    }
    let got = as_f32(&download(&mut dev, &d_x));
    for (r, j) in [(2usize, 0usize), (0, 1)] {
        for c in 0..cols {
            let f = bf16_to_f64(src[j * cols + c]) as f32;
            assert_eq!(got[r * cols + c], f + f, "rows: row {r} column {c} is the feature written, then added");
        }
    }
    assert_eq!(got[cols..2 * cols], x0[cols..2 * cols], "rows: row 1 untouched");

    // residual_norm: x += delta, then the fp32 norm, against f64.
    let delta: Vec<f32> = (0..tokens * cols).map(|i| ((i * 7) % 11) as f32 * 0.125).collect();
    let w: Vec<u16> = (0..cols).map(|c| to_bf16(1.0 + (c % 5) as f32 * 0.0625)).collect();
    let (d_delta, d_w) = (upload(&mut dev, &f32_bytes(&delta)), upload(&mut dev, &w.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>()));
    let d_y = dev.allocate((tokens * cols * 4) as u64).unwrap();
    let before = as_f32(&download(&mut dev, &d_x));
    if !ran(unsafe { image_encoder::residual_norm(ptr(&d_x) as *mut f32, ptr(&d_delta) as *const f32, tokens as u32, cols as u32, ptr(&d_w), 1e-6, ptr(&d_y), null as *mut f32, true, null) }, "residual_norm") {
        return;
    }
    let (sum, y) = (as_f32(&download(&mut dev, &d_x)), as_f32(&download(&mut dev, &d_y)));
    for t in 0..tokens {
        let row: Vec<f32> = (0..cols).map(|c| before[t * cols + c] + delta[t * cols + c]).collect();
        assert_eq!(sum[t * cols..(t + 1) * cols], row[..], "residual_norm: the sum is the host's fp32 add");
        let r = 1.0 / (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / cols as f64 + 1e-6f32 as f64).sqrt();
        for c in 0..cols {
            let want = row[c] as f64 * r * bf16_to_f64(w[c]);
            assert!((y[t * cols + c] as f64 - want).abs() <= 1e-5 * (1.0 + want.abs()), "residual_norm: token {t} column {c}: {} for {want}", y[t * cols + c]);
        }
    }

    // qk_norm_mrope: one token at positions t 3, h 5, w 7, against f64 rotate_half with the
    // interleaved axes.
    let q: Vec<f32> = (0..4096).map(|i| ((i * 31) % 17) as f32 / 8.0 - 1.0).collect();
    let k: Vec<f32> = (0..1024).map(|i| ((i * 13) % 19) as f32 / 9.0 - 1.0).collect();
    let nw: Vec<u16> = (0..128).map(|c| to_bf16(0.75 + (c % 3) as f32 * 0.25)).collect();
    let nw_bytes: Vec<u8> = nw.iter().flat_map(|h| h.to_le_bytes()).collect();
    let (d_q, d_k, d_nw) = (upload(&mut dev, &f32_bytes(&q)), upload(&mut dev, &f32_bytes(&k)), upload(&mut dev, &nw_bytes));
    let d_pos = upload(&mut dev, &[3i32, 5, 7].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let (d_qo, d_ko) = (dev.allocate(4096 * 2).unwrap(), dev.allocate(1024 * 2).unwrap());
    if !ran(unsafe { image_encoder::qk_norm_mrope(ptr(&d_q) as *const f32, 4096, ptr(&d_k) as *const f32, 1024, 1, ptr(&d_nw), ptr(&d_nw), 1e-6, ptr(&d_pos) as *const i32, 5e6, [24, 20, 20], ptr(&d_qo), ptr(&d_ko), null) }, "qk_norm_mrope") {
        return;
    }
    let out = |b: Vec<u8>| -> Vec<f64> { b.chunks_exact(2).map(|c| bf16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect() };
    let (qo, ko) = (out(download(&mut dev, &d_qo)), out(download(&mut dev, &d_ko)));
    for (x, o) in [(&q, &qo), (&k, &ko)] {
        for h in 0..x.len() / 128 {
            let head = &x[h * 128..(h + 1) * 128];
            let r = 1.0 / (head.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 128.0 + 1e-6).sqrt();
            let n: Vec<f64> = (0..128).map(|c| head[c] as f64 * r * bf16_to_f64(nw[c])).collect();
            for i in 0..64 {
                let axis = if i % 3 == 1 && i < 60 { 1 } else if i % 3 == 2 && i < 60 { 2 } else { 0 };
                let angle = [3.0, 5.0, 7.0][axis] * 5e6f64.powf(-((2 * i) as f64) / 128.0);
                let (s, c) = angle.sin_cos();
                let (lo, hi) = (n[i] * c - n[i + 64] * s, n[i + 64] * c + n[i] * s);
                for (got, want) in [(o[h * 128 + i], lo), (o[h * 128 + i + 64], hi)] {
                    assert!((got - want).abs() <= 2f64.powi(-8) * want.abs() + 1e-4, "qk_norm_mrope: head {h} slot {i}: {got} for {want}");
                }
            }
        }
    }

    // A refusal comes back with the leaf's message.
    let message = unsafe { image_encoder::rows(ptr(&d_x) as *mut f32, tokens as u32, cols as u32, ptr(&d_idx) as *const i32, 5, ptr(&d_src), false, null) }.unwrap_err();
    assert!(message.contains("ignis_image_encoder_rows") && message.contains("rows must be 1 .. tokens"), "message: {message}");
    for b in [d_x, d_src, d_idx, d_delta, d_w, d_y, d_q, d_k, d_nw, d_pos, d_qo, d_ko] {
        dev.deallocate(b).unwrap();
    }
}
