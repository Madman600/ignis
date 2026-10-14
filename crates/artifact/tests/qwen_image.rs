//! The image model's artifact (spec image-model/01, AC 3): the converter's
//! trees pack and bind, the family is read from the header, a missing object
//! is refused by name, and the real inventory is the pinned checkpoint's.
//!
//! - `tests/fixtures/qwen_image_work/` is the work tree the converter wrote
//!   from its synthetic checkpoint (`tools/image-converter/convert.py
//!   fixture`: 2 DiT blocks, 2 text layers, a small VAE); committed, it runs
//!   everywhere.
//! - Machine-local: `convert.py fixture` itself is run with the converter's
//!   venv, and the real artifact is bound when it is on this machine (both
//!   skip when absent, CPU only).

use std::path::{Path, PathBuf};
use std::process::Command;

use ignis_artifact::packer::{pack, PackOptions, PackOutcome};
use ignis_artifact::qwen_image::{self, QwenImageGeometry, Role, MODEL_ID};
use ignis_artifact::{materialize, CpuDevice, Object, Reader};
use ignis_core::compute::ModelFamily;
use ignis_core::image_encoder::{embed_rows, TextEncoderConfig};
use ignis_core::image_vision::VisionTowerConfig;

const PYTHON: &str = "F:/ai/venvs/ignis-image/Scripts/python.exe";
const REAL_ARTIFACT: &str = "F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer";

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("ignis-qwen-image-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn committed_tree() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen_image_work")
}

/// Pack a work tree without consuming it.
fn pack_tree(work: &Path, artifact: &Path, options: impl FnOnce(&mut PackOptions)) -> Reader {
    let mut o = PackOptions::qwen_image(work.to_path_buf(), artifact.to_path_buf());
    o.keep_work = true;
    o.accept_status = vec!["fixture".into()];
    o.header_bytes = 1 << 20;
    options(&mut o);
    let outcome = pack(&o, &mut |_| {}).unwrap();
    assert!(matches!(outcome, PackOutcome::Finished { converter_merged: true, .. }), "{outcome:?}");
    Reader::open(artifact).unwrap()
}

#[test]
fn the_converters_synthetic_tree_packs_binds_and_names_its_family() {
    let scratch = Scratch::new("bind");
    let artifact = scratch.0.join("synthetic-v2.ninfer");
    let reader = pack_tree(&committed_tree(), &artifact, |_| {});

    assert_eq!(reader.identity().model_id, MODEL_ID);
    assert_eq!(ModelFamily::of_model_id(&reader.identity().model_id), Some(ModelFamily::QwenImage21));

    let bound = qwen_image::bind(&reader, &QwenImageGeometry::fixture()).expect("the converter's tree binds");
    assert_eq!(bound.plan.object_count, reader.objects().len());
    assert_eq!(reader.objects().len(), 351 + qwen_image::FRONTEND_FILES.len());
    // the token embedding stays on the host; the frontend is host-retained
    assert_eq!(bound.plan.host_tensor_objects.len(), 1);
    let embedding = bound.handles["text_encoder/embed_tokens.weight"];
    assert_eq!(bound.plan.host_tensor_objects[0].handle, embedding);
    assert_eq!(bound.plan.host_objects.len(), qwen_image::FRONTEND_FILES.len());

    // The sidecar beside the artifact carries the converter's record: every
    // stored tensor with its error, the VAE's convolutions, the revision.
    let sidecar = read_json(&ignis_artifact::packer::sidecar_path(&artifact));
    assert_eq!(sidecar["schema"], "image-converter-v1");
    assert_eq!(sidecar["objects"]["count"], reader.objects().len());
    let tensors = sidecar["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 351);
    for record in tensors {
        let name = record["object"].as_str().unwrap();
        let Some(Object::Tensor(t)) = reader.find(name) else { panic!("{name} is not in the container") };
        assert_eq!(record["bytes"], t.bytes, "{name}");
        assert_eq!(record["format"], t.format.name(), "{name}");
    }
    assert!(!sidecar["vae_convs"].as_array().unwrap().is_empty());
    assert_eq!(sidecar["source"]["revision"].as_str().unwrap().len(), 40);
}

#[test]
fn a_missing_or_unknown_object_or_another_identity_is_refused_by_name() {
    let scratch = Scratch::new("refuse");
    let artifact = scratch.0.join("synthetic-v2.ninfer");
    let reader = pack_tree(&committed_tree(), &artifact, |_| {});

    let mut more = QwenImageGeometry::fixture();
    more.dit.blocks = 3;
    let missing = qwen_image::bind(&reader, &more).unwrap_err().to_string();
    assert!(missing.contains("required artifact object is missing: transformer/transformer_blocks.2."), "{missing}");

    let mut fewer = QwenImageGeometry::fixture();
    fewer.text.layers = 1;
    let unknown = qwen_image::bind(&reader, &fewer).unwrap_err().to_string();
    assert!(unknown.contains("was not consumed") && unknown.contains("text_encoder/layers.1."), "{unknown}");

    let other = scratch.0.join("other-v2.ninfer");
    let reader = pack_tree(&committed_tree(), &other, |o| o.identity.model_id = "qwen3.8-27b".into());
    let refused = qwen_image::bind(&reader, &QwenImageGeometry::fixture()).unwrap_err().to_string();
    assert!(refused.contains("model_id qwen3.8-27b is not a Qwen-Image-2.1 artifact"), "{refused}");
}

/// The text encoder binds alone (spec image-model/03): its config resource
/// and its tensors, the embedding on the host, nothing else of the artifact.
/// The plan materializes, and its config is the checkpoint's, which the
/// encoder reads its eps and rope from.
#[test]
fn the_text_encoder_binds_alone() {
    let scratch = Scratch::new("text");
    let artifact = scratch.0.join("synthetic-v2.ninfer");
    let reader = pack_tree(&committed_tree(), &artifact, |_| {});
    let g = QwenImageGeometry::fixture();

    let bound = qwen_image::bind_text_encoder(&reader, &g.text).expect("the text encoder binds");
    let tensors = qwen_image::text_entries(&g.text).len();
    assert_eq!(bound.plan.object_count, tensors + 1);
    assert_eq!(bound.plan.device_objects.len(), tensors - 1);
    assert_eq!(bound.plan.host_tensor_objects.len(), 1);
    assert_eq!(bound.plan.host_tensor_objects[0].handle, bound.handles["text_encoder/embed_tokens.weight"]);
    assert_eq!(bound.plan.host_objects.len(), 1);
    assert!(bound.handles.keys().all(|n| n.starts_with("text_encoder/") || n == qwen_image::TEXT_ENCODER_CONFIG));

    let mut device = CpuDevice::new();
    let m = materialize(&reader, &bound.plan, &mut device, None).expect("the scoped plan materializes");
    let config = m.resource_bytes(bound.handles[qwen_image::TEXT_ENCODER_CONFIG]).unwrap();
    assert_eq!(config, std::fs::read(committed_tree().join("frontend/text_encoder_config.json")).unwrap());
    let parsed = TextEncoderConfig::parse(config, &g.text).expect("the fixture's text config");
    assert_eq!((parsed.rms_norm_eps, parsed.rope_theta, parsed.mrope_section), (1e-6, 5e6, [2, 3, 3]));
    let embedding = m.host_tensor_bytes(bound.handles["text_encoder/embed_tokens.weight"]).unwrap();
    assert_eq!(embedding.len() as u64, g.text.vocab * g.text.hidden * 2);
    assert_eq!(embed_rows(embedding, g.text.hidden as usize, &[7]).unwrap().len(), g.text.hidden as usize);

    let mut fewer = g.text.clone();
    fewer.layers = 3;
    let missing = qwen_image::bind_text_encoder(&reader, &fewer).unwrap_err().to_string();
    assert!(missing.contains("required artifact object is missing: text_encoder/layers.2."), "{missing}");
}

/// Spec image-model/03 (GitHub #320): the vision tower binds alone, the
/// config resource with it (its `vision_config` names the DeepStack taps),
/// every matrix a W8 payload with its scale plane.
#[test]
fn the_vision_tower_binds_alone() {
    let scratch = Scratch::new("vision");
    let artifact = scratch.0.join("synthetic-v2.ninfer");
    let reader = pack_tree(&committed_tree(), &artifact, |_| {});
    let g = QwenImageGeometry::fixture();

    let bound = qwen_image::bind_vision(&reader, &g.vision).expect("the vision tower binds");
    let tensors = qwen_image::vision_entries(&g.vision).len();
    assert_eq!(bound.plan.object_count, tensors + 1);
    assert_eq!(bound.plan.device_objects.len(), tensors);
    assert_eq!(bound.plan.host_objects.len(), 1);
    assert!(bound.handles.keys().all(|n| n.starts_with("vision/") || n == qwen_image::TEXT_ENCODER_CONFIG));

    let mut device = CpuDevice::new();
    let m = materialize(&reader, &bound.plan, &mut device, None).expect("the scoped plan materializes");
    let config = m.resource_bytes(bound.handles[qwen_image::TEXT_ENCODER_CONFIG]).unwrap();
    let parsed = VisionTowerConfig::parse(config, &g.vision).expect("the fixture's vision config");
    assert_eq!(parsed, VisionTowerConfig { num_heads: 2, deepstack_layers: vec![0, 1, 2] });
    for name in ["vision/patch_embedding", "vision/layers/2/mlp/fc2", "vision/merger/fc2", "vision/deepstack/2/fc1"] {
        let view = m.device_view(bound.handles[name]).unwrap();
        assert!(view.scale_plane().is_some(), "{name} has a scale plane");
    }

    let mut deeper = g.vision.clone();
    deeper.depth = 4;
    let missing = qwen_image::bind_vision(&reader, &deeper).unwrap_err().to_string();
    assert!(missing.contains("required artifact object is missing: vision/layers/3/"), "{missing}");
}

/// The real geometry's DiT, VAE and text-decoder objects are the pinned
/// checkpoint's tensors (`tools/image-converter/pin.json`, read from the
/// safetensors headers at the pinned revision), by name and shape; the vision
/// tower, renamed to the 27B's names, by count. The packer's `weights_id`
/// names the pin's revision, so the two cannot drift apart.
#[test]
fn the_real_inventory_is_the_pinned_checkpoints() {
    let pin = read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/image-converter/pin.json"));
    assert_eq!(pin["revision"], "fc13ba1986fcb905854dc8de2f52eee395f49b5b");
    let revision = pin["revision"].as_str().unwrap();
    assert!(ignis_artifact::packer::qwen_image_identity().weights_id.ends_with(&revision[..7]));
    let g = QwenImageGeometry::qwen_image_21_turbo();
    let expected = |component: &str, rename: &dyn Fn(&str) -> Option<String>| {
        let mut out: Vec<(String, Vec<u64>)> = pin["tensors"][component]
            .as_object()
            .unwrap()
            .iter()
            .filter_map(|(name, entry)| {
                let shape = entry[2].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
                rename(name).map(|object| (object, shape))
            })
            .collect();
        out.sort();
        out
    };
    let ours = |entries: Vec<qwen_image::Entry>| {
        let mut out: Vec<(String, Vec<u64>)> = entries.into_iter().map(|e| (e.name, e.shape)).collect();
        out.sort();
        out
    };
    assert_eq!(ours(qwen_image::dit_entries(&g.dit)), expected("transformer", &|n| Some(format!("transformer/{n}"))));
    assert_eq!(ours(qwen_image::vae_entries(&g.vae)), expected("vae", &|n| Some(format!("vae/{n}"))));
    let text = expected("text_encoder", &|n| {
        n.strip_prefix("model.language_model.").filter(|rest| *rest != "norm.weight").map(|rest| format!("text_encoder/{rest}"))
    });
    assert_eq!(ours(qwen_image::text_entries(&g.text)), text);
    let visual = pin["tensors"]["text_encoder"].as_object().unwrap().keys().filter(|n| n.starts_with("model.visual.")).count();
    assert_eq!(qwen_image::vision_entries(&g.vision).len(), visual);
}

#[test]
fn the_converters_python_fixture_packs_and_binds() {
    if !Path::new(PYTHON).exists() {
        eprintln!("skip: {PYTHON} does not exist");
        return;
    }
    let tool = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/image-converter");
    let scratch = Scratch::new("python");
    let output = Command::new(PYTHON)
        .current_dir(&tool)
        .env("CUDA_VISIBLE_DEVICES", "")
        .arg("convert.py")
        .arg("fixture")
        .arg("--out")
        .arg(&scratch.0)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "convert.py fixture failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let artifact = scratch.0.join("synthetic-v2.ninfer");
    let reader = pack_tree(&scratch.0.join("work"), &artifact, |_| {});
    qwen_image::bind(&reader, &QwenImageGeometry::fixture()).expect("a fresh converter tree binds");
    // the converter still writes the committed tree, byte for byte (the
    // synthetic checkpoint is seeded): every unit's DONE, which hashes each file
    for unit in std::iter::once("frontend").chain(qwen_image::WORK_UNITS) {
        let done = |root: &Path| read_json(&root.join(unit).join("DONE"));
        assert_eq!(done(&scratch.0.join("work")), done(&committed_tree()), "{unit}");
    }
}

/// The real artifact (milestone 2 of #318), bound on the CPU when it is on
/// this machine: every object of the inventory, nothing else, and the
/// converter's record of every tensor beside it.
#[test]
fn the_real_artifact_binds() {
    let path = Path::new(REAL_ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {REAL_ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap();
    assert_eq!(ModelFamily::of_model_id(&reader.identity().model_id), Some(ModelFamily::QwenImage21));
    let bound = qwen_image::bind(&reader, &QwenImageGeometry::qwen_image_21_turbo()).expect("the real artifact binds");
    assert_eq!(bound.plan.object_count, reader.objects().len());
    let host: Vec<_> = qwen_image::entries(&QwenImageGeometry::qwen_image_21_turbo())
        .into_iter()
        .filter(|e| e.role == Role::Host)
        .map(|e| e.name)
        .collect();
    assert_eq!(host, ["text_encoder/embed_tokens.weight"]);
    let sidecar = read_json(&ignis_artifact::packer::sidecar_path(path));
    assert_eq!(sidecar["source"]["revision"], "fc13ba1986fcb905854dc8de2f52eee395f49b5b");
    assert_eq!(sidecar["all_finite"], true);
    let bound_fp8 = sidecar["fp8_bound"].as_f64().unwrap();
    for record in sidecar["tensors"].as_array().unwrap() {
        if record["encoding"] == "fp8" {
            assert!(record["rel_error"].as_f64().unwrap() <= bound_fp8, "{}", record["object"]);
        }
    }
}
