//! The image model's weight descriptors over a full `qwen_image::bind` (spec image-model/03,
//! GitHub #320), on the converter's synthetic artifact and a `CpuDevice`: no GPU, but the
//! descriptors are the `cuda` feature's, so it builds with that feature.
//!
//! - `TextEncoderWeights::new` reads the handles of the whole-artifact bind as it does the scoped
//!   `bind_text_encoder`'s: the names are the same (`qwen_image::text_entries`).
//! - `VisionWeights::new` refuses a geometry or a head count the leaf does not run.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use ignis_artifact::packer::{pack, PackOptions, PackOutcome};
use ignis_artifact::qwen_image::{self, QwenImageGeometry};
use ignis_artifact::{materialize, CpuDevice, Reader};
use ignis_core::image_encoder::{TextEncoderConfig, TextEncoderWeights};
use ignis_core::image_vision::{VisionTowerConfig, VisionWeights};

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The committed synthetic tree (`crates/artifact/tests/fixtures/qwen_image_work`), packed.
fn packed(tag: &str) -> (Scratch, Reader) {
    let dir = std::env::temp_dir().join(format!("ignis-image-weights-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let scratch = Scratch(dir);
    let tree = Path::new(env!("CARGO_MANIFEST_DIR")).join("../artifact/tests/fixtures/qwen_image_work");
    let artifact = scratch.0.join("synthetic-v2.ninfer");
    let mut o = PackOptions::qwen_image(tree, artifact.clone());
    o.keep_work = true;
    o.accept_status = vec!["fixture".into()];
    o.header_bytes = 1 << 20;
    let outcome = pack(&o, &mut |_| {}).unwrap();
    assert!(matches!(outcome, PackOutcome::Finished { converter_merged: true, .. }), "{outcome:?}");
    let reader = Reader::open(&artifact).unwrap();
    (scratch, reader)
}

#[test]
fn the_text_descriptors_read_the_whole_artifacts_handles() {
    let (_scratch, reader) = packed("text");
    let g = QwenImageGeometry::fixture();
    let full = qwen_image::bind(&reader, &g).expect("the whole artifact binds");
    let mut device = CpuDevice::new();
    let m = materialize(&reader, &full.plan, &mut device, None).expect("the whole plan materializes");
    let config = m.resource_bytes(full.handles[qwen_image::TEXT_ENCODER_CONFIG]).unwrap();
    let config = TextEncoderConfig::parse(config, &g.text).expect("the fixture's text config");
    let weights = TextEncoderWeights::new(&m, &full.handles, &g.text, config).expect("the full bind's handles name every layer's tensors");
    let desc = weights.descriptor();
    assert_eq!(desc.layer_count as usize, g.text.layers);
    let layers = unsafe { std::slice::from_raw_parts(desc.layers, g.text.layers) };
    assert!(layers.iter().all(|l| !l.q_proj.is_null() && !l.down_proj.is_null() && !l.input_norm.is_null()));

    // The scoped bind gives the same descriptors: same device bases, layer by layer.
    let scoped = qwen_image::bind_text_encoder(&reader, &g.text).expect("the text encoder binds alone");
    let mut device = CpuDevice::new();
    let ms = materialize(&reader, &scoped.plan, &mut device, None).expect("the scoped plan materializes");
    let sc = TextEncoderConfig::parse(ms.resource_bytes(scoped.handles[qwen_image::TEXT_ENCODER_CONFIG]).unwrap(), &g.text).unwrap();
    let only = TextEncoderWeights::new(&ms, &scoped.handles, &g.text, sc).expect("the scoped handles");
    assert_eq!(only.descriptor().layer_count, desc.layer_count);

    // A name that is not bound is named.
    let mut missing = full.handles.clone();
    missing.remove("text_encoder/layers.1.mlp.up_proj.weight");
    let refused = TextEncoderWeights::new(&m, &missing, &g.text, only.config).unwrap_err();
    assert!(refused.contains("text_encoder/layers.1.mlp.up_proj.weight is not bound"), "{refused}");
}

#[test]
fn the_tower_refuses_a_geometry_or_a_head_count_the_leaf_does_not_run() {
    let (_scratch, reader) = packed("vision");
    let g = QwenImageGeometry::fixture();
    let bound = qwen_image::bind_vision(&reader, &g.vision).expect("the vision tower binds");
    let mut device = CpuDevice::new();
    let m = materialize(&reader, &bound.plan, &mut device, None).expect("the scoped plan materializes");
    let config = VisionTowerConfig::parse(m.resource_bytes(bound.handles[qwen_image::TEXT_ENCODER_CONFIG]).unwrap(), &g.vision).unwrap();

    // The fixture's 3-block tower is not the leaf's 27 blocks.
    let refused = VisionWeights::new(&m, &bound.handles, &g.vision, &config).unwrap_err();
    assert!(refused.contains("is not the one the leaf runs"), "{refused}");

    // The real geometry with a head count other than the leaf's 16 (the geometry is checked first,
    // so no handle is read).
    let real = QwenImageGeometry::qwen_image_21_turbo().vision;
    let heads = VisionTowerConfig { num_heads: 2, deepstack_layers: vec![8, 16, 24] };
    let refused = VisionWeights::new(&m, &bound.handles, &real, &heads).unwrap_err();
    assert!(refused.contains("is not the one the leaf runs (16 heads)"), "{refused}");
    let taps = VisionTowerConfig { num_heads: 16, deepstack_layers: vec![8, 16] };
    assert!(VisionWeights::new(&m, &bound.handles, &real, &taps).is_err());
}
