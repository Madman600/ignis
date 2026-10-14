//! The image encoder's input (spec image-model/03, ACs 1, 2 and 6): the
//! server's tokens, image preprocessing and mrope positions against the
//! pipeline's own.
//!
//! - `tests/fixtures/image_encoder_input.json` is recorded from diffusers'
//!   `QwenImage21Pipeline` on the pinned processor by
//!   `tools/image-converter/record_encoder_input.py` (its docstring has the
//!   command). The composite, preprocessing and mrope checks need nothing
//!   else and run everywhere.
//! - The token checks need the artifact's tokenizer, and the oracle checks
//!   spec 01's recordings: machine-local, they skip when absent
//!   (`$IGNIS_IMAGE_ARTIFACT`, `$IGNIS_IMAGE_ORACLE`, else the owner's layout).

use std::path::{Path, PathBuf};

use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use ignis_artifact::vision::Grid;
use ignis_artifact::Reader;
use ignis_server::image_encoder_input::{mrope_positions, on_white, ConditionImage, EncoderFrontend, ImageProcessor};

fn fixture() -> JsonValue {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/image_encoder_input.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn machine_local(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| default.to_string()))
}

/// The recipe's `pattern`: r = 3x + 5y, g = 7x + 2y + 85, b = xy + 170,
/// a = 11x + 13y, each mod 256.
fn pattern(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height as u64 {
        for x in 0..width as u64 {
            rgba.extend([3 * x + 5 * y, 7 * x + 2 * y + 85, x * y + 170, 11 * x + 13 * y].map(|v| (v % 256) as u8));
        }
    }
    rgba
}

fn ints<T: TryFrom<i64>>(value: &JsonValue) -> Vec<T> {
    value.as_array().unwrap().iter().map(|v| T::try_from(v.as_i64().unwrap()).ok().unwrap()).collect()
}

fn grid(value: &JsonValue) -> Grid {
    let [t, h, w] = ints::<u32>(value)[..] else { panic!("a grid is three numbers") };
    Grid { t, h, w }
}

fn sha256_le(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for v in values {
        hasher.update(v.to_le_bytes());
    }
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// AC 2: Pillow's paste over white, every value at every alpha.
#[test]
fn the_white_composite_is_pillows_paste() {
    let fixture = fixture();
    let composite = &fixture["composite"];
    // table[a][v]: pixel (v, a) of a 256 x 256 RGBA image holding (v, v, v, a)
    let rgba: Vec<u8> = (0..256u32).flat_map(|a| (0..256u32).flat_map(move |v| [v as u8, v as u8, v as u8, a as u8])).collect();
    let rgb = on_white(&rgba);
    let table: Vec<u8> = rgb.chunks_exact(3).map(|px| {
        assert!(px[0] == px[1] && px[1] == px[2]);
        px[0]
    }).collect();
    for (alpha, row) in composite["alphas"].as_object().unwrap() {
        let a: usize = alpha.parse().unwrap();
        assert_eq!(table[a * 256..(a + 1) * 256], ints::<u8>(row)[..], "alpha {a}");
    }
    let digest: String = Sha256::digest(&table).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(digest, composite["table_sha256"].as_str().unwrap(), "the whole 256 x 256 table");
    // alpha 0 is white, alpha 255 is the pixel itself
    assert!(table[..256].iter().all(|&v| v == 255));
    assert!(table[255 * 256..].iter().enumerate().all(|(v, &out)| out as usize == v));
}

/// AC 2: 1:1, 16:9, 9:16, a 2-pixel-wide image and an odd size, bit for bit
/// (bound 0): the grid, and the sha256 of `pixel_values`.
#[test]
fn condition_images_are_prepared_as_the_processor_prepares_them() {
    let fixture = fixture();
    let processor = ImageProcessor::from_json(&fixture["image_processor"]).unwrap();
    for case in fixture["preprocess"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (width, height) = (case["width"].as_u64().unwrap() as u32, case["height"].as_u64().unwrap() as u32);
        let rgba = pattern(width, height);
        let got = processor.prepare(ConditionImage { width, height, rgba: &rgba }).unwrap();
        assert_eq!(got.grid, grid(&case["image_grid_thw"]), "{name}: image_grid_thw");
        let [rows, columns] = ints::<usize>(&case["shape"])[..] else { panic!() };
        assert_eq!(got.pixels.len(), rows * columns, "{name}: pixel_values' size");
        let first: Vec<f32> = case["first"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let last: Vec<f32> = case["last"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(got.pixels[..8], first[..], "{name}: the first values");
        assert_eq!(got.pixels[got.pixels.len() - 8..], last[..], "{name}: the last values");
        assert_eq!(sha256_le(&got.pixels), case["sha256"].as_str().unwrap(), "{name}: pixel_values");
    }
}

/// AC 6: transformers' positions for every recorded layout: text-to-image
/// (one arange), 1, 2, 3 and 10 condition images.
#[test]
fn mrope_positions_are_transformers() {
    let fixture = fixture();
    let pad = fixture["image_token_id"].as_u64().unwrap() as u32;
    let mut images_seen = Vec::new();
    for case in fixture["prompts"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ids = ints::<u32>(&case["input_ids"]);
        let grids: Vec<Grid> = case["image_grid_thw"].as_array().unwrap().iter().map(grid).collect();
        let want: Vec<i32> = case["positions"].as_array().unwrap().iter().flat_map(ints::<i32>).collect();
        assert_eq!(mrope_positions(&ids, pad, &grids).unwrap(), want, "{name}");
        images_seen.push(grids.len());
    }
    assert!([0, 1, 2, 3, 10].iter().all(|n| images_seen.contains(n)), "layouts: {images_seen:?}");
    // a run of pads that is not the next grid's size is refused, not guessed (the 3-image case's
    // grids are 144, 112 and 64 merged tokens)
    let case = &fixture["prompts"].as_array().unwrap().iter().find(|c| c["name"] == "3 images").unwrap();
    let ids = ints::<u32>(&case["input_ids"]);
    let grids: Vec<Grid> = case["image_grid_thw"].as_array().unwrap().iter().map(grid).collect();
    assert!(mrope_positions(&ids, pad, &grids[..2]).is_err(), "a run without a grid");
    assert!(mrope_positions(&ids, pad, &[grids[1], grids[0], grids[2]]).is_err(), "grids out of order");
}

fn frontend() -> Option<EncoderFrontend> {
    let path = machine_local("IGNIS_IMAGE_ARTIFACT", "F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer");
    if !path.exists() {
        eprintln!("SKIP: {} (the image artifact) is not here", path.display());
        return None;
    }
    Some(EncoderFrontend::from_reader(&Reader::open(&path).unwrap()).expect("the artifact's front end"))
}

/// AC 1 (and 6 through the whole input): the pipeline's tokens for an empty
/// prompt, a CJK prompt, a 2,000-token prompt, and 1, 2, 3 and 10 condition
/// images (one with an empty prompt).
#[test]
fn prompts_tokenize_as_the_pipeline_tokenizes_them() {
    let Some(frontend) = frontend() else { return };
    let fixture = fixture();
    for case in fixture["prompts"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let images: Vec<(u32, u32, Vec<u8>)> = case["images"].as_array().unwrap().iter().map(|s| {
            let [w, h] = ints::<u32>(s)[..] else { panic!() };
            (w, h, pattern(w, h))
        }).collect();
        let views: Vec<ConditionImage> = images.iter().map(|(w, h, rgba)| ConditionImage { width: *w, height: *h, rgba }).collect();
        let got = frontend.input(case["prompt"].as_str().unwrap(), &views).unwrap();
        let want = ints::<u32>(&case["input_ids"]);
        assert_eq!(got.input_ids, want, "{name}: input_ids");
        assert_eq!(got.drop_idx, case["drop_idx"].as_u64().unwrap() as usize, "{name}: drop_idx");
        let grids: Vec<Grid> = case["image_grid_thw"].as_array().unwrap().iter().map(grid).collect();
        assert_eq!(got.images.iter().map(|i| i.grid).collect::<Vec<_>>(), grids, "{name}: grids");
        let positions: Vec<i32> = case["positions"].as_array().unwrap().iter().flat_map(ints::<i32>).collect();
        assert_eq!(got.positions, positions, "{name}: positions");
        let pad = fixture["image_token_id"].as_u64().unwrap() as u32;
        let pads: Vec<u32> = (0..).zip(&want).filter(|&(_, &id)| id == pad).map(|(row, _)| row).collect();
        assert_eq!(got.image_pad_rows, pads, "{name}: image-pad rows");
        if name == "2000 tokens" {
            assert!(want.len() >= 2000, "{} tokens", want.len());
        }
    }
    // an image pad written in the prompt would take an image's place
    assert!(frontend.input("a <|image_pad|> here", &[]).is_err());
}

/// A little-endian C-order .npy's shape and payload; None when the file is missing.
fn load_npy(path: &Path) -> Option<(Vec<usize>, Vec<u8>)> {
    let raw = std::fs::read(path).ok()?;
    assert_eq!(&raw[..6], b"\x93NUMPY", "{} is not a .npy", path.display());
    let (len, at) = if raw[6] == 1 { (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10) } else { (u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize, 12) };
    let header = std::str::from_utf8(&raw[at..at + len]).unwrap();
    assert!(header.contains("'fortran_order': False"), "{} is not C order", path.display());
    let shape_at = header.find("'shape': (").unwrap() + 10;
    let shape: Vec<usize> = header[shape_at..header[shape_at..].find(')').unwrap() + shape_at].split(',').filter_map(|s| s.trim().parse().ok()).collect();
    Some((shape, raw[at + len..].to_vec()))
}

fn i64s(bytes: &[u8]) -> Vec<i64> {
    bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// ACs 1, 2 and 6 on spec 01's recordings: every case (t2i, t2i-2k, edit),
/// every field bit for bit. The edit case's condition image is the VAE's input read back, which
/// is exact: `2 · (u / 255) − 1` of the recovered bytes is the recording.
#[test]
fn the_oracle_cases_are_reproduced_exactly() {
    let oracle = machine_local("IGNIS_IMAGE_ORACLE", "F:/ai/models/ignis-image-2.1-turbo/oracle");
    if !oracle.join("edit/case.json").exists() {
        eprintln!("SKIP: {} (spec 01's recordings) is not here", oracle.display());
        return;
    }
    let Some(frontend) = frontend() else { return };
    for name in ["t2i", "t2i-2k", "edit"] {
        let dir = oracle.join(name);
        let case: JsonValue = serde_json::from_slice(&std::fs::read(dir.join("case.json")).unwrap()).unwrap();
        let npy = |field: &str| load_npy(&dir.join(format!("{field}.npy"))).unwrap_or_else(|| panic!("{name}/{field}.npy"));
        let mut rgba = Vec::new();
        let mut size = (0, 0);
        if name == "edit" {
            let (shape, bytes) = npy("condition_vae_input");
            let [1, 4, 1, h, w] = shape[..] else { panic!("condition_vae_input is {shape:?}") };
            let planes = f32s(&bytes);
            let bytes: Vec<u8> = planes.iter().map(|&x| ((x + 1.0) * 127.5).round() as u8).collect();
            for (u, x) in bytes.iter().zip(&planes) {
                assert_eq!(2.0f32 * (*u as f32 / 255.0f32) - 1.0f32, *x, "the VAE's input read back is not exact");
            }
            rgba = (0..h * w).flat_map(|p| (0..4).map(move |c| (c, p))).map(|(c, p)| bytes[c * h * w + p]).collect();
            size = (w as u32, h as u32);
        }
        let images = if name == "edit" { vec![ConditionImage { width: size.0, height: size.1, rgba: &rgba }] } else { vec![] };
        let got = frontend.input(case["case"]["prompt"].as_str().unwrap(), &images).unwrap();

        let want: Vec<u32> = i64s(&npy("input_ids").1).into_iter().map(|id| id as u32).collect();
        assert_eq!(got.input_ids, want, "{name}: input_ids");
        assert_eq!(got.drop_idx as u64, case["drop_idx"].as_u64().unwrap(), "{name}: drop_idx");
        let positions: Vec<i32> = i64s(&npy("mrope_position_ids").1).into_iter().map(|p| p as i32).collect();
        assert_eq!(got.positions, positions, "{name}: mrope_position_ids");
        let types = i64s(&npy("mm_token_type_ids").1);
        let pads: Vec<u32> = (0..).zip(&types).filter(|&(_, &t)| t == 1).map(|(row, _)| row).collect();
        assert_eq!(got.image_pad_rows, pads, "{name}: image-pad rows");
        if name == "edit" {
            let grid_thw = i64s(&npy("image_grid_thw").1);
            assert_eq!(got.images.len(), 1);
            assert_eq!([got.images[0].grid.t, got.images[0].grid.h, got.images[0].grid.w].map(i64::from)[..], grid_thw[..], "image_grid_thw");
            let (shape, bytes) = npy("pixel_values");
            assert_eq!(got.images[0].pixels.len(), shape.iter().product::<usize>(), "pixel_values' size");
            let mismatches = got.images[0].pixels.iter().zip(f32s(&bytes)).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(mismatches, 0, "pixel_values: bound 0, bit for bit");
        }
    }
}
