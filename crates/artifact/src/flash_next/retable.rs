//! Put Qwen's FP8 n-gram table into a packed Flash-Next container (spec
//! flash-next/08), and re-pin the MTP companion to the result.
//!
//! [`retable`] copies a main container object for object, in directory
//! order, replacing only the n-gram table: its INT4 rows become the FP8
//! release's rows, 160 E4M3FN bytes verbatim and the source's one bf16
//! `weight_scale` after each (`FP8_E4M3FN_ROW_BF16S`, 162 bytes a row).
//! Every other payload is copied byte for byte; offsets after the table move.
//! The new container keeps `model_id`, extends `weights_id`, and gets a
//! sidecar copied from the old one with its file size, object count, table
//! row size and a `retable` provenance record updated.
//!
//! [`repin_companion`] copies the companion and writes its sidecar's
//! `pair.main` for the retabled container (layout.md §13.2): the head was
//! calibrated on the INT4 table, which the sidecar records; acceptance is
//! measured, a mismatched pair never changes a token.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::{fail, mtp, ngram_table_name, FlashNextGeometry, MODEL_ID};
use crate::packer::sidecar_path;
use crate::writer::ContainerWriter;
use crate::{ArtifactIdentity, NumericFormat, Object, Reader, Result, StorageLayout};

/// Rows streamed per read of a source shard.
const CHUNK_ROWS: u64 = 1 << 16;
/// `weights_id` gains this, so no record of the INT4 container matches it.
pub const WEIGHTS_ID_SUFFIX: &str = "-ngramfp8";

/// One source shard's table tensor: where its rows are in which file.
#[derive(Debug, Clone)]
pub struct SourceShard {
    pub file: PathBuf,
    pub data_offset: u64,
    pub rows: u64,
}

/// The FP8 release's n-gram table, located in its safetensors shards.
#[derive(Debug, Clone)]
pub struct Fp8TableSource {
    pub shards: Vec<SourceShard>,
    /// The table's one `weight_scale`, bf16 little-endian.
    pub scale: [u8; 2],
    pub columns: u64,
}

impl Fp8TableSource {
    pub fn rows(&self) -> u64 {
        self.shards.iter().map(|s| s.rows).sum()
    }
}

fn io<E: std::fmt::Display>(what: impl std::fmt::Display) -> impl FnOnce(E) -> crate::ArtifactError {
    move |e| fail(format!("{what}: {e}"))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A safetensors file's header: tensor name -> (dtype, shape, absolute data offset).
fn safetensors_header(path: &Path) -> Result<BTreeMap<String, (String, Vec<u64>, u64)>> {
    let mut file = File::open(path).map_err(io(path.display()))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).map_err(io(path.display()))?;
    let len = u64::from_le_bytes(len);
    let mut header = vec![0u8; len as usize];
    file.read_exact(&mut header).map_err(io(path.display()))?;
    let value: Value = serde_json::from_slice(&header).map_err(io(path.display()))?;
    let mut out = BTreeMap::new();
    for (name, entry) in value.as_object().ok_or_else(|| fail(format!("{} header is not an object", path.display())))? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry["dtype"].as_str().unwrap_or_default().to_owned();
        let shape = entry["shape"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        let begin = entry["data_offsets"][0].as_u64().ok_or_else(|| fail(format!("{name} has no data_offsets")))?;
        out.insert(name.clone(), (dtype, shape, 8 + len + begin));
    }
    Ok(out)
}

/// Locate the FP8 table in `dir` (the release's `model.safetensors.index.json`
/// and the shards it names): `shard_0..` in order, each `F8_E4M3`
/// `[rows, head_dim]`, and the `BF16` `[1]` `weight_scale`.
pub fn read_fp8_source(dir: &Path, geometry: &FlashNextGeometry) -> Result<Fp8TableSource> {
    let index_path = dir.join("model.safetensors.index.json");
    let index: Value = serde_json::from_slice(&std::fs::read(&index_path).map_err(io(index_path.display()))?)
        .map_err(io(index_path.display()))?;
    let map = index["weight_map"].as_object().ok_or_else(|| fail("the index has no weight_map"))?;
    let prefix = format!("model.language_model.layers.{}.ple.ple_embedding.ngram_embedding", geometry.ple_layer);
    let mut headers: BTreeMap<String, BTreeMap<String, (String, Vec<u64>, u64)>> = BTreeMap::new();
    let mut lookup = |name: &str| -> Result<(PathBuf, String, Vec<u64>, u64)> {
        let file = map.get(name).and_then(Value::as_str).ok_or_else(|| fail(format!("the index names no {name}")))?;
        if !headers.contains_key(file) {
            headers.insert(file.to_owned(), safetensors_header(&dir.join(file))?);
        }
        let (dtype, shape, offset) = headers[file]
            .get(name)
            .cloned()
            .ok_or_else(|| fail(format!("{file} holds no {name}")))?;
        Ok((dir.join(file), dtype, shape, offset))
    };

    let (scale_file, dtype, shape, offset) = lookup(&format!("{prefix}.weight_scale"))?;
    if dtype != "BF16" || shape != [1] {
        return Err(fail(format!("{prefix}.weight_scale is {dtype} {shape:?}, not BF16 [1]")));
    }
    let mut scale = [0u8; 2];
    let mut file = File::open(&scale_file).map_err(io(scale_file.display()))?;
    file.seek(SeekFrom::Start(offset)).map_err(io(scale_file.display()))?;
    file.read_exact(&mut scale).map_err(io(scale_file.display()))?;

    let columns = geometry.ngram_head_dim;
    let mut shards = Vec::new();
    while map.contains_key(&format!("{prefix}.shard_{}.weight", shards.len())) {
        let name = format!("{prefix}.shard_{}.weight", shards.len());
        let (file, dtype, shape, data_offset) = lookup(&name)?;
        if dtype != "F8_E4M3" || shape.len() != 2 || shape[1] != columns {
            return Err(fail(format!("{name} is {dtype} {shape:?}, not F8_E4M3 [rows, {columns}]")));
        }
        shards.push(SourceShard { file, data_offset, rows: shape[0] });
    }
    if shards.is_empty() {
        return Err(fail(format!("the index names no {prefix}.shard_0.weight")));
    }
    Ok(Fp8TableSource { shards, scale, columns })
}

/// What [`retable`] wrote.
#[derive(Debug, Clone)]
pub struct RetableReport {
    pub path: PathBuf,
    pub bytes: u64,
    pub objects: usize,
    pub table_rows: u64,
    /// Per source shard: SHA-256 of the FP8 bytes copied from it.
    pub shard_sha256: Vec<String>,
}

/// Write `out`: `main` with its n-gram table replaced by `source`'s FP8 rows,
/// and `out`'s sidecar. `progress` hears each source shard as it lands.
pub fn retable(
    main: &Path,
    source: &Fp8TableSource,
    out: &Path,
    geometry: &FlashNextGeometry,
    progress: &mut dyn FnMut(&str),
) -> Result<RetableReport> {
    let reader = Reader::open(main)?;
    let identity = reader.identity().clone();
    if identity.model_id != MODEL_ID {
        return Err(fail(format!("{} is {}, not a Flash-Next main container", main.display(), identity.model_id)));
    }
    let table_name = ngram_table_name(geometry);
    let rows = match reader.find(&table_name) {
        Some(Object::Tensor(t)) if t.format == NumericFormat::Q4G32F16S && t.shape == [t.shape[0], source.columns] => t.shape[0],
        Some(Object::Tensor(t)) => {
            return Err(fail(format!("{table_name} is {} {:?}: only an INT4 table is retabled", t.format.name(), t.shape)))
        }
        _ => return Err(fail(format!("{} holds no {table_name}", main.display()))),
    };
    if source.rows() != rows {
        return Err(fail(format!("the FP8 source holds {} table rows, the container {rows}", source.rows())));
    }

    let new_identity = ArtifactIdentity {
        model_id: identity.model_id.clone(),
        weights_id: format!("{}{WEIGHTS_ID_SUFFIX}", identity.weights_id),
    };
    // The directory grows by a few dozen bytes (a longer format name and
    // weights_id); one more page holds it.
    let header_bytes = reader.payload_offset() + 4096;
    let mut writer = ContainerWriter::create(out, new_identity.clone(), header_bytes)?;
    let mut shard_sha256 = Vec::with_capacity(source.shards.len());
    let row_out = source.columns as usize + 2;
    for object in reader.objects() {
        match object {
            Object::Tensor(t) if t.name == table_name => {
                writer.begin_tensor(&t.name, NumericFormat::Fp8E4M3FnRowBf16S, StorageLayout::RowInterleavedV1, &t.shape)?;
                for (n, shard) in source.shards.iter().enumerate() {
                    let mut file = File::open(&shard.file).map_err(io(shard.file.display()))?;
                    file.seek(SeekFrom::Start(shard.data_offset)).map_err(io(shard.file.display()))?;
                    let mut hash = Sha256::new();
                    let mut done = 0u64;
                    while done < shard.rows {
                        let take = CHUNK_ROWS.min(shard.rows - done) as usize;
                        let mut codes = vec![0u8; take * source.columns as usize];
                        file.read_exact(&mut codes).map_err(io(shard.file.display()))?;
                        hash.update(&codes);
                        let mut rows_out = vec![0u8; take * row_out];
                        for (row, codes) in rows_out.chunks_exact_mut(row_out).zip(codes.chunks_exact(source.columns as usize)) {
                            row[..codes.len()].copy_from_slice(codes);
                            row[codes.len()..].copy_from_slice(&source.scale);
                        }
                        writer.write(&mut rows_out.as_slice())?;
                        done += take as u64;
                    }
                    shard_sha256.push(hex(&hash.finalize()));
                    progress(&format!("table shard {}/{}", n + 1, source.shards.len()));
                }
                writer.end_object()?;
            }
            Object::Tensor(t) => {
                writer.begin_tensor(&t.name, t.format, t.layout, &t.shape)?;
                writer.write(&mut reader.payload_at(object)?.data)?;
                writer.end_object()?;
            }
            Object::Resource(r) => {
                writer.begin_resource(&r.name, r.encoding, r.bytes)?;
                writer.write(&mut reader.payload_at(object)?.data)?;
                writer.end_object()?;
            }
        }
    }
    let objects = reader.objects().len();
    let bytes = writer.finish()?;

    let old_sidecar = sidecar_path(main);
    let mut sidecar = if old_sidecar.exists() {
        match serde_json::from_slice(&std::fs::read(&old_sidecar).map_err(io(old_sidecar.display()))?)
            .map_err(io(old_sidecar.display()))?
        {
            Value::Object(map) => map,
            _ => return Err(fail(format!("{} is not a JSON object", old_sidecar.display()))),
        }
    } else {
        Map::new()
    };
    let stem = out.file_stem().and_then(|s| s.to_str()).ok_or_else(|| fail("the output has no file stem"))?;
    sidecar.insert("recipe_id".into(), json!(stem));
    sidecar.insert("artifact".into(), json!({"bytes": bytes}));
    sidecar.insert("objects".into(), json!({"count": objects}));
    if let Some(Value::Object(ngram)) = sidecar.get_mut("ngram") {
        let hot = ngram.get("hot_rows").and_then(Value::as_u64).unwrap_or(0);
        ngram.insert("row_bytes".into(), json!(row_out));
        ngram.insert("hot_rows_bytes".into(), json!(hot * row_out as u64));
        ngram.insert("format".into(), json!(NumericFormat::Fp8E4M3FnRowBf16S.name()));
    }
    sidecar.insert(
        "retable".into(),
        json!({
            "spec": "flash-next/08",
            "from": {
                "file": main.file_name().and_then(|n| n.to_str()),
                "weights_id": identity.weights_id,
                "content_hash": hex(&reader.content_hash()),
                "bytes": reader.file_bytes(),
            },
            "table": {
                "format": NumericFormat::Fp8E4M3FnRowBf16S.name(),
                "rows": rows,
                "scale_bf16_le": hex(&source.scale),
                "shard_sha256": shard_sha256,
            },
            "note": "every other object is the source container's, byte for byte; the measurements above it (kld, mmlu, acceptance, flags) were taken on the INT4 table",
        }),
    );
    let new_sidecar = sidecar_path(out);
    std::fs::write(&new_sidecar, serde_json::to_vec_pretty(&Value::Object(sidecar)).expect("json"))
        .map_err(io(new_sidecar.display()))?;
    Ok(RetableReport { path: out.to_owned(), bytes, objects, table_rows: rows, shard_sha256 })
}

/// SHA-256 of a whole file, streamed.
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(io(path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let got = file.read(&mut buffer).map_err(io(path.display()))?;
        if got == 0 {
            break;
        }
        hash.update(&buffer[..got]);
    }
    Ok(hex(&hash.finalize()))
}

/// Copy `companion` to `out` and write `out`'s sidecar pinned to `main`
/// (a container [`retable`] wrote). The copy's objects are unchanged.
pub fn repin_companion(companion: &Path, main: &Path, out: &Path) -> Result<()> {
    let old_sidecar = sidecar_path(companion);
    let mut sidecar: Value = serde_json::from_slice(&std::fs::read(&old_sidecar).map_err(io(old_sidecar.display()))?)
        .map_err(io(old_sidecar.display()))?;
    let reader = Reader::open(main)?;
    let previous = sidecar.pointer("/pair/main/content_hash").cloned().unwrap_or(Value::Null);
    let pair = sidecar
        .pointer_mut("/pair/main")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| fail(format!("{} records no pair.main", old_sidecar.display())))?;
    pair.insert("file".into(), json!(main.file_name().and_then(|n| n.to_str())));
    pair.insert("model_id".into(), json!(reader.identity().model_id));
    pair.insert("weights_id".into(), json!(reader.identity().weights_id));
    pair.insert("content_hash".into(), json!(hex(&reader.content_hash())));
    pair.insert("bytes".into(), json!(reader.file_bytes()));
    pair.insert("objects".into(), json!(reader.objects().len()));
    pair.insert("file_sha256".into(), json!(file_sha256(main)?));
    sidecar["repinned"] = json!({
        "spec": "flash-next/08",
        "from_content_hash": previous,
        "note": "the head was calibrated on the INT4 n-gram table; its acceptance on the FP8 table is measured, not assumed",
    });
    mtp::check_pair(&sidecar, &reader)?;
    std::fs::copy(companion, out).map_err(io(out.display()))?;
    let new_sidecar = sidecar_path(out);
    std::fs::write(&new_sidecar, serde_json::to_vec_pretty(&sidecar).expect("json")).map_err(io(new_sidecar.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_next::{bind, fixture};

    /// A safetensors file holding the given (name, dtype, shape, bytes) tensors.
    fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
        let mut header = Map::new();
        let mut at = 0u64;
        for (name, dtype, shape, bytes) in tensors {
            header.insert((*name).into(), json!({"dtype": dtype, "shape": shape, "data_offsets": [at, at + bytes.len() as u64]}));
            at += bytes.len() as u64;
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        for (_, _, _, bytes) in tensors {
            out.extend_from_slice(bytes);
        }
        std::fs::write(path, out).unwrap();
    }

    /// The fixture's table as an FP8 release: two shards in two files, the
    /// scale beside shard 0, an index naming them.
    fn fp8_source(dir: &Path, geometry: &FlashNextGeometry) -> Vec<u8> {
        std::fs::create_dir_all(dir).unwrap();
        let prefix = format!("model.language_model.layers.{}.ple.ple_embedding.ngram_embedding", geometry.ple_layer);
        let cols = geometry.ngram_head_dim;
        let half = geometry.ngram_rows / 2;
        let codes: Vec<u8> = (0..geometry.ngram_rows * cols).map(|i| (i * 7 % 251) as u8).collect();
        let split = (half * cols) as usize;
        let (s0, s1) = (format!("{prefix}.shard_0.weight"), format!("{prefix}.shard_1.weight"));
        let scale = format!("{prefix}.weight_scale");
        write_safetensors(
            &dir.join("a.safetensors"),
            &[(&s0, "F8_E4M3", vec![half, cols], codes[..split].to_vec()), (&scale, "BF16", vec![1], vec![0x80, 0x3b])],
        );
        write_safetensors(
            &dir.join("b.safetensors"),
            &[(&s1, "F8_E4M3", vec![geometry.ngram_rows - half, cols], codes[split..].to_vec())],
        );
        let index = json!({"weight_map": {s0: "a.safetensors", scale: "a.safetensors", s1: "b.safetensors"}});
        std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
        codes
    }

    #[test]
    fn retable_swaps_only_the_table_and_repins_the_companion() {
        let main = fixture::build("retable-main").unwrap();
        main.tree.write_mtp(&main.path).unwrap();
        crate::packer::pack(&main.tree.mtp_pack_options(main.path.clone()), &mut |_| {}).unwrap();
        let companion = main.tree.mtp_artifact_path();
        let geometry = FlashNextGeometry::fixture();
        let dir = main.path.parent().unwrap().join("fp8-source");
        let codes = fp8_source(&dir, &geometry);

        let source = read_fp8_source(&dir, &geometry).unwrap();
        assert_eq!((source.rows(), source.scale), (geometry.ngram_rows, [0x80, 0x3b]));
        let out = main.path.with_file_name("retabled.ninfer");
        let report = retable(&main.path, &source, &out, &geometry, &mut |_| {}).unwrap();

        let old = Reader::open(&main.path).unwrap();
        let new = Reader::open(&out).unwrap();
        assert_eq!(new.identity().weights_id, format!("{}{WEIGHTS_ID_SUFFIX}", old.identity().weights_id));
        assert_eq!((report.bytes, report.objects), (new.file_bytes(), old.objects().len()));
        let table = ngram_table_name(&geometry);
        for object in old.objects() {
            if object.name() == table {
                continue;
            }
            assert_eq!(old.payload_at(object).unwrap().data, new.payload(object.name()).unwrap().data, "{}", object.name());
        }
        let rows = new.payload(&table).unwrap().data;
        assert_eq!(rows.len() as u64, geometry.ngram_rows * 162);
        let cols = geometry.ngram_head_dim as usize;
        for r in [0usize, 1, geometry.ngram_rows as usize / 2, geometry.ngram_rows as usize - 1] {
            assert_eq!(&rows[r * 162..r * 162 + cols], &codes[r * cols..(r + 1) * cols], "row {r}");
            assert_eq!(&rows[r * 162 + cols..(r + 1) * 162], &[0x80, 0x3b], "row {r}'s scale");
        }
        assert_eq!(bind(&new, &geometry).unwrap().ngram_format, NumericFormat::Fp8E4M3FnRowBf16S);

        let sidecar: Value = serde_json::from_slice(&std::fs::read(sidecar_path(&out)).unwrap()).unwrap();
        assert_eq!(sidecar["artifact"]["bytes"], json!(new.file_bytes()));
        assert_eq!(sidecar["retable"]["from"]["content_hash"], json!(hex(&old.content_hash())));

        let repinned = out.with_file_name("retabled-mtp.ninfer");
        let companion_sidecar: Value = serde_json::from_slice(&std::fs::read(sidecar_path(&companion)).unwrap()).unwrap();
        assert!(mtp::check_pair(&companion_sidecar, &new).is_err(), "the old pin does not fit the new container");
        repin_companion(&companion, &out, &repinned).unwrap();
        let pinned: Value = serde_json::from_slice(&std::fs::read(sidecar_path(&repinned)).unwrap()).unwrap();
        mtp::check_pair(&pinned, &new).unwrap();
        assert!(mtp::check_pair(&pinned, &old).is_err());
        assert_eq!(std::fs::read(&repinned).unwrap(), std::fs::read(&companion).unwrap());
    }

    #[test]
    fn an_fp8_table_is_not_retabled_again() {
        let mut tree = fixture::WorkTree::new("retable-twice").unwrap();
        tree.ngram_format = NumericFormat::Fp8E4M3FnRowBf16S;
        tree.write_all().unwrap();
        let main = fixture::build_from(tree).unwrap();
        let geometry = FlashNextGeometry::fixture();
        let dir = main.path.parent().unwrap().join("fp8-source");
        fp8_source(&dir, &geometry);
        let source = read_fp8_source(&dir, &geometry).unwrap();
        let err = retable(&main.path, &source, &main.path.with_file_name("x.ninfer"), &geometry, &mut |_| {})
            .unwrap_err()
            .to_string();
        assert!(err.contains("only an INT4 table is retabled"), "{err}");
    }
}
