# 08 — The FP8 n-gram table

The n-gram table (layout.md §7) is stored INT4: 15 levels per value, one fp16 scale per 32.
It is the one Flash-Next object the converter quantizes without measuring its cost on its
own, and community measurements on the same checkpoint (r/LocalLLaMA, "Qwen3.8 Flash Next
unsloth quants KLD/top1", u/MarkoMarjamaa; the "full n-gram table" follow-up) found a 4-bit
table costs measurably, and that Qwen's own FP8 table restores it at no decode-speed cost.
Our own sidecar shows a quantized MMLU of 0.701 against 0.737 for BF16 (n = 281, McNemar
p = 0.087) with no split of that gap between the experts and the table.

This spec adds a second table format, **Qwen's FP8 table carried verbatim**, and the tools
to put it in an existing container and measure it.

## The format

- Source: `Qwen/Qwen3.8-Flash-Next-FP8`, shards `model-00005` .. `model-00037`:
  `layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`, each `F8_E4M3`
  `[2,500,012, 160]`, and one `ngram_embedding.weight_scale`, `BF16` `[1]`.
- Container object: the same name and shape as layout.md §7.1, format
  **`FP8_E4M3FN_ROW_BF16S`**, layout **`row-interleaved-v1`**. A row is **162 bytes**:

  | bytes | content |
  |---|---|
  | `[0, 160)` | the source row's 160 FP8 E4M3FN bytes, verbatim |
  | `[160, 162)` | the row's BF16 scale, little-endian: the source's one `weight_scale` |

  Value `i` decodes to `fp8_e4m3fn(byte_i) · scale`, rounded to bf16 as the INT4 path's
  decode is. The table is 320,001,536 × 162 = **51,840,248,832 bytes**.
- A per-row scale costs 2 bytes a row (640 MB) over a per-table one. It buys reuse of the
  existing `FP8_E4M3FN_ROW_BF16S` format name, a row that decodes on its own, and room for a
  future per-row-scaled table; nothing reads the 160 bytes differently.
- Lossless against the FP8 release: no rounding happens on our side.

## Changes

1. **Artifact** (`crates/artifact`): `row-interleaved-v1` accepts `FP8_E4M3FN_ROW_BF16S`
   with `row_bytes = columns + 2`; `Q4G32_F16S` is unchanged; every other format is still
   refused by name.
2. **Flash-Next plan** (`flash_next.rs`): the table entry takes either format; the format
   is read from the container, never assumed. A container with any other table format is
   refused at bind, naming the format.
3. **Host reader** (`ngram_table.rs`, `ngram_cache.rs`): already sized by the layout's
   `row_bytes`; the persisted hot-row cache's identity includes the row bytes, so a 162-byte
   table never reuses a 90-byte cache file.
4. **Device** (`kernel/src/flash_next`): `Geometry` carries the table format;
   `ngram_row_bytes()` follows it; `dequant_rows` gains the FP8 branch. The kernel test
   decodes both formats against a host reference, bit for bit.
5. **Retable tool** (`ignis-retable`, `crates/artifact/src/bin`): copies a packed main
   container, replacing only the table object with one built from the FP8 source shards.
   Every other object's bytes are copied unchanged; offsets after the table move. The new
   container keeps `model_id`, takes `weights_id` + `-ngramfp8`, and gets a fresh sidecar
   recording the source repo, revision and shard hashes. The hot list (`hot_rows`) is
   copied: row ids do not change with the format.
6. **Companion re-pin**: the MTP companion pins the main container's `content_hash`
   (layout.md §13.2), which the retable changes. `ignis-retable --companion` writes a copy of
   the companion whose `pair.main` names the new container, and records in it that the head
   was calibrated on the INT4 table. A mismatched pair costs acceptance, never correctness;
   acceptance is measured (below), not assumed.

## Costs

- Disk: +23.0 GB (51.8 GB table against 28.8 GB).
- RAM: the hot list at 162 + index bytes a row is ~2.6 GB, against 1.47 GB. A
  `--ngram-hot-bytes` that held the INT4 hot list (2G) holds a ranked prefix of it; 3G holds
  all of it. The whole table no longer fits the host plan on a 64 GB machine either way.
- NVMe: a missing row is 162 bytes instead of 90, inside the same 4 KiB sector reads.

## Acceptance

- `cargo test` passes workspace-wide; the kernel test covers the FP8 decode.
- The retabled container loads, serves, and answers the G1 canary sequences.
- A/B against the INT4-table container, same everything else, logged in `docs/findings/`:
  MMLU on the conversion's 281 questions; top-1 agreement and KLD between the two streams on
  the same prompts; MTP acceptance with the re-pinned companion; decode and prefill tokens/s.
  BF16 KLD only if the conversion's `references/` are available.
- The production unit switches only if MMLU does not drop and speed holds within noise.
