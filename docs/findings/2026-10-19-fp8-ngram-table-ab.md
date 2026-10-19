# Qwen's FP8 n-gram table moves 4.5% of top-1 tokens, gains ~0.5% prose perplexity, and costs MTP acceptance

- Kind: experiment
- Status: current
- Observed: 2026-10-19
- Last verified: 2026-10-19
- Scope: artifact / the Flash-Next n-gram table's format (`Q4G32_F16S` against `FP8_E4M3FN_ROW_BF16S`), serving on the owner's RTX 5090 box (Linux, 64 GB RAM)
- Related: spec flash-next/08, layout.md §7 and §13.2, r/LocalLLaMA "Qwen3.8 Flash Next unsloth quants KLD/top1" (u/MarkoMarjamaa) and its "full n-gram table" follow-up
- Superseded by: none

## Question

The converter stores the n-gram table at 4 bits (15 levels, an fp16 scale per 32 values)
and never measures that choice on its own. Community runs of the same checkpoint in
llama.cpp-family engines report that swapping the GGUF's ~4-bit table for Qwen's own FP8
table visibly improves output at no speed cost. Does it in ignis, and at what cost?

## Evidence

Branch `ngram-fp8-table` (spec flash-next/08). `ignis-artifact-retable` built the FP8
container from the served one (`gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis` at
`8d94db83`) and `Qwen/Qwen3.8-Flash-Next-FP8` shards 5-37: every object but the table
byte-identical, 94,800,826,368 bytes (+72 bytes a row, +one header page), 3 min 40 s. The
MTP companion was re-pinned, unchanged. GPU work ran with the owner's server stopped.

**Distribution and perplexity** (`flash_next_table_ab`, teacher-forced, 2,048-token windows,
8 per file, 16,376 scored positions each; corpora outside the calibration: wikitext-103
*test*, and JavaScript from an unrelated project):

| corpus | ppl INT4 | ppl FP8 | mean ΔNLL (FP8 − INT4), 95% block-bootstrap CI | top-1 agree | KL(INT4 ‖ FP8), top-64 |
|---|---|---|---|---|---|
| code | 2.3194 | 2.3202 | +0.0003 nats [−0.0031, +0.0034] | 96.11% | 0.018 |
| prose | 2.2509 | 2.2397 | −0.0050 nats [−0.0114, +0.0007] | 94.93% | 0.038 |
| all | 2.2849 | 2.2796 | −0.0023 nats | 95.52% | 0.028 |

**Serving** (same flags as production but `--ngram-hot-bytes 3G`; greedy; one 1,024-token
generation ×3, a 29,192-token prefill ×2):

| | INT4 table | FP8 table |
|---|---|---|
| decode | 206-212 tok/s | 194-200 tok/s |
| MTP acceptance | 627 / 790 = 79.4% | 597 / 851 = 70.2% |
| prefill | 4,605-4,615 tok/s | 3,908 (first, cold hot rows), 4,588 tok/s |
| hot rows in RAM | 1.47 GB | 2.60 GB (whole hot list) |

The G1 canary suite passes on the FP8 container; the kernel test's FP8 cases match the
host model's FP8 decode within the device tolerance.

## Conclusion

The table's format is not a rounding detail: switching it flips the top-1 token at ~1 in
22 positions, a KL the size of the whole conversion's FP8-only non-expert cost
(0.026-0.06). Measured quality moves the community's way on prose (−0.5% perplexity, just
short of significance at 16k positions) and not at all on code. The serving cost is MTP
acceptance (79.4% → 70.2%, decode −6%): the head was calibrated on the INT4 table, so the
re-pinned pair drafts against a trunk that has moved. Prefill and NVMe reads are unchanged
within noise.

The FP8 table is therefore a small quality gain bought with decode speed, until the MTP
head is recalibrated on the FP8 trunk (spec flash-next/07 phase B,
`flash_next_mtp_calibration`), which should recover most of the acceptance. Without BF16
references (not published with the artifact) this cannot say how much of the conversion's
MMLU gap (0.701 against 0.737) the table explains.

## Reuse

- A table-format A/B needs both containers, the corpora above and ~2 minutes of GPU per
  container: `flash_next_table_ab score <dir> <corpus> <out>` then `compare`.
- A re-pinned companion is a measurement, not a free pass: compare `spec.accepted /
  spec.drafted` from `ignis.request.done` on the same prompt before switching.
- One prompt's acceptance is deterministic but narrow; a production decision should repeat
  it on a few prompts and recalibrate the head first.
