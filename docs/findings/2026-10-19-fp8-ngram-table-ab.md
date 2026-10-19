# Qwen's FP8 n-gram table moves 4.5% of top-1 tokens, gains ~0.5% prose perplexity, and costs no measurable MTP acceptance

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

**Acceptance across prompts** (the single prompt above is one prompt: re-measured on six
prompts, 512 greedy tokens each, scored on the five that ran to ≥ 500 tokens):

| pair | acceptance | decode (6 prompts) |
|---|---|---|
| INT4 table + its own head (production) | 1,455 / 2,198 = 66.2% | 184.5 tok/s |
| FP8 table + the same head, re-pinned | 1,479 / 2,144 = 69.0% | 189.3 tok/s |
| FP8 table + a head re-calibrated on the FP8 trunk | 1,472 / 2,160 = 68.2% | 188.2 tok/s |

Per prompt the acceptance moves both ways (the B-tree prompt 70.5% INT4 against 68.6% FP8;
others the reverse). The re-calibrated head used a substitute corpus
(`tools/flash-next-mtp/substitute_corpus.py`: 130 chunks, 266k tokens of the allowlisted
sources; the study's token files are not published), took 11 min on the 5090 (rates 2.999
bits, MoE −20.4 dB) and verified bit-identical; it is no better than the original head
re-pinned.

The G1 canary suite passes on the FP8 container; the kernel test's FP8 cases match the
host model's FP8 decode within the device tolerance.

## Conclusion

The table's format is not a rounding detail: switching it flips the top-1 token at ~1 in
22 positions, a KL the size of the whole conversion's FP8-only non-expert cost
(0.026-0.06). Measured quality moves the community's way on prose (−0.5% perplexity, just
short of significance at 16k positions) and not at all on code. Serving costs nothing
measurable: across prompts the re-pinned head accepts as often on the FP8 trunk as on the
INT4 one, decode and prefill are unchanged within noise. The one-prompt 79.4% → 70.2%
first reading was that prompt, not the table. Re-calibrating the head's quantization on
the FP8 trunk buys nothing: acceptance follows the trunk's distribution, which the head's
expert quantization barely touches.

The FP8 table is therefore a small quality gain at no serving cost but 23 GB of disk and
1.1 GB of hot-row RAM. Without BF16 references (not published with the artifact) this
cannot say how much of the conversion's MMLU gap (0.701 against 0.737) the table explains.

## Reuse

- A table-format A/B needs both containers, the corpora above and ~2 minutes of GPU per
  container: `flash_next_table_ab score <dir> <corpus> <out>` then `compare`.
- A re-pinned companion is a measurement, not a free pass: compare `spec.accepted /
  spec.drafted` from `ignis.request.done` on the same prompt before switching.
- One prompt's acceptance is deterministic but narrow: the first reading here (one prompt,
  −9 points) did not survive six. Compare pairs on several prompts, the same ones, scored
  only on requests that ran to length.
