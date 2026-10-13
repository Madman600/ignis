# The image model in ignis — the feature, its tickets, and how to run them

GitHub: the five tickets are created on the owner's OK of these specs.

The feature-level document the image-model tickets decompose
(`docs/agents/issue-tracker.md`). Read this first, then the ticket's spec.

## What and why

ignis gets an **image model**: **Qwen-Image-2.1-Turbo**, text-to-image and
editing with up to 10 condition images, in RGBA. It is a third model family,
reached by a switch and never loaded beside a text model (ADR 0050). Clients:
- the Playground;
- OpenAI-compatible clients, on `/v1/images/generations` and
  `/v1/images/edits`.

To fit one RTX 5090 and run fast:
- **linears:** FP8 weights and FP8 activations;
- **transformer attention:** 8 bits (INT8 QK, FP8 PV), adapted from
  SageAttention2++;
- **condition prefix:** stored in the same 8-bit form, about 1 GiB per 1024²
  condition image;
- **norms, embeddings, modulation, VAE:** bf16.

**Decisions:**
- ADR 0050: ignis serves an image model, one model at a time.
- ADR 0051: kernel code is ours, adapted from a named source, or vendored.
- The owner's decisions of 2026-10-13 (grilling session, Q1-Q34; the log is in
  `F:/ai/opencode/inference/.scratch/qwen-image-grill-2026-10-13/decisions.md`):
  - only the Turbo checkpoint: no 2.1 base, no `steps`, no negative prompt, no
    CFG, PNG only;
  - correctness against diffusers in fp32, speed from the literature, no gate;
  - target 90-100% of the best published implementation per kernel class,
    measured once at the end;
  - FP8 W8A8 linears, 8-bit attention from the first version;
  - the condition budget is computed at load, and condition images beyond it
    are scaled down;
  - prompt enhancement by a text model, optional, off by default.
- The kernel survey:
  [kernels for the image model on the 5090](../../findings/2026-10-13-image-model-kernels-on-the-5090.md).

## The model

| part | shape | ignis |
|---|---|---|
| text encoder | Qwen3-VL-8B: 36 layers, hidden 4096, GQA 32/8 of 128, SwiGLU 12288, vocab 151936, mrope [24,20,20] | FP8 linears; output = last layer **before** the final norm |
| vision tower | 27 layers, hidden 1152, 16 heads, patch 16, merge 2, DeepStack taps at 8/16/24, out 4096 | the 27B's tower code (same geometry), plus DeepStack |
| DiT | 32 single-stream blocks, hidden 4096, 32 heads of 128, SwiGLU ×3, one shared modulation 4096→16384, QK RMSNorm, 3-axis RoPE (16,56,56), no biases; 7.1B | FP8 linears, 8-bit attention |
| VAE | 2D, RGBA in and out, z 64, 16× spatial, base 96 / decoder 144, mults [1,2,4,8,8], one-head mid-block attention | bf16 implicit-GEMM convolution |
| sampler | 8 fixed sigmas (`sample_sigmas`), Euler, no shift, no CFG | Rust drives one Step ABI call per denoising step |

Sequence: text tokens interleaved with the condition images' latents (4
latent tokens per vision slot of the encoder), then the target image. Text is
causal, each image block bidirectional. Text and condition tokens are
modulated at t = 0, so their keys and values are the same at every
denoising step: they are the **condition prefix**, computed in the first
denoising step and reused by the other seven.

## The tickets

| spec | what | blocked by | can start without the others |
|---|---|---|---|
| 01 | converter, artifact, HF publication, diffusers recordings | — | everything |
| 02 | kernels: FP8 GEMM, 8-bit attention, encoder attention, convolution, fused DiT/VAE ops | — | everything; fp64 references are our own |
| 03 | text encoder: Qwen3-VL text + vision tower + DeepStack | 01, 02 | tokenizer, template, preprocessing, CPU tests |
| 04 | the image leaf: DiT, condition prefix, sampler, VAE, Step ABI, end-to-end test | 01, 02, 03 | block structure, segment plan, RoPE positions, sampler |
| 05 | serving: model kind, routes, queue, condition budget, PNG, Playground, prompt enhancement | 04 for the real leaf | all of it against a fake leaf |

Spec 05 can split in two tickets (server; Playground and prompt enhancement)
only if two sessions run them in parallel.

**Order.** 01 and 02 start together. 05 starts with them, against a fake
`ModelLoader`. 03 and 04 follow 02. In 03, 04 and 05 the acceptance puts
text-to-image before editing, so text-to-image works end to end first.

## Environment

- **Checkpoint:** `Qwen/Qwen-Image-2.1-Turbo` revision
  `fc13ba1986fcb905854dc8de2f52eee395f49b5b`, 32.5 GB:
  - text encoder 17.5 GB;
  - transformer 14.2 GB;
  - VAE 0.7 GB.
- **Python:** a venv pinned by `tools/image-converter/requirements.txt`, with:
  - torch cu130;
  - diffusers at a commit that has `QwenImage21Pipeline`;
  - transformers with Qwen3-VL.
  Never set `TORCH_CUDA_ARCH_LIST` in a venv shared with exllamav3.
- **Artifact output:** `F:/ai/models/ignis-image-2.1-turbo/`, ~17 GB.
- **Recordings:** `F:/ai/models/ignis-image-2.1-turbo/oracle/`, not in the
  repo.
- **Worktrees:** `../.inference-qwen-worktrees/<branch>/`, one per ticket.

## The GPU: one card, one run at a time

- `make gpu-status` before any GPU work.
- Take the shared lock (`../.inference-qwen-worktrees/.swarm/gpu-lock.sh`).
- A second job on the card kills the first with no diagnostic.
- ninfer may be stopped for a GPU run; the owner restarts it.
- GPU jobs in this feature:
  - the fp32 and bf16 recordings (01): minutes;
  - the GEMM microbenchmark (02): minutes;
  - the per-op GPU tests (02-04);
  - the end-to-end test and the one timing run (04).
- Keep total VRAM ≤ ~28 GB, desktop included.

## Rules

- **Tests:** every change ships with a test. `cargo test` must pass
  workspace-wide; GPU tests are `--ignored` and run under the GPU profile.
- **Formatting:** never run `cargo fmt`; diffs are semantic only.
- **Merging and publishing:** merge a ticket's branch into local `main` only
  when every acceptance criterion holds. Never push.
- **Closing:** close the ticket with a comment listing the evidence for each
  criterion. Otherwise leave it open with a progress comment saying what holds,
  what does not and why.
- **Copied code** (ADR 0051):
  - only MIT, BSD or Apache-2.0;
  - the upstream's repo, commit, path and licence go in the file header and in
    `kernel/NOTICE`;
  - never copy FlashInfer's `csrc/fmha_v2` (TensorRT licence).
- **CUTLASS 3.x:** not used. Its Windows build is unsupported, and its sm120
  TMA kernels crash on Windows 11 with the 5090 (NVIDIA/cutlass #2905, #2906).
  CUTLASS 2.x sm80 code is fine to read or adapt.
- **Weights licence:** Qwen Research License, non-commercial. Anything we
  publish says "Built with Qwen", and its primary name never contains "Qwen".
- **Thresholds are config fields** with a default, never constants in code.
