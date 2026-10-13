# Image converter

Spec 01 (`docs/specs/image-model/01-the-image-artifact.md`, GitHub #318): the offline tool that
turns `Qwen/Qwen-Image-2.1-Turbo` at revision `fc13ba1986fcb905854dc8de2f52eee395f49b5b` into the
ignis image artifact, a `.ninfer` v2 container of family `Qwen-Image-2.1` (`model_id`
`ignis-image-2.1-turbo`). CPU only. Python writes **work files**; the Rust packer
(`ignis-artifact-pack --family qwen-image`, `crates/artifact`) assembles the container from them,
as for Flash-Next (`docs/specs/flash-next/layout.md` §2 and §6 are the byte contract). The binder
is `crates/artifact/src/qwen_image.rs`.

## Files

| File | What it is |
|---|---|
| `convert.py` | The command: `run` (check, convert, pack) and `fixture` (a synthetic checkpoint and its work tree, for `crates/artifact/tests/qwen_image.rs`). |
| `pin.py`, `pin.json` | The accepted checkpoint: the revision, every file the converter reads with its size and sha256 (a safetensors file's is its LFS oid), every tensor with its dtype and shape. `check` refuses any difference, naming it. |
| `recipe.py` | The tensor map: spec 01's encoding table, each checkpoint tensor's object name, encoding and stored shape. A tensor it does not name is refused. |
| `fp8.py` | FP8 E4M3FN with a bf16 scale per row (`row-scale-v1`): a copy of `tools/flash-next-converter/fp8.py`, plus the round-trip error. |
| `rowsplit.py` | Q4/Q5/Q6 G64 and W8 G32 (`row-split-k128-v1`), the 27B vision tower's formats: adapted from the reference converter (gpillon/ninfer `a00648c`, Apache-2.0), bit-identical to it. |
| `work.py` | The DONE protocol, copied from `tools/flash-next-converter/layout.py`. |
| `metadata.py` | `frontend/image_model.json` (sampler sigmas, scheduler, image processor, latent normalization, the prompt templates verbatim from diffusers 0.41.0) and the VAE's convolution shapes. |
| `synthetic.py` | The synthetic checkpoint: 2 DiT blocks, 2 text layers, a 3-block vision tower, a small VAE, from diffusers' and transformers' own modules. |
| `fp8_bound.py` | The measurement behind `--fp8-bound`'s default (below). |
| `requirements.txt` | The venv `F:/ai/venvs/ignis-image` (torch cu130, diffusers 0.41.0, transformers 5.17.0); the recorder uses it too. |

## What is stored

| part | object names | encoding |
|---|---|---|
| DiT block linears, `img_in`, `txt_in`'s two linears, `proj_out` | `transformer/<checkpoint name>` | FP8 E4M3FN, a bf16 scale per output row |
| text encoder decoder-layer linears (36 layers) | `text_encoder/layers.{l}...` | FP8, as above |
| DiT modulation, timestep embedder, `norm_out.linear` (`norm_out`'s modulation), every norm | as above | bf16, the checkpoint's bits |
| token embedding | `text_encoder/embed_tokens.weight` | bf16; the binder keeps it on the host |
| vision tower, merger, 3 DeepStack mergers | the 27B's (`vision/layers/{b}/attention/qkv`, ...; `vision/deepstack/{i}/...`) | the 27B's: Q6 patch embedding (as `[1152, 1536]`), Q4 qkv/fc1, Q5 output/fc2, W8 mergers, bf16 biases, norms, positions |
| VAE, `quant_conv`, `post_quant_conv` | `vae/<checkpoint name>` | bf16, the checkpoint's shapes |
| frontend | `frontend/<file>` | the tokenizer, chat template, the checkpoint's configs, `image_model.json` |

Not stored: `lm_head` and `model.language_model.norm` (the encoder's output is taken before it).

## The record

`work/converter.json`, merged by the packer into `<artifact>.conversion.json`: every stored
tensor (source name, object, encoding, format, shape, bytes, sha256, max |value|, round-trip
relative error for FP8 and the grouped formats), the tensors not stored, the VAE's convolutions
(kernel, stride, padding as `[left, right, top, bottom]`, channels, the 2x upsampling before the
decoder's upsampler convolutions), the source revision and file hashes, this tool's commit and the
library versions.

## The FP8 bound

The conversion refuses an FP8 tensor whose round-trip relative error,
`||W - decode(encode(W))||_F / ||W||_F` with the decode in fp32 (`e4m3fn(code) * bf16(scale)`), is
above `--fp8-bound`. The default, **0.03**, comes from the 27B's FP8 linears: `fp8_bound.py`
fetched ten of them (`unsloth/Qwen3.8-27B-NVFP4@7d6f8d4`, the source of the 27B artifact's FP8
words, against the BF16 `Qwen/Qwen3.8-27B@1d4bf0f`: q/k/v/o of layer 3, GDN in/out of layer 0, MLP
gate/up/down of layer 60) and measured 2.659-2.663% (max 2.6626e-02); this encoder gives
2.643-2.648% on the same weights. 3% is that maximum rounded up.

## Tests

```
F:/ai/venvs/ignis-image/Scripts/python.exe -m pytest tools/image-converter -q
```

CPU only; two tests run only on this machine (the reference converter in `F:/ai/q38/ninfer` and
the 27B artifact). The Rust side: `cargo test -p ignis-artifact --test qwen_image`.

## The conversion

```
F:/ai/venvs/ignis-image/Scripts/python.exe tools/image-converter/convert.py run \
    --checkpoint F:/ai/models/hf/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba1986fcb905854dc8de2f52eee395f49b5b \
    --out F:/ai/models/ignis-image-2.1-turbo
```

Download the checkpoint first with `HF_HUB_CACHE=F:/ai/models/hf` (32.5 GB; the default cache is
on C:), and build the packer (`cargo build --release -p ignis-artifact --bin ignis-artifact-pack`).
The work tree (~17 GB) is deleted unit by unit as the packer appends it: the disk peak is the
checkpoint, the work tree and the unit being appended (~58 GB with the checkpoint). The output is `ignis_image_2_1_turbo-v2.ninfer` and its
`.conversion.json` in `--out`.
