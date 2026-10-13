"""The diffusers recordings (spec 01, AC 5, 6 and 8): the oracle specs 03 and 04 are checked
against, and the two yardsticks their tolerance comes from (bf16, and simulated W8A8 on the
artifact's own values).

The oracle is diffusers' `QwenImage21Pipeline` computing in fp32. The checkpoint is bf16, so fp32
is a compute precision, not a weight precision: the weights stay bf16 at rest (exactly the
checkpoint's values) and are upcast where they are used, which gives the arithmetic of
`from_pretrained(torch_dtype=torch.float32)` bit for bit (test_record.py) in about half the memory.
- The token embedding, the 36 text decoder layers, `lm_head` and the 32 DiT blocks hold bf16 at
  rest and are upcast to fp32 inside their own forward only (`set_precision`). The hooks sit on
  whole layers, not on leaves, because the models read a weight's dtype ahead of using it (the
  vision patch embedding, `get_image_features`, diffusers' RMSNorm).
- Everything else (the vision tower and its mergers, the DiT's input projections, modulation and
  output, the VAE) is fp32 for the whole run.
- Each pipeline call is split at `encode_prompt`, the only thing the rest of `__call__` reads from
  the text encoder: an encoder phase runs every case's prompt (fp32 then bf16) with the encoder on
  the GPU and the DiT in host RAM; the encoder is then dropped and a DiT phase replays those
  outputs with the DiT and the VAE on the GPU, up to the final latents. Last, with the DiT dropped,
  a decode phase runs the pipeline's own decode tail (`decode_image`): on the GPU at 1024^2, on
  the host at 2048^2, where the fp32 decode does not fit the GPU budget (33.5 GiB peak working set
  on the host, 316 s).
- The VAE's causal feature cache keeps nothing for a single frame (`_NoRetain`).

    python record.py run --checkpoint SNAPSHOT_DIR [--out ORACLE_DIR]
    python record.py yardstick --checkpoint SNAPSHOT_DIR [--oracle ORACLE_DIR] [--artifact NINFER]
                               [--sim-factor 2] [--bf16-factor 5] [--replace-thresholds [--supersede-bound]]

`run` writes one directory per case (`FIELDS`), a bf16-only thresholds.json and, last,
manifest.json; it prints RECORD DONE or RECORD FAILED at the end. `yardstick` (`run_yardstick`)
runs both yardsticks against that recording, without recording it again, and writes the bound
ignis is held to as thresholds.json (`write_bounds`), with a report; YARDSTICK DONE or FAILED.
The default paths are the owner's layout (`ORACLE`, `ARTIFACT`).

The pipeline's decode tail (`decode_image`) and the hooks on its call (`_Capture`) are adapted
from diffusers 0.41.0 pipeline_qwenimage21.py (Apache-2.0, (c) The HuggingFace Team).
"""
import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

import fp8
import recipe
import rowsplit

HERE = os.path.dirname(os.path.abspath(__file__))
# The owner's layout: the recordings beside the artifact
ORACLE = "F:/ai/models/ignis-image-2.1-turbo/oracle"
ARTIFACT = "F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer"
# The defaults `run`, `run_yardstick` and their flags share
BLOCKS, BLOCK_STEPS = (0, 15, 31), (0, 1)
FACTOR = 5.0                      # run's bf16-only thresholds: factor x the bf16 error
SIM_FACTOR, BF16_FACTOR = 2.0, 5.0
MIN_FREE_DISK_GB, DISK_BUDGET_GB = 12.0, 10.0
RUN_MIN_FREE_RAM_GB = 40.0        # the 2048^2 fp32 host decode peaks at ~36 GB
YARDSTICK_MIN_FREE_RAM_GB = 24.0
MIN_FREE_VRAM_GB, MAX_VRAM_GIB = 27.0, 24.0
HOST_DECODE_FROM = 2048
TOP_ROWS = 8


@dataclass(frozen=True)
class Case:
    """One recorded case: a prompt at size x size, its noise seed, and the condition image's path
    (None for text-to-image)."""
    name: str
    prompt: str
    size: int
    seed: int
    image: str | None = None


PROMPT = ('A red fox sitting in fresh snow at dawn, holding a wooden sign that reads "ignis", '
          'detailed oil painting')
EDIT_PROMPT = "Turn the shapes into polished glass objects and keep the transparent background"
# spec 01's three cases: t2i at 1024^2, the same prompt at the checkpoint presets' 2048^2, and an
# edit with one RGBA condition image at 1024^2
CASES = {
    "t2i": Case("t2i", PROMPT, 1024, seed=1),
    "t2i-2k": Case("t2i-2k", PROMPT, 2048, seed=2),
    "edit": Case("edit", EDIT_PROMPT, 1024, seed=3, image=os.path.join(HERE, "fixtures", "edit-condition.png")),
}


def noise(case, channels):
    """The case's initial noise, (1, channels, size/16, size/16) float32 standard normal from a
    fixed numpy seed: the layout spec 04 draws. The pipeline reads it packed (`_pack_latents`):
    packed[0, y * w + x, c] == noise[0, c, y, x]."""
    side = case.size // 16
    rng = np.random.default_rng(case.seed)
    return rng.standard_normal((1, channels, side, side), dtype=np.float32)


def edit_image(size):
    """The edit case's condition image, procedural so it carries no licence question: shapes on a
    horizontal alpha ramp, with real transparency (alpha 0, partial and 255). The transparent
    pixels are not white, so compositing on white (the encoder's copy) differs from the RGBA the
    VAE reads."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[..., :3] = (40, 90, 160)                                   # under alpha 0: never shown on white
    rgba[..., 3] = np.where(y > 0.75, np.round(255 * x), 0)        # a ramp along the bottom band
    disc = (x - 0.32) ** 2 + (y - 0.36) ** 2 < 0.2 ** 2
    rgba[disc] = (220, 40, 30, 255)
    square = (x > 0.5) & (x < 0.86) & (y > 0.14) & (y < 0.5)
    rgba[square] = (30, 170, 60, 128)
    triangle = (y > 0.45) & (y < 0.72) & (np.abs(x - 0.66) < (y - 0.45))
    rgba[triangle] = (250, 200, 20, 255)
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def condition_image(case):
    """The case's condition image as RGBA, or None for text-to-image."""
    if case.image is None:
        return None
    with Image.open(case.image) as img:
        return img.convert("RGBA")


class _Fp32Forward:
    """Forward hooks for layers whose parameters are bf16 at rest: fp32 inside their forward. The
    bf16 tensors are kept aside and put back after, so the way back allocates nothing."""

    def __init__(self):
        self.on = False
        self.stash = {}

    def pre(self, module, args):
        if self.on:
            params = list(module.parameters())
            self.stash[module] = [p.data for p in params]
            for p in params:
                p.data = p.data.to(torch.float32)

    def post(self, module, args, output):
        if self.on:
            for p, bf16 in zip(module.parameters(), self.stash.pop(module)):
                p.data = bf16


def _upcast_layers(pipe):
    """The layers that hold bf16 at rest: the bulk of the two big models (the token embedding and
    `lm_head` are 2.5 GB each in fp32)."""
    return [pipe.text_encoder.model.language_model.embed_tokens, *pipe.text_encoder.model.language_model.layers,
            pipe.text_encoder.lm_head, *pipe.transformer.transformer_blocks]


def set_precision(pipe, mode):
    """`fp32`: diffusers' fp32 arithmetic with bf16 storage for the upcast layers; `bf16`: a plain
    bf16 pipeline. Buffers are never touched: both diffusers loads keep them fp32."""
    if mode not in ("fp32", "bf16"):
        raise ValueError(f"precision {mode!r}: fp32 or bf16")
    hooks = getattr(pipe, "_ignis_fp32_forward", None)
    layers = _upcast_layers(pipe)
    if hooks is None:
        hooks = _Fp32Forward()
        for layer in layers:
            layer.register_forward_pre_hook(hooks.pre)
            layer.register_forward_hook(hooks.post)
        pipe._ignis_fp32_forward = hooks
    in_layers = {id(p) for layer in layers for p in layer.parameters()}
    rest = torch.float32 if mode == "fp32" else torch.bfloat16
    with torch.no_grad():
        for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
            for p in component.parameters():
                p.data = p.data.to(torch.bfloat16 if id(p) in in_layers else rest)
    hooks.on = mode == "fp32"


class _NoRetain(list):
    """The VAE's causal feature cache, for one frame: tensor stores are dropped. For a single frame
    every read sees None (fresh from `clear_cache`) and the image convolution refuses a cache, so
    dropping them changes no value; keeping them holds a copy of every convolution's input until
    the decode ends (tens of GB at 2048^2 in fp32)."""

    def __setitem__(self, index, value):
        super().__setitem__(index, None if isinstance(value, torch.Tensor) else value)


def _no_retain(vae):
    clear = vae.clear_cache

    def clear_cache():
        clear()
        vae._feat_map = _NoRetain(vae._feat_map)
        vae._enc_feat_map = _NoRetain(vae._enc_feat_map)
    vae.clear_cache = clear_cache
    clear_cache()


def load_pipeline(checkpoint, device):
    """diffusers' pipeline with bf16 weights, for the encoder phase: the text encoder on `device`,
    the DiT and the VAE in host RAM until `denoise_phase`. Host RAM never holds both big models."""
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLForConditionalGeneration

    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(checkpoint, "text_encoder"), dtype=torch.bfloat16)
    text_encoder.to(device)
    pipe = QwenImage21Pipeline.from_pretrained(checkpoint, text_encoder=text_encoder, dtype=torch.bfloat16)
    _no_retain(pipe.vae)
    return pipe


def _drop(module, device):
    """On a GPU, a model no later phase needs: its weights go to `meta` (`pipe.device` then
    resolves to a component still on the GPU)."""
    if torch.device(device).type == "cuda":
        module.to("meta")
        torch.cuda.empty_cache()


def denoise_phase(pipe, device):
    """After every prompt is encoded: the text encoder is dropped and the DiT and the VAE go to
    `device`."""
    _drop(pipe.text_encoder, device)
    pipe.vae.to(device)
    pipe.transformer.to(device)


def decode_phase(pipe, device):
    """After the DiT's last step: the DiT is dropped, so the VAE has the device to itself."""
    _drop(pipe.transformer, device)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(value, indent=2) + "\n")


def _numpy(t):
    t = t.detach().to("cpu")
    if t.dtype == torch.bool:
        return t.to(torch.uint8).numpy()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy()


class _PromptDone(Exception):
    """Ends a pipeline call right after `encode_prompt` (the encoder phase)."""


@dataclass
class Prompt:
    """What the encoder phase hands the DiT: `encode_prompt`'s output, on the host."""
    embeds: torch.Tensor
    pad: torch.Tensor
    files: dict


class _Capture:
    """Hooks on one pipeline call that write the oracle's fields as they appear, as .npy files
    under `out` (a field `a/b` is `out/a/b.npy`), and index them in `files`; with `out` None they
    are kept in `arrays` instead. Every hook is on the instance and comes off when the call ends;
    the pipeline's code is not touched.

    The call is split at `encode_prompt`, the only thing the rest of `__call__` reads from the
    text encoder: `stop_after_prompt` ends it there (the encoder phase), and `prompt` (a Prompt)
    stands in for it without running the encoder (the DiT phase)."""

    def __init__(self, pipe, out, blocks=(), block_steps=(), prompt=None, stop_after_prompt=False):
        self.pipe, self.out = pipe, out
        self.blocks, self.block_steps = tuple(blocks), tuple(block_steps)
        self.prompt, self.stop_after_prompt = prompt, stop_after_prompt
        self.files, self.arrays, self.kv_modes, self.dit_timesteps = {}, {}, [], []
        self.step = -1
        self.values = {}
        self._undo = []

    def save(self, field, tensor):
        array = _numpy(tensor) if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        self.files[field] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if self.out is None:
            self.arrays[field] = array
            return
        path = os.path.join(self.out, field + ".npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.ascontiguousarray(array))

    def _hook(self, handle):
        self._undo.append(handle.remove)

    def _wrap(self, owner, name, wrapper):
        original = owner.__dict__.get(name)
        bound = getattr(owner, name)
        setattr(owner, name, wrapper(bound))
        self._undo.append(lambda: setattr(owner, name, original) if original is not None else delattr(owner, name))

    def __enter__(self):
        import diffusers.models.transformers.transformer_qwenimage21 as dit_module

        pipe = self.pipe
        text_model = pipe.text_encoder.model.language_model

        def encoder_inputs(module, args, kwargs):
            for key in ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"):
                if kwargs.get(key) is not None:
                    self.values[key] = kwargs[key].detach().clone()
                    self.save(key, kwargs[key])
        self._hook(pipe.text_encoder.register_forward_pre_hook(encoder_inputs, with_kwargs=True))

        def mrope(module, args):
            self.save("mrope_position_ids", args[1])
        self._hook(text_model.rotary_emb.register_forward_pre_hook(mrope))

        # the encoder's output is the final norm's input (the pipeline's own hook returns it)
        def before_norm(module, args):
            self.values["encoder_hidden_states"] = args[0].detach().clone()
            self.save("encoder_hidden_states", args[0])
        self._hook(text_model.norm.register_forward_pre_hook(before_norm))

        def encode_prompt(original):
            def wrapped(*args, **kwargs):
                if self.prompt is not None:
                    device = pipe._execution_device
                    return self.prompt.embeds.to(device), None, self.prompt.pad.to(device)
                embeds, mask, pad = original(*args, **kwargs)
                if mask is not None:
                    raise AssertionError("batch 1 has no padding, but the prompt mask is not None")
                self.values["prompt_embeds"] = embeds.detach().clone()
                self.values["image_pad_mask"] = pad.detach().clone()
                self.save("prompt_embeds", embeds)
                self.save("image_pad_mask", pad)
                if self.stop_after_prompt:
                    raise _PromptDone
                return embeds, mask, pad
            return wrapped
        self._wrap(pipe, "encode_prompt", encode_prompt)

        def vae_encode(original):
            def wrapped(image, *args, **kwargs):
                self.values["condition_vae_input"] = image.detach().clone()
                self.save("condition_vae_input", image)
                return original(image, *args, **kwargs)
            return wrapped
        self._wrap(pipe.vae, "encode", vae_encode)

        def prepare_latents(original):
            def wrapped(*args, **kwargs):
                latents, condition = original(*args, **kwargs)
                if condition is not None:
                    self.save("condition_latents", condition)
                return latents, condition
            return wrapped
        self._wrap(pipe, "prepare_latents", prepare_latents)

        def dit_inputs(module, args, kwargs):
            self.step += 1
            self.kv_modes.append(kwargs.get("kv_cache_mode"))
            self.dit_timesteps.append(float(kwargs["timestep"].reshape(-1)[0]))
            if self.step == 0:
                self.values["dit_encoder_hidden_states"] = kwargs["encoder_hidden_states"].detach().clone()
                self.img_shapes = [list(s) for s in kwargs["img_shapes"][0]]
        self._hook(pipe.transformer.register_forward_pre_hook(dit_inputs, with_kwargs=True))

        def rope(module, args, output):
            if self.step == 0:
                self.save("rope", torch.view_as_real(output.contiguous()))
        self._hook(pipe.transformer.pos_embed.register_forward_hook(rope))

        def token_metadata(original):
            def wrapped(image_pad_mask, img_shapes):
                image_ids, target = original(image_pad_mask, img_shapes)
                if self.step == 0:
                    self.save("joint_image_mask", image_pad_mask)
                    self.save("image_ids", image_ids)
                    self.save("target_token_mask", target)
                return image_ids, target
            return wrapped
        self._wrap(pipe.transformer, "build_token_metadata", token_metadata)

        original_segments = dit_module._qwenimage21_prefix_segments

        def segments(image_ids, prefix_len):
            result = original_segments(image_ids, prefix_len)
            if self.step == 0:
                self.segments = [list(s) for s in result]
            return result
        dit_module._qwenimage21_prefix_segments = segments
        self._undo.append(lambda: setattr(dit_module, "_qwenimage21_prefix_segments", original_segments))

        for b in self.blocks:
            def block_output(module, args, output, b=b):
                if self.step in self.block_steps:
                    self.save(f"step{self.step}/block{b:02d}", output)
            self._hook(pipe.transformer.transformer_blocks[b].register_forward_hook(block_output))

        def scheduler_step(original):
            def wrapped(model_output, timestep, sample, *args, **kwargs):
                result = original(model_output, timestep, sample, *args, **kwargs)
                self.save(f"step{self.step}/latents_in", sample)
                self.save(f"step{self.step}/velocity", model_output)
                self.save(f"step{self.step}/latents_out", result[0])
                return result
            return wrapped
        self._wrap(pipe.scheduler, "step", scheduler_step)
        return self

    def __exit__(self, *exc):
        for undo in reversed(self._undo):
            undo()
        self._undo = []
        return False


def _pack(pipe, latents):
    """(1, C, h, w) -> the (1, h * w, C) the pipeline reads."""
    _, channels, h, w = latents.shape
    return pipe._pack_latents(torch.from_numpy(latents), 1, channels, h, w)


def _call(pipe, case, packed, output_type, callback=None):
    return pipe(prompt=case.prompt, image=condition_image(case), height=case.size, width=case.size,
                output_resolution=case.size, latents=packed.to(pipe.device), use_kv_cache=True,
                output_type=output_type, callback_on_step_end=callback).images


def on_white(rgba):
    """The encoder's copy of a condition image: RGBA composited over white (the pipeline's own
    way, `paste` with the alpha as mask)."""
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.getchannel("A"))
    return white


def _check_prompt_traps(pipe, case, cap):
    """Spec 01's encoder traps, asserted on the run itself (test_record.py pins each on a tiny
    pipeline)."""
    import metadata

    ids = cap.values["input_ids"]
    drop = pipe._drop_idx
    prompt = case.prompt or metadata.EMPTY_PROMPT       # trap: an empty prompt becomes " "
    image = None
    if case.image is not None:
        image = on_white(pipe.image_processor.resize(condition_image(case), width=case.size, height=case.size))
    # trap: the prompt is the raw template string, not apply_chat_template
    template = metadata.TEMPLATE_T2I if image is None else metadata.TEMPLATE_TI2I
    raw = pipe.processor(text=[template.format(prompt)], images=None if image is None else [image],
                         return_tensors="pt")
    if not torch.equal(ids.cpu(), raw["input_ids"]):
        raise AssertionError("the encoder's input_ids are not the raw template's")
    # trap: drop_idx is the length of the system message through the chat template
    system = f"<|im_start|>system\n{metadata.SYSTEM_PROMPT}<|im_end|>\n"
    if pipe.processor.tokenizer.decode(ids[0, :drop].tolist()) != system:
        raise AssertionError(f"drop_idx {drop} does not end the system message")
    # trap: the encoder output is the final norm's input, not its output
    if not torch.equal(cap.values["prompt_embeds"], cap.values["encoder_hidden_states"][:, drop:]):
        raise AssertionError("prompt_embeds are not the final norm's input after drop_idx")
    # trap: the encoder sees the image composited on white
    if image is not None:
        got = cap.values["pixel_values"]
        if not torch.equal(got.cpu(), raw["pixel_values"].to(got.dtype)):
            raise AssertionError("pixel_values are not the image composited on white")


def _check_denoise_traps(pipe, case, cap, prompt):
    """The DiT phase's: it read the encoder phase's output, the VAE read the alpha, and the KV cache
    ran as use_kv_cache=True lays it out."""
    if not torch.equal(cap.values["dit_encoder_hidden_states"].cpu(), prompt.embeds):
        raise AssertionError("the DiT did not read the encoder phase's prompt_embeds")
    if case.image is not None:
        # trap: the VAE reads all four channels
        rgba = pipe.image_processor.resize(condition_image(case), width=case.size, height=case.size)
        read = cap.values["condition_vae_input"][0, 3, 0].cpu()
        alpha = torch.from_numpy(np.asarray(rgba)[..., 3].astype(np.float32)) / 127.5 - 1
        if (read.float() - alpha.to(read.dtype).float()).abs().max() > 1e-6:
            raise AssertionError("the VAE did not read the condition image's alpha")
    # use_kv_cache=True: the prefix is extracted at the first step and read at every other
    if cap.kv_modes != ["extract"] + ["cached"] * (len(cap.kv_modes) - 1):
        raise AssertionError(f"kv cache modes {cap.kv_modes}: not use_kv_cache=True's layout")


def _encode(pipe, case, out):
    """The encoder phase of one case: the pipeline up to `encode_prompt`, its fields written under
    `out` (or kept in memory with `out` None). Returns its Prompt."""
    latents = noise(case, pipe.transformer.config.in_channels)
    with _Capture(pipe, out, stop_after_prompt=True) as cap:
        try:
            _call(pipe, case, _pack(pipe, latents), "latent")
        except _PromptDone:
            pass
        else:
            raise AssertionError("the pipeline returned without calling encode_prompt")
    _check_prompt_traps(pipe, case, cap)
    return Prompt(cap.values["prompt_embeds"].cpu(), cap.values["image_pad_mask"].cpu(), cap.files)


def record_prompt(pipe, case, out):
    """The encoder phase's recording: input_ids, the processor's outputs, the mrope positions, the
    encoder's last hidden state and prompt_embeds, under `out` (empty). Returns the Prompt the DiT
    phase replays."""
    os.makedirs(out, exist_ok=True)
    if os.listdir(out):
        raise ValueError(f"{out} is not empty: a recording never mixes two runs")
    return _encode(pipe, case, out)


def record_denoise(pipe, case, out, prompt, blocks=BLOCKS, block_steps=BLOCK_STEPS):
    """The DiT phase's recording, on `prompt` from `record_prompt`: the noise, the condition
    latents, the RoPE table and segments, every step, the block outputs (the pipeline stops at its
    final latents; `record_image` decodes them). Writes `case.json`, the index of both phases'
    fields, and returns it."""
    latents = noise(case, pipe.transformer.config.in_channels)
    with _Capture(pipe, out, blocks, block_steps, prompt=prompt) as cap:
        cap.save("noise", latents)
        _call(pipe, case, _pack(pipe, latents), "latent")
    _check_denoise_traps(pipe, case, cap, prompt)
    cap.save("timesteps", pipe.scheduler.timesteps)
    cap.save("sigmas", pipe.scheduler.sigmas)
    cap.save("dit_timesteps", np.array(cap.dit_timesteps, dtype=np.float32))
    files = {**prompt.files, **cap.files}
    info = {
        "case": {"name": case.name, "prompt": case.prompt, "size": case.size, "seed": case.seed,
                 "image": None if case.image is None else os.path.basename(case.image)},
        "drop_idx": pipe._drop_idx,
        "sequence": {"text_encoder": files["input_ids"]["shape"][1], "dit": files["image_ids"]["shape"][0]},
        "img_shapes": cap.img_shapes,
        "segments": cap.segments,
        "kv_cache_modes": cap.kv_modes,
        "sigmas": [float(s) for s in pipe.scheduler.sigmas],
        "blocks": list(blocks),
        "block_steps": list(block_steps),
        "files": files,
    }
    _write_json(os.path.join(out, "case.json"), info)
    return info


def decode_image(pipe, case, latents, device):
    """The pipeline's own decode of its final (packed) latents, on `device`: `__call__`'s tail after
    the denoising loop (diffusers 0.41.0, pipeline_qwenimage21.py), line for line. A 2048^2 decode
    in fp32 needs more than the GPU budget, so the recorder runs it on the host (`run`)."""
    vae = pipe.vae.to(device)
    latents = torch.as_tensor(latents).to(device)
    latents = pipe._unpack_latents(latents, case.size, case.size, pipe.vae_scale_factor)
    latents = latents.to(vae.dtype)
    shape = (1, vae.config.z_dim, 1, 1, 1)
    latents_mean = torch.tensor(vae.config.latents_mean).view(shape).to(latents.device, latents.dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(shape).to(latents.device, latents.dtype)
    latents = latents * latents_std + latents_mean
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def record_image(pipe, case, out, device):
    """The decode phase's recording: `image.png` from the recorded final latents, decoded on
    `device`."""
    with open(os.path.join(out, "case.json"), encoding="utf-8") as f:
        last = len(json.load(f)["sigmas"]) - 2
    latents = np.load(os.path.join(out, f"step{last}", "latents_out.npy"))
    decode_image(pipe, case, latents, device).save(os.path.join(out, "image.png"))


def record_case(pipe, case, out, blocks=BLOCKS, block_steps=BLOCK_STEPS):
    """Every phase of one case on one pipeline: every field of spec 01's list under `out` (empty),
    .npy files indexed in `case.json`, the decoded `image.png`. Returns the case record."""
    info = record_denoise(pipe, case, out, record_prompt(pipe, case, out), blocks, block_steps)
    record_image(pipe, case, out, pipe.device)
    return info


def rel_l2(x, ref):
    """||x - ref||_2 / ||ref||_2 over the whole tensor, in float64."""
    x, ref = np.asarray(x, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def encode(pipe, case):
    """The encoder phase without a recording: the Prompt, in memory (the yardstick's)."""
    return _encode(pipe, case, None)


def encoder_rows(image_pad_mask):
    """The row sets of prompt_embeds, (1, L - drop_idx, D), the encoder is measured over, as
    boolean masks:
    - `encoder`: the text rows from row 1 on, those the DiT reads as text (image_pad_mask 0). The
      DiT puts the condition latents in place of the image-pad rows, so their values never reach it;
    - `encoder_after_vision`, with a condition image: the text rows after the first image pad, the
      only ones the vision features (DeepStack among them) can reach;
    - `encoder_row0`: the first row after drop_idx (the user turn's `<|im_start|>`), information
      only. Its error is an ill-conditioned template row's, not a sink's (bf16 alone is 54% off on
      it, and it varies with the sequence length), so it bounds nothing."""
    pad = np.asarray(image_pad_mask)[0].astype(bool)
    index = np.arange(pad.size)
    rows = {"encoder": ~pad & (index > 0), "encoder_row0": index == 0}
    if pad.any():
        rows["encoder_after_vision"] = ~pad & (index > np.flatnonzero(pad)[0])
    return rows


def row_rel_l2(x, ref):
    """The encoder metric: per row r, ||x_r - ref_r||_2 / ||ref_r||_2, and their RMS,
    sqrt(mean_r ||x_r - ref_r||^2 / ||ref_r||^2), in float64. Per row because the DiT reads each
    row through an RMSNorm (`txt_in`, transformer_qwenimage21.py:197): a row's scale does not
    weigh in, and one large row cannot stand for the others. Refuses an empty row set and a zero
    reference row (a nan bound would pass silently)."""
    x, ref = np.asarray(x, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    norm = np.linalg.norm(ref, axis=-1)
    if not norm.size or not (norm > 0).all():
        raise ValueError(f"the encoder metric needs rows with a nonzero reference norm ({norm.size} rows)")
    per_row = np.linalg.norm(x - ref, axis=-1) / norm
    return float(np.sqrt(np.mean(per_row ** 2)))


def encoder_errors(embeds, recorded):
    """The encoder quantities of thresholds.json against the recording's prompt_embeds:
    `row_rel_l2` over each row set of `encoder_rows` (the recording's image_pad_mask)."""
    ref = np.load(os.path.join(recorded, "prompt_embeds.npy"))[0]
    rows = encoder_rows(np.load(os.path.join(recorded, "image_pad_mask.npy")))
    return {name: row_rel_l2(np.asarray(embeds)[0][r], ref[r]) for name, r in rows.items()}


def yardstick(pipe, case, recorded, prompt=None):
    """The pipeline's per-step error against a recording, teacher-forced: step i's recorded input
    latents go in (in the pipeline's dtype, so rounded to bf16 in bf16), and its velocity is
    compared with step i's recorded velocity. The DiT reads this pipeline's own encoder output
    (`prompt`) and condition latents, never the recorded ones. Also the encoder's errors
    (`encoder_errors`; `encoder_whole`, `rel_l2` over every row) and, with a condition image, the
    condition latents'. Relative L2 (`rel_l2`). `prompt` is this pipeline's own encoder phase
    (`encode`), run here when None."""
    with open(os.path.join(recorded, "case.json"), encoding="utf-8") as f:
        info = json.load(f)
    steps = len(info["sigmas"]) - 1

    def load(field):
        return np.load(os.path.join(recorded, field + ".npy"))
    inputs = [load(f"step{i}/latents_in") for i in range(steps)]

    def force(pipe_, i, t, tensors):
        if i + 1 == steps:
            return {}
        latents = tensors["latents"]
        return {"latents": torch.from_numpy(inputs[i + 1]).to(latents.device, latents.dtype)}

    prompt = prompt if prompt is not None else encode(pipe, case)
    if not np.array_equal(_numpy(prompt.pad), load("image_pad_mask")):
        raise AssertionError("the encoder's image_pad_mask is not the recording's")
    with _Capture(pipe, None, prompt=prompt) as cap:
        _call(pipe, case, torch.from_numpy(inputs[0]), "latent", callback=force)
    _check_denoise_traps(pipe, case, cap, prompt)
    errors = {"velocity": [rel_l2(cap.arrays[f"step{i}/velocity"], load(f"step{i}/velocity")) for i in range(steps)],
              **encoder_errors(_numpy(prompt.embeds), recorded),
              "encoder_whole": rel_l2(_numpy(prompt.embeds), load("prompt_embeds"))}
    if case.image is not None:
        errors["condition_latents"] = rel_l2(cap.arrays["condition_latents"], load("condition_latents"))
    return errors


BF16_ONLY_SCHEMA = "ignis-image-thresholds-v1"


def write_thresholds(path, errors, factor, session):
    """The recording session's bf16-only thresholds.json (schema v1): per case, the bf16
    teacher-forced error of each step and `factor` times it, the encoder's error (its text rows)
    and `factor` times it. Written once: an existing file is never replaced. `yardstick`
    supersedes it with the bound of both yardsticks (`write_bounds`)."""
    cases = {}
    for name, e in errors.items():
        cases[name] = {
            "velocity": {"bf16_error": e["velocity"], "threshold": [factor * x for x in e["velocity"]]},
            "encoder": {"bf16_error": e["encoder"], "threshold": factor * e["encoder"]},
        }
        if e.get("condition_latents") is not None:
            cases[name]["condition_latents"] = {"bf16_error": e["condition_latents"]}
    value = {
        "schema": BF16_ONLY_SCHEMA,
        "metric": "relative L2, ||x - fp32||_2 / ||fp32||_2 over the whole tensor (the encoder: per row, RMS over "
                  "its text rows), float64",
        "yardstick": "diffusers QwenImage21Pipeline in bf16, teacher-forced: step i's recorded fp32 input "
                     "latents in, its velocity against step i's recorded fp32 velocity",
        "factor": factor,
        "session": session,
        "cases": cases,
    }
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write(json.dumps(value, indent=2) + "\n")
    return value


# The simulated W8A8 yardstick: the pipeline as ignis quantizes it, computing in fp32. The bf16
# yardstick holds no quantization error, so 5x it alone would fail every step of a correct ignis.

E4M3_MAX = 448.0
# #319's floor (kernel/src/image/fp8_quant.cuh, kE4m3MinAmax = 0x1.cp-118): 448 * 2^-126, the
# smallest amax whose scale amax / 448 is a normal fp32
E4M3_MIN_AMAX = 7 * 2.0 ** -120


def fake_quant_e4m3(x):
    """x through #319's per-token E4M3 activations (kernel/include/ignis_image_gemm.h,
    kernel/src/image/fp8_quant.cuh), in fp32: per row of the last dimension, amax = max |x|,
    scale = amax / 448 and inverse = 448 / amax (two correctly rounded fp32 divisions), codes =
    E4M3(x * inverse) with the product rounded once in fp32, then to nearest even and saturated to
    +-448; a row whose amax is below `E4M3_MIN_AMAX` (an all-zero row among them) gets scale 1 and
    zero codes. Returns scale * e4m3(codes), what the GEMM multiplies back. The input is the fp32
    activation: no bf16 rounding before the quantization."""
    amax = x.abs().amax(-1, keepdim=True)
    full = torch.full_like(amax, E4M3_MAX)
    quantized = amax >= E4M3_MIN_AMAX
    # tensor by tensor: a division by a scalar runs as a multiplication by its reciprocal on the GPU
    scale = torch.where(quantized, amax / full, torch.ones_like(amax))
    inverse = torch.where(quantized, full / amax, torch.zeros_like(amax))
    # torch's cast does not saturate (480 and above become NaN): the clamp is the kernel's satfinite
    codes = (x * inverse).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return codes.float() * scale


class _SimQuantLinear(torch.nn.Module):
    """An FP8 linear as ignis runs it (W8A8), simulated in fp32: the weight is the artifact's
    payload, E4M3 codes times their row's bf16 scale (exact in fp32, `fp8.decode_fp32`), dequantized
    in each forward so that only the codes stay resident; the input goes through `fake_quant_e4m3`
    (`quantize_input`). None of the FP8 linears has a bias."""

    def __init__(self, codes, scales, quantize_input=True):
        super().__init__()
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.quantize_input = quantize_input

    def forward(self, x):
        weight = self.codes.float() * self.scales.float()[:, None]
        return torch.nn.functional.linear(fake_quant_e4m3(x) if self.quantize_input else x, weight)


def sim_linear(payload, shape, device):
    """The _SimQuantLinear of an FP8_E4M3FN_ROW_BF16S / row-scale-v1 payload, on `device`."""
    rows, cols = shape
    n = rows * cols
    buf = bytearray(payload)
    codes = torch.frombuffer(buf, dtype=torch.uint8, count=n).view(torch.float8_e4m3fn).view(rows, cols)
    scales = torch.frombuffer(buf, dtype=torch.bfloat16, count=rows, offset=n + (-n % fp8.ALIGN))
    return _SimQuantLinear(codes.to(device), scales.to(device))


class Artifact:
    """The packed `.ninfer` v2 container's objects, read back with the framing of crates/artifact's
    reader (as tools/flash-next-converter/container.py reads it): an 8-byte magic, the directory's
    JSON length as a little-endian u64, the JSON directory, then the payload from the next
    4096-byte boundary; object offsets are payload-relative."""
    MAGIC = b"NINFER\x00\x02"

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            prefix = f.read(16)
            if prefix[:8] != self.MAGIC:
                raise ValueError(f"{path}: not a .ninfer v2 container")
            n = int.from_bytes(prefix[8:16], "little")
            self.directory = json.loads(f.read(n))
        self.payload_start = -(-(16 + n) // 4096) * 4096
        self.objects = {o["name"]: o for o in self.directory["objects"]}

    def payload(self, name):
        o = self.objects[name]
        with open(self.path, "rb") as f:
            f.seek(self.payload_start + o["offset"])
            return f.read(o["bytes"])


def simquant(model, comp, artifact, records):
    """Makes `model` (the pipeline's text encoder, DiT or VAE, `comp` in recipe.py's terms) compute
    with what the artifact stores, from the artifact's own bytes:
    - a linear recipe.py stores as FP8 becomes the _SimQuantLinear of its payload;
    - a grouped-format (vision) weight takes what its payload decodes to, in fp32;
    - every other stored tensor is bf16, the checkpoint's bits, which it must hash to.
    Every payload read and every bf16 tensor must hash to its record in the artifact's conversion
    record (`records`, by source name): the M3 probe's proof, on the packed bytes. One way: the
    replaced linears' bf16 weights are gone. Returns the count per encoding."""
    import hashlib
    counts = {}
    # by name, so that no list holds a replaced linear's bf16 weight until the end (on the GPU,
    # the bf16 and the codes of a whole model would not fit together)
    for name in [name for name, _ in model.named_parameters()]:
        p = model.get_parameter(name)
        item = recipe.plan(comp, name, list(p.shape))
        if item is None:
            continue
        record = records.get(f"{comp}/{name}")
        if record is not None and record["encoding"] != item.encoding:
            raise ValueError(f"{comp}/{name}: the artifact records {record['encoding']}, recipe.py plans {item.encoding}")
        if item.encoding == "bf16":
            data = p.detach().to(torch.bfloat16).cpu().contiguous().view(torch.int16).numpy().tobytes()
        else:
            data = artifact.payload(item.object)
        if record is None or record["sha256"] != hashlib.sha256(data).hexdigest():
            raise ValueError(f"{comp}/{name}: not the artifact's {item.object}")
        if item.encoding == "fp8":
            parent, _, child = name[:-len(".weight")].rpartition(".")
            setattr(model.get_submodule(parent), child, sim_linear(data, item.shape, p.device))
        elif item.encoding != "bf16":
            p.data = rowsplit.decode(data, item.encoding, item.shape).reshape(p.shape).to(p.device)
        counts[item.encoding] = counts.get(item.encoding, 0) + 1
    return counts


@contextlib.contextmanager
def deepstack_off(pipe):
    """The text model without DeepStack: the vision features it adds after its first layers are
    dropped (as if zero)."""
    text_model = pipe.text_encoder.model.language_model
    text_model._deepstack_process = lambda hidden_states, visual_pos_masks, visual_embeds: hidden_states
    try:
        yield
    finally:
        del text_model._deepstack_process


def encoder_breakdown(embeds, recorded, top=TOP_ROWS):
    """Where an encoder error lies, against the recording's prompt_embeds: the share of the squared
    error and of the squared reference norm, and the relative L2 (`rel_l2` pooled over the rows),
    of the text rows and of the image-pad rows; each text row's own relative L2, as [row, error]
    (`row_rel_l2` is the RMS of those from row 1 on, the `encoder` quantity); and the `top` rows by reference norm and by error norm."""
    ref = np.load(os.path.join(recorded, "prompt_embeds.npy"))[0].astype(np.float64)
    pad = np.load(os.path.join(recorded, "image_pad_mask.npy"))[0].astype(bool)
    err = np.linalg.norm(np.asarray(embeds, dtype=np.float64)[0] - ref, axis=-1) ** 2
    norm = np.linalg.norm(ref, axis=-1) ** 2

    def part(rows):
        return {"rows": int(rows.sum()), "error_share": float(err[rows].sum() / err.sum()),
                "norm_share": float(norm[rows].sum() / norm.sum()),
                "rel_l2": float(np.sqrt(err[rows].sum() / norm[rows].sum())) if rows.any() else None}

    def rows(order):
        return [{"row": int(i), "image_pad": bool(pad[i]), "ref_norm": float(np.sqrt(norm[i])),
                 "error_norm": float(np.sqrt(err[i])), "error_share": float(err[i] / err.sum())}
                for i in order[:top]]
    text = np.flatnonzero(~pad)
    bounded = np.flatnonzero(encoder_rows(pad[None])["encoder"])
    return {"whole": float(np.sqrt(err.sum() / norm.sum())), "text_rows": part(~pad), "image_pad_rows": part(pad),
            "row_rel_l2": float(np.sqrt(np.mean(err[bounded] / norm[bounded]))),
            "text_row_rel_l2": [[int(i), float(np.sqrt(err[i] / norm[i]))] for i in text],
            "top_rows_by_ref_norm": rows(np.argsort(-norm)), "top_rows_by_error": rows(np.argsort(-err))}


# What each quantity of thresholds.json measures.
METRICS = {
    "velocity": "step i's predicted velocity, (1, N, 64): relative L2 ||x - fp32||_2 / ||fp32||_2 over the whole "
                "tensor in float64, teacher-forced (step i's recorded fp32 input latents in); each yardstick's DiT "
                "reads that pipeline's own encoder output and condition latents, never the recorded fp32 ones",
    "encoder": "prompt_embeds, (1, L - drop_idx, 4096), the encoder's last hidden state before the final norm, over "
               "its text rows from row 1 on, the rows whose recorded image_pad_mask is 0 (the DiT replaces the "
               "image-pad rows with the condition latents; row 0 is encoder_row0): per row r, ||x_r - fp32_r||_2 / "
               "||fp32_r||_2, and their RMS sqrt(mean_r ||x_r - fp32_r||^2 / ||fp32_r||^2), float64 (txt_in "
               "RMS-normalizes each row)",
    "encoder_after_vision": "with a condition image: the encoder metric over the text rows after the first image "
                            "pad only, the rows the vision features (DeepStack among them) can reach",
    "encoder_row0": "the encoder metric over the first row after drop_idx alone (the user turn's <|im_start|>, an "
                    "ill-conditioned template row: bf16 alone is 54% off on it, varying with the sequence length): "
                    "information, never bounded",
    "condition_latents": "the edit case's condition latents, (1, Nc, 64), packed and normalized: relative L2 over "
                         "the whole tensor",
}
YARDSTICKS = {
    "bf16": "diffusers QwenImage21Pipeline in bf16 (the checkpoint's weights), encoder, DiT and VAE, teacher-forced",
    "sim": "simulated W8A8, teacher-forced: the pipeline computing in fp32 on the artifact's stored values; each "
           "linear the artifact stores as FP8 (FP8_E4M3FN_ROW_BF16S / row-scale-v1, the text encoder's and the "
           "DiT's) uses its payload's E4M3 codes x bf16 row scales and fake-quantizes its fp32 input per token to "
           "E4M3 with scale amax / 448 (#319's kernel: zero rows scale 1, the 7 * 2^-120 amax floor, round to "
           "nearest even, saturate; the kernel's input is bf16, the sim's the fp32 activation); the vision tower "
           "its W8 payloads decoded; every other tensor the artifact's bf16; attention not simulated (fp32). Left "
           "out, for the sim factor to absorb: ignis's bf16 intermediates and its 8-bit attention and KV",
}
# The quantities with a bound; encoder_row0 is information only
BOUNDED = ("encoder", "encoder_after_vision", "condition_latents")


def write_bounds(path, sim, bf16, sim_factor, bf16_factor, oracle_session, session, artifact):
    """thresholds.json, schema v4 (spec 01, AC 6): per case, each step's velocity, the encoder
    (its text rows from row 1; with a condition image, also after the first image pad), the
    condition latents and, unbounded, the encoder's first row: the error of both yardsticks (`sim`, `bf16`: yardstick's
    errors per case) and the bound ignis is held to, max(sim_factor x sim, bf16_factor x bf16); each
    quantity's metric; the factors; the oracle's session and this one's; the artifact measured.
    Refuses an existing file."""
    def bound(s, b):
        return max(sim_factor * s, bf16_factor * b)
    cases = {}
    for name in sim:
        s, b = sim[name], bf16[name]
        cases[name] = {
            "velocity": {"sim_error": s["velocity"], "bf16_error": b["velocity"],
                         "bound": [bound(x, y) for x, y in zip(s["velocity"], b["velocity"])]},
        }
        for quantity in BOUNDED:
            if s.get(quantity) is not None:
                cases[name][quantity] = {"sim_error": s[quantity], "bf16_error": b[quantity],
                                         "bound": bound(s[quantity], b[quantity])}
        cases[name]["encoder_row0"] = {"sim_error": s["encoder_row0"], "bf16_error": b["encoder_row0"]}
    value = {
        "schema": "ignis-image-thresholds-v4",
        "bound": "max(factors.sim x sim_error, factors.bf16 x bf16_error), per quantity: each step's velocity, the "
                 "encoder (text rows from row 1), the encoder after the first image pad, the condition latents "
                 "(encoder_row0 has none)",
        "factors": {"sim": sim_factor, "bf16": bf16_factor},
        "metrics": METRICS,
        "yardsticks": YARDSTICKS,
        "oracle_session": oracle_session,
        "session": session,
        "artifact": artifact,
        "cases": cases,
    }
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write(json.dumps(value, indent=2) + "\n")
    return value


def fp32_ieee():
    """TF32 off for matmul and cuDNN, set through the legacy flags only (torch 2.13 refuses to read
    them back after a mix with the new `fp32_precision` API) and asserted through both APIs: cuDNN
    convolutions default to TF32. Returns the flags for the manifest."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return check_fp32_ieee()


def check_fp32_ieee():
    """Asserts TF32 is off for matmul and cuDNN (conv and RNN); returns the flags. Without a
    visible GPU the cuDNN precisions are not read (None): cuDNN's first read checks the devices'
    capabilities and fails on an empty list."""
    gpu = torch.cuda.device_count() > 0
    flags = {"matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
             "float32_matmul_precision": torch.get_float32_matmul_precision(),
             "matmul.fp32_precision": torch.backends.cuda.matmul.fp32_precision,
             "cudnn.conv.fp32_precision": torch.backends.cudnn.conv.fp32_precision if gpu else None,
             "cudnn.rnn.fp32_precision": torch.backends.cudnn.rnn.fp32_precision if gpu else None}
    if (flags["matmul.allow_tf32"] or flags["cudnn.allow_tf32"] or flags["float32_matmul_precision"] != "highest"
            or "tf32" in (flags["matmul.fp32_precision"], flags["cudnn.conv.fp32_precision"],
                          flags["cudnn.rnn.fp32_precision"])):
        raise AssertionError(f"TF32 is on: {flags}")
    return flags


def efficient_attention_only():
    """SDPA restricted to the memory-efficient kernel: a call it cannot serve raises instead of
    falling back to the math kernel, which at 2048^2 in fp32 would materialise a ~35 GB score
    matrix."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])


def _guard_dit(pipe, on, guard=efficient_attention_only):
    """The DiT's forward under `guard` (on), or its own forward again (off)."""
    if on:
        forward = pipe.transformer.forward

        def guarded(*args, **kwargs):
            with guard():
                return forward(*args, **kwargs)
        pipe.transformer.forward = guarded
    elif "forward" in pipe.transformer.__dict__:
        del pipe.transformer.forward


def _attention_selfcheck(device):
    """The memory-efficient kernel against the math kernel on a small fp32 problem with TF32 off:
    it must not compute in reduced precision. Returns the max abs difference."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    g = torch.Generator(device=device).manual_seed(0)
    q, k, v = (torch.randn(1, 8, 333, 128, generator=g, device=device) for _ in range(3))
    with efficient_attention_only():
        fast = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    with sdpa_kernel([SDPBackend.MATH]):
        exact = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    diff = float((fast - exact).abs().max())
    if diff > 1e-5:
        raise AssertionError(f"memory-efficient fp32 attention is {diff:.3g} off the math kernel")
    return diff


GB = 1e9
# What each case directory holds (spec 01's list), for the manifest and the spec 03/04 readers.
# Arrays are .npy (little-endian, C order); masks are uint8; L is the encoder's sequence, S the
# DiT's joint sequence, N = (size/16)^2 the target's latent tokens, C = 64.
FIELDS = {
    "noise": "(1, C, size/16, size/16) f32, the initial noise; packed[0, y*w+x, c] = noise[0, c, y, x]",
    "input_ids": "(1, L) i64, the raw template tokenized by the processor (image pads expanded)",
    "attention_mask": "(1, L) i64, all ones (batch 1)",
    "mm_token_type_ids": "(1, L) the processor's token types (1 at image pads), read by transformers' mrope",
    "pixel_values": "edit: (patches, 1536) f32, the condition image on white as the processor prepares it",
    "image_grid_thw": "edit: (1, 3) i64",
    "mrope_position_ids": "(3, 1, L) i64, the text model's rotary positions (t, h, w)",
    "encoder_hidden_states": "(1, L, 4096) f32, the last decoder layer's output, before the final norm",
    "prompt_embeds": "(1, L - drop_idx, 4096) f32, encoder_hidden_states without the system message: the DiT's text input",
    "image_pad_mask": "(1, L - drop_idx) u8, 1 at the encoder's image pads",
    "condition_vae_input": "edit: (1, 4, 1, size, size) f32, the RGBA in [-1, 1] the VAE encodes",
    "condition_latents": "edit: (1, Nc, C) f32, packed and normalized ((z - mean) / std)",
    "rope": "(S, 64, 2) f32, the DiT's 3-axis RoPE table as (cos, sin) = (real, imag) of diffusers' complex table: "
            "column j rotates the head's adjacent channel pair (2j, 2j+1); the 64 columns are frame 8 | height 28 | "
            "width 28",
    "joint_image_mask": "(S,) u8, 1 at image tokens of the joint sequence (4 latent tokens per image pad)",
    "image_ids": "(S,) i64, -1 at text, the image block's index elsewhere (conditions first, target last)",
    "target_token_mask": "(S,) u8, 1 at the target image's tokens",
    "timesteps": "(steps,) f32, the scheduler's timesteps (sigma * 1000)",
    "sigmas": "(steps + 1,) f32, the scheduler's sigmas, terminal 0 included",
    "dit_timesteps": "(steps,) f32, the timestep the DiT read: timesteps x fl(1/1000) on CUDA (a division by a "
                     "scalar runs as a multiplication by its reciprocal), 1 ULP off sigmas at some steps; the "
                     "schedule's reference is sigmas and timesteps",
    "step{i}/latents_in": "(1, N, C) f32, step i's input latents (packed)",
    "step{i}/velocity": "(1, N, C) f32, step i's predicted velocity (target rows)",
    "step{i}/latents_out": "(1, N, C) f32, step i's output: latents_in + (sigma[i+1] - sigma[i]) * velocity",
    "step{i}/block{b}": "steps 0 and 1 (the spec's 1 and 2), blocks 0, 15, 31: the block's output; "
                        "step 0 is the prefill, (1, S, 4096), step 1 a cached step, (1, N, 4096)",
    "image.png": "the decoded RGBA image",
    "condition.png": "edit: the condition image, as checked in",
    "case.json": "the case, drop_idx, the sequence lengths, img_shapes, the prefix segments (start, end, is_text), "
                 "the kv cache modes per step, the sigmas, and every .npy's shape and dtype",
}


def _preflight(out, device, min_free_disk_gb, disk_budget_gb, min_free_ram_gb, min_free_vram_gb):
    """Refuses to start unless `out` is new and the disk, host RAM and VRAM have room. Returns
    what it measured."""
    import psutil
    if os.path.exists(out) and os.listdir(out):
        raise FileExistsError(f"{out} is not empty: the oracle is recorded once (delete it to record again)")
    os.makedirs(out, exist_ok=True)
    measured = {"free_disk_gb": shutil.disk_usage(out).free / GB,
                "free_ram_gb": psutil.virtual_memory().available / GB, "free_vram_gb": None}
    if measured["free_disk_gb"] < min_free_disk_gb + disk_budget_gb:
        raise RuntimeError(f"{measured['free_disk_gb']:.1f} GB free on {out}'s disk: the recordings need "
                           f"{disk_budget_gb} GB above the {min_free_disk_gb} GB floor")
    if measured["free_ram_gb"] < min_free_ram_gb:
        raise RuntimeError(f"{measured['free_ram_gb']:.1f} GB of host RAM free, below {min_free_ram_gb} GB")
    if device.type == "cuda":
        measured["free_vram_gb"] = torch.cuda.mem_get_info(device)[0] / GB
        if measured["free_vram_gb"] < min_free_vram_gb:
            raise RuntimeError(f"{measured['free_vram_gb']:.1f} GB of VRAM free, below {min_free_vram_gb} GB")
    return measured


def _versions():
    import importlib.metadata as md
    import convert
    v = convert.versions()
    for lib in ("accelerate", "torchvision", "tokenizers", "pillow", "psutil"):
        v[lib] = md.version(lib)
    v["cuda"] = torch.version.cuda
    v["cudnn"] = torch.backends.cudnn.version() if torch.cuda.device_count() > 0 else None
    return v


def run(checkpoint, out, device, cases, blocks=BLOCKS, block_steps=BLOCK_STEPS, factor=FACTOR,
        min_free_disk_gb=MIN_FREE_DISK_GB, disk_budget_gb=DISK_BUDGET_GB, min_free_ram_gb=RUN_MIN_FREE_RAM_GB,
        min_free_vram_gb=MIN_FREE_VRAM_GB, max_vram_gib=MAX_VRAM_GIB, host_decode_from=HOST_DECODE_FROM, log=print):
    """One session: every case recorded in fp32, then the bf16 yardstick on the same weights, then
    the images decoded (on the host from `host_decode_from` pixels a side; 0: never), then
    thresholds.json and manifest.json (written last: its presence means the run completed)."""
    import convert
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    started = time.time()
    resources = _preflight(out, device, min_free_disk_gb, disk_budget_gb, min_free_ram_gb, min_free_vram_gb)
    tf32 = fp32_ieee()
    attention = {"dit_fp32": "memory-efficient SDPA only (sdpa_kernel), asserted by construction",
                 "selfcheck_max_abs_diff": None}
    if device.type == "cuda":
        if max_vram_gib:
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, max_vram_gib * 2**30 / total), device)
        attention["selfcheck_max_abs_diff"] = _attention_selfcheck(device)
    session = {"id": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **convert.tool_commit()}
    log(f"session {session['id']}: free disk {resources['free_disk_gb']:.1f} GB, RAM {resources['free_ram_gb']:.1f} GB, "
        f"VRAM {resources['free_vram_gb']}")

    t = time.time()
    pipe = load_pipeline(checkpoint, device)
    load_s = time.time() - t
    log(f"loaded in {load_s:.0f} s")

    def peak():
        if device.type != "cuda":
            return None
        value = torch.cuda.max_memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        return round(value, 2)

    def timed(entry, key, work):
        peak()
        t = time.time()
        result = work()
        entry[key + "_wall_s"] = round(time.time() - t, 1)
        entry[key + "_peak_vram_gib"] = peak()
        return result

    entries = {case.name: {"prompt": case.prompt, "size": case.size, "seed": case.seed,
                           "image": None if case.image is None else "condition.png"} for case in cases}
    # the encoder phase: every prompt in fp32 (recorded), then in bf16 (the yardstick's)
    set_precision(pipe, "fp32")
    prompts32, prompts16 = {}, {}
    for case in cases:
        check_fp32_ieee()
        directory = os.path.join(out, case.name)
        prompts32[case.name] = timed(entries[case.name], "fp32_encoder",
                                     lambda: record_prompt(pipe, case, directory))
        if case.image is not None:
            shutil.copyfile(case.image, os.path.join(directory, "condition.png"))
        log(f"fp32 encoder {case.name}: {entries[case.name]['fp32_encoder_wall_s']} s")
    set_precision(pipe, "bf16")
    for case in cases:
        prompts16[case.name] = timed(entries[case.name], "bf16_encoder", lambda: encode(pipe, case))
    # the DiT phase: every case in fp32 (recorded), then the teacher-forced bf16 yardstick
    t = time.time()
    denoise_phase(pipe, device)
    switch_s = time.time() - t
    set_precision(pipe, "fp32")
    _guard_dit(pipe, device.type == "cuda")
    for case in cases:
        check_fp32_ieee()
        timed(entries[case.name], "fp32_dit", lambda: record_denoise(
            pipe, case, os.path.join(out, case.name), prompts32[case.name], blocks, block_steps))
        log(f"fp32 dit {case.name}: {entries[case.name]['fp32_dit_wall_s']} s, "
            f"peak {entries[case.name]['fp32_dit_peak_vram_gib']} GiB")
    _guard_dit(pipe, False)
    set_precision(pipe, "bf16")
    errors = {}
    for case in cases:
        errors[case.name] = timed(entries[case.name], "bf16_dit", lambda: yardstick(
            pipe, case, os.path.join(out, case.name), prompts16[case.name]))
        log(f"bf16 {case.name}: velocity error {[round(e, 5) for e in errors[case.name]['velocity']]}, "
            f"encoder {errors[case.name]['encoder']:.5f}")
    # the decode phase: the VAE alone, in fp32
    decode_phase(pipe, device)
    set_precision(pipe, "fp32")
    for case in cases:
        check_fp32_ieee()
        where = torch.device("cpu") if host_decode_from and case.size >= host_decode_from else device
        entries[case.name]["decode_device"] = str(where)
        timed(entries[case.name], "fp32_decode", lambda: record_image(pipe, case, os.path.join(out, case.name), where))
        log(f"fp32 decode {case.name} on {where}: {entries[case.name]['fp32_decode_wall_s']} s")
    write_thresholds(os.path.join(out, "thresholds.json"), errors, factor, session)

    revision = os.path.basename(os.path.normpath(checkpoint))
    with open(os.path.join(HERE, "pin.json"), encoding="utf-8") as f:
        pinned = json.load(f)
    size = sum(os.path.getsize(os.path.join(root, name)) for root, _, files in os.walk(out) for name in files)
    manifest = {
        "schema": "ignis-image-oracle-v1",
        "session": session,
        "checkpoint": {"repo": pinned["repo"] if pinned["revision"] == revision else None, "revision": revision},
        "versions": _versions(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "pipeline": "diffusers QwenImage21Pipeline, batch 1, use_kv_cache=True, default attention processor "
                    "(QwenImage21AttnProcessor); fp32 compute: weights bf16 at rest (the checkpoint's), the text "
                    "decoder layers, lm_head and the DiT blocks upcast to fp32 in their own forward, everything else "
                    "fp32; each call split at encode_prompt (the encoder phase for every case, then the DiT phase "
                    "replaying its output up to the final latents), then the decode phase (the pipeline's decode "
                    "tail, the VAE alone, on the host for the cases in decode_device cpu); the VAE keeps no "
                    "single-frame causal cache. All bit-identical to from_pretrained(torch_dtype=float32) on one "
                    "device (test_record.py)",
        "use_kv_cache": True,
        "tf32": tf32,
        "attention": attention,
        "noise": "numpy.random.default_rng(seed).standard_normal((1, 64, size/16, size/16), dtype=float32)",
        "cases": entries,
        "fields": FIELDS,
        "blocks": list(blocks),
        "block_steps": list(block_steps),
        "thresholds": "thresholds.json",
        "resources": resources,
        "wall_s": {"load": round(load_s, 1), "encoder_to_dit": round(switch_s, 1),
                   "total": round(time.time() - started, 1)},
        "bytes": size,
    }
    _write_json(os.path.join(out, "manifest.json"), manifest)
    log(f"done: {size / GB:.2f} GB in {out}, {manifest['wall_s']['total']} s")
    return manifest


# What `--replace-thresholds` alone supersedes: the recorder's bf16-only file and the first bound,
# both written before any ignis result
REPLACEABLE = (BF16_ONLY_SCHEMA, "ignis-image-thresholds-v2")


def run_yardstick(checkpoint, oracle, artifact, device, cases, sim_factor=SIM_FACTOR, bf16_factor=BF16_FACTOR,
                  replace=False, supersede=False, report=None, top=TOP_ROWS, min_free_ram_gb=YARDSTICK_MIN_FREE_RAM_GB,
                  min_free_vram_gb=MIN_FREE_VRAM_GB, max_vram_gib=MAX_VRAM_GIB, log=print):
    """One session of both yardsticks against an existing oracle, which it never records again (the
    fp32 oracle does not depend on the artifact), then thresholds.json (`write_bounds`) and the
    report (default ORACLE/yardstick-report.<session id>.json, written always; once the new file
    is in place, ORACLE/yardstick-report.json is that report, the earlier one archived under its
    own session). In order: the bf16 encoder on every case; the
    fp32 encoder replayed (its error against the oracle is the session's wiring check) and, for a
    case with an image, with DeepStack off (spec 03 AC 4's premise); the encoder made sim
    (`simquant`), the sim encoder; the encoder dropped, the bf16 DiT; the DiT and the VAE made sim,
    the sim DiT. The artifact's sha256 is computed beside them.

    An existing thresholds.json is superseded only before any ignis result exists: with `replace`
    if its schema is in REPLACEABLE, with `replace` and `supersede` otherwise. It is kept as
    thresholds.<its session id>.json (copied, then the new file replaces it in one step). A
    supersede is refused, after the report is written, if the velocity errors moved, or the errors
    after the vision span the replaced file also holds: the bf16's on the same oracle, the sim's
    on the same artifact (its sha256) too."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    import convert
    import pin
    import psutil
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    started = time.time()
    with open(os.path.join(oracle, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    sidecar = artifact + ".conversion.json"
    with open(sidecar, "rb") as f:
        raw = f.read()
    conversion = json.loads(raw)
    revision = os.path.basename(os.path.normpath(checkpoint))
    if not manifest["checkpoint"]["revision"] == conversion["source"]["revision"] == revision:
        raise ValueError(f"the oracle ({manifest['checkpoint']['revision']}), the artifact "
                         f"({conversion['source']['revision']}) and the checkpoint ({revision}) are not one revision")
    thresholds = os.path.join(oracle, "thresholds.json")
    old = kept = None
    if os.path.exists(thresholds):
        with open(thresholds, encoding="utf-8") as f:
            old = json.load(f)
        if not replace:
            raise FileExistsError(f"{thresholds} exists: --replace-thresholds supersedes it, kept under its session's name")
        if old["schema"] not in REPLACEABLE and not supersede:
            raise FileExistsError(f"{thresholds} is a bound ({old['schema']}): superseding it takes --supersede-bound "
                                  f"as well, and only before any ignis result exists")
        kept = os.path.join(oracle, f"thresholds.{old['session']['id'].replace(':', '')}.json")
        if os.path.exists(kept):
            raise FileExistsError(f"{kept} exists: a superseded file is never overwritten")
    free_ram = psutil.virtual_memory().available / GB
    if free_ram < min_free_ram_gb:
        raise RuntimeError(f"{free_ram:.1f} GB of host RAM free, below {min_free_ram_gb} GB")
    fp32_ieee()
    if device.type == "cuda":
        free_vram = torch.cuda.mem_get_info(device)[0] / GB
        if free_vram < min_free_vram_gb:
            raise RuntimeError(f"{free_vram:.1f} GB of VRAM free, below {min_free_vram_gb} GB")
        if max_vram_gib:
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, max_vram_gib * 2**30 / total), device)
    session = {"id": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **convert.tool_commit()}
    log(f"yardstick session {session['id']} on oracle session {manifest['session']['id']}")
    hashing = ThreadPoolExecutor(1)
    container_sha256 = hashing.submit(pin.sha256_file, artifact)   # ~1 min for 16.7 GB, beside the GPU work
    hashing.shutdown(wait=False)
    records = {t["source"]: t for t in conversion["tensors"]}
    container = Artifact(artifact)
    dirs = {case.name: os.path.join(oracle, case.name) for case in cases}
    wall, peaks = {}, {}

    def timed(key, work):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t = time.time()
        result = work()
        wall[key] = round(time.time() - t, 1)
        if device.type == "cuda":
            peaks[key] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        log(f"{key}: {wall[key]} s, peak {peaks.get(key)} GiB")
        return result

    pipe = timed("load", lambda: load_pipeline(checkpoint, device))
    checks = {case.name: {} for case in cases}
    # the encoder phase
    set_precision(pipe, "bf16")
    prompts16 = {case.name: timed(f"bf16_encoder_{case.name}", lambda: encode(pipe, case)) for case in cases}
    set_precision(pipe, "fp32")
    prompts_off = {}
    for case in cases:
        replay = _numpy(encode(pipe, case).embeds)
        checks[case.name]["fp32_replay_encoder"] = rel_l2(replay, np.load(os.path.join(dirs[case.name], "prompt_embeds.npy")))
        if case.image is not None:
            with deepstack_off(pipe):
                prompts_off[case.name] = encode(pipe, case)
    counts = {"text_encoder": timed("sim_install_text_encoder",
                                    lambda: simquant(pipe.text_encoder, "text_encoder", container, records))}
    prompts_sim = {case.name: timed(f"sim_encoder_{case.name}", lambda: encode(pipe, case)) for case in cases}
    # the DiT phase
    timed("encoder_to_dit", lambda: denoise_phase(pipe, device))
    set_precision(pipe, "bf16")
    bf16 = {case.name: timed(f"bf16_dit_{case.name}", lambda: yardstick(pipe, case, dirs[case.name], prompts16[case.name]))
            for case in cases}
    set_precision(pipe, "fp32")
    counts["transformer"] = timed("sim_install_dit", lambda: simquant(pipe.transformer, "transformer", container, records))
    counts["vae"] = simquant(pipe.vae, "vae", container, records)
    checked = sum(n for c in counts.values() for n in c.values())
    if checked != len(records):
        raise AssertionError(f"{checked} stored tensors checked, the artifact records {len(records)}")
    _guard_dit(pipe, device.type == "cuda")
    sim = {case.name: timed(f"sim_dit_{case.name}", lambda: yardstick(pipe, case, dirs[case.name], prompts_sim[case.name]))
           for case in cases}
    _guard_dit(pipe, False)

    conversion_sha256 = hashlib.sha256(raw).hexdigest()
    pending = os.path.join(oracle, f"thresholds.{session['id'].replace(':', '')}.new.json")
    value = write_bounds(pending, sim, bf16, sim_factor, bf16_factor, manifest["session"], session,
                         {"file": os.path.basename(artifact), "bytes": os.path.getsize(artifact),
                          "sha256": timed("artifact_sha256_wait", container_sha256.result),
                          "identity": container.directory.get("identity"),
                          "conversion_sha256": conversion_sha256, "tensors_checked": checked})
    moved = {}
    for case in cases:
        c, bounds = checks[case.name], value["cases"][case.name]
        if case.name in prompts_off:
            off = encoder_errors(_numpy(prompts_off[case.name].embeds), dirs[case.name])
            c["deepstack_off"] = {q: {"error": off[q], "bound": bounds[q]["bound"],
                                      "fails_the_bound": off[q] > bounds[q]["bound"]}
                                  for q in ("encoder", "encoder_after_vision")}
            c["deepstack_off"]["encoder_row0"] = {"error": off["encoder_row0"]}
        if old is not None and case.name in old["cases"]:
            was = old["cases"][case.name]["velocity"]
            was_artifact = old.get("artifact", {})
            same_artifact = (was_artifact["sha256"] == value["artifact"]["sha256"] if "sha256" in was_artifact
                             else was_artifact.get("conversion_sha256") == conversion_sha256)
            c["velocity_vs_replaced"] = {
                name: max(abs(x - y) for x, y in zip(errors[case.name]["velocity"], was[f"{name}_error"]))
                for name, errors in (("bf16", bf16), ("sim", sim))
                if f"{name}_error" in was and (name == "bf16" or same_artifact)}
            if any(c["velocity_vs_replaced"].values()):
                moved[case.name] = c["velocity_vs_replaced"]
            # a quantity the replaced file also measured, over the same rows, must not move either
            kept_rows = old["cases"][case.name].get("encoder_after_vision")
            if kept_rows is not None:
                c["after_vision_vs_replaced"] = {
                    name: abs(errors[case.name]["encoder_after_vision"] - kept_rows[f"{name}_error"])
                    for name, errors in (("bf16", bf16), ("sim", sim)) if name == "bf16" or same_artifact}
                if any(c["after_vision_vs_replaced"].values()):
                    moved[case.name] = dict(moved.get(case.name, {}), after_vision=c["after_vision_vs_replaced"])
    breakdown = {}
    for case in cases:
        breakdown[case.name] = {name: encoder_breakdown(_numpy(prompts[case.name].embeds), dirs[case.name], top)
                                for name, prompts in (("sim", prompts_sim), ("bf16", prompts16),
                                                      ("deepstack_off", prompts_off)) if case.name in prompts}
    result = {
        "session": session,
        "oracle_session": manifest["session"],
        "replaced": None if old is None else old.get("schema"),
        "errors": {"sim": sim, "bf16": bf16},
        "encoder_breakdown": breakdown,
        "checks": checks,
        "tensors_checked": counts,
        "wall_s": dict(wall, total=round(time.time() - started, 1)),
        "peak_vram_gib": peaks,
    }
    archive = report or os.path.join(oracle, f"yardstick-report.{session['id'].replace(':', '')}.json")
    _write_json(archive, result)
    if moved:
        os.remove(pending)
        raise AssertionError(f"the errors moved against the superseded {old['schema']} (max abs difference "
                             f"per yardstick): {moved}; {thresholds} is kept, the report is {archive}")
    if kept is not None:
        shutil.copyfile(thresholds, kept)
    os.replace(pending, thresholds)
    if report is None:
        latest = os.path.join(oracle, "yardstick-report.json")
        if os.path.exists(latest):
            with open(latest, encoding="utf-8") as f:
                earlier = os.path.join(oracle, f"yardstick-report.{json.load(f)['session']['id'].replace(':', '')}.json")
            if not os.path.exists(earlier):
                shutil.copyfile(latest, earlier)
        shutil.copyfile(archive, latest)
    log(f"done: thresholds.json and the report in {oracle}, {result['wall_s']['total']} s")
    return value, result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="record the cases in fp32, then the bf16 yardstick; writes thresholds.json")
    r.add_argument("--checkpoint", required=True, help="the pinned snapshot directory")
    r.add_argument("--out", default=ORACLE)
    r.add_argument("--device", default="cuda")
    r.add_argument("--cases", default=",".join(CASES), help="comma-separated, from " + ", ".join(CASES))
    r.add_argument("--blocks", default=",".join(map(str, BLOCKS)), help="DiT blocks whose outputs are recorded")
    r.add_argument("--block-steps", default=",".join(map(str, BLOCK_STEPS)),
                   help="denoising steps (0-based) whose block outputs are recorded")
    r.add_argument("--factor", type=float, default=FACTOR, help="threshold = factor x the bf16 error")
    r.add_argument("--min-free-disk-gb", type=float, default=MIN_FREE_DISK_GB, help="the disk floor kept free after the run")
    r.add_argument("--disk-budget-gb", type=float, default=DISK_BUDGET_GB, help="room the recordings may take")
    r.add_argument("--min-free-ram-gb", type=float, default=RUN_MIN_FREE_RAM_GB,
                   help="host RAM free at the start: the 2048^2 fp32 host decode peaks at ~36 GB")
    r.add_argument("--min-free-vram-gb", type=float, default=MIN_FREE_VRAM_GB)
    r.add_argument("--host-decode-from", type=int, default=HOST_DECODE_FROM,
                   help="decode images this many pixels a side or more on the host (0: never)")
    r.add_argument("--max-vram-gib", type=float, default=MAX_VRAM_GIB,
                   help="this process's VRAM cap in GiB, desktop excluded (0: none)")
    y = sub.add_parser("yardstick", help="the bf16 and the simulated W8A8 yardsticks against the existing oracle; "
                                         "writes thresholds.json (the bound) and a report")
    y.add_argument("--checkpoint", required=True, help="the pinned snapshot directory")
    y.add_argument("--oracle", default=ORACLE)
    y.add_argument("--artifact", default=ARTIFACT, help="the packed artifact, its .conversion.json beside it")
    y.add_argument("--device", default="cuda")
    y.add_argument("--cases", default=",".join(CASES), help="comma-separated, from " + ", ".join(CASES))
    y.add_argument("--sim-factor", type=float, default=SIM_FACTOR,
                   help="bound = max(sim-factor x the simulated W8A8 error, bf16-factor x the bf16 error)")
    y.add_argument("--bf16-factor", type=float, default=BF16_FACTOR)
    y.add_argument("--replace-thresholds", action="store_true",
                   help="supersede an existing thresholds.json of schema v1 or v2, kept as thresholds.<its session "
                        "id>.json; only before any ignis result exists")
    y.add_argument("--supersede-bound", action="store_true",
                   help="with --replace-thresholds: supersede a bound of any later schema too")
    y.add_argument("--report", help="the encoder breakdown and the checks (default ORACLE/yardstick-report.<session>.json)")
    y.add_argument("--top-rows", type=int, default=TOP_ROWS, help="rows listed by reference norm and by error in the report")
    y.add_argument("--min-free-ram-gb", type=float, default=YARDSTICK_MIN_FREE_RAM_GB)
    y.add_argument("--min-free-vram-gb", type=float, default=MIN_FREE_VRAM_GB)
    y.add_argument("--max-vram-gib", type=float, default=MAX_VRAM_GIB,
                   help="this process's VRAM cap in GiB, desktop excluded (0: none)")
    a = ap.parse_args(argv)
    ints = lambda text: tuple(int(x) for x in text.split(",") if x)
    log = lambda *x: print(*x, flush=True)
    what = "RECORD" if a.cmd == "run" else "YARDSTICK"
    try:
        if a.cmd == "run":
            run(a.checkpoint, a.out, a.device, [CASES[n] for n in a.cases.split(",")], blocks=ints(a.blocks),
                block_steps=ints(a.block_steps), factor=a.factor, min_free_disk_gb=a.min_free_disk_gb,
                disk_budget_gb=a.disk_budget_gb, min_free_ram_gb=a.min_free_ram_gb,
                min_free_vram_gb=a.min_free_vram_gb, max_vram_gib=a.max_vram_gib,
                host_decode_from=a.host_decode_from, log=log)
        else:
            run_yardstick(a.checkpoint, a.oracle, a.artifact, a.device, [CASES[n] for n in a.cases.split(",")],
                          sim_factor=a.sim_factor, bf16_factor=a.bf16_factor, replace=a.replace_thresholds,
                          supersede=a.supersede_bound, report=a.report, top=a.top_rows,
                          min_free_ram_gb=a.min_free_ram_gb, min_free_vram_gb=a.min_free_vram_gb,
                          max_vram_gib=a.max_vram_gib, log=log)
    except BaseException as e:
        print(f"{what} FAILED: {type(e).__name__}: {e}", flush=True)
        raise
    print(f"{what} DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
