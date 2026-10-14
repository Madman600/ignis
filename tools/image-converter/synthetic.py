"""A synthetic Qwen-Image-2.1 checkpoint in the diffusers layout, CPU only, a few hundred kB:
2 DiT blocks, 2 text encoder layers, a 3-block vision tower with 3 DeepStack mergers, a small VAE
with the real one's five levels. Built from the modeling code (diffusers' and transformers'
own modules), so every tensor name and shape rule is the real checkpoint's; the weights are
seeded random. The geometry is crates/artifact's `QwenImageGeometry::fixture()`.

The small files are stand-ins (the tokenizer is "{}"): the converter copies them, nothing reads
them. Written as a Hugging Face cache snapshot, `<out>/snapshots/<REVISION>/`, with its pin.
"""
import json
import os

import torch

import pin

REPO = "synthetic/qwen-image-2.1-turbo"
REVISION = "0000000000000000000000000000000000f1c7e5"

TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, vocab_size=256, rms_norm_eps=1e-6, rope_theta=5000000,
            max_position_embeddings=4096,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True})
VISION = dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=4, temporal_patch_size=2,
              in_channels=3, spatial_merge_size=2, out_hidden_size=64, num_position_embeddings=16,
              deepstack_visual_indexes=[0, 1, 2])
DIT = dict(num_layers=2, num_attention_heads=4, attention_head_dim=16, mlp_ratio=3, in_channels=16, out_channels=16,
           patch_size=1, axes_dims_rope=[4, 6, 6], context_in_dim=64, causal_condition=True)
VAE = dict(base_dim=2, decoder_base_dim=3, z_dim=16, dim_mult=[1, 2, 4, 8, 8], num_res_blocks=2, in_channels=4,
           out_channels=4, is_residual=True, temperal_downsample=[False, True, True, True],
           latents_mean=[0.1 * i for i in range(16)], latents_std=[1.0 + 0.1 * i for i in range(16)])
SAMPLE_SIGMAS = [1.0, 0.75, 0.5]
STAND_INS = {
    "LICENSE": "synthetic checkpoint: no licence\n",
    "processor/chat_template.jinja": "{# synthetic #}\n",
    "processor/tokenizer.json": "{}\n",
    "processor/tokenizer_config.json": "{}\n",
    "text_encoder/generation_config.json": "{}\n",
}


def _randomize(module, g):
    for p in module.parameters():
        p.data = (torch.randn(p.shape, generator=g) * 0.05).to(torch.bfloat16)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def make(out, seed=0, poison=None):
    """Writes the checkpoint under `out` and returns (snapshot directory, its pin). `poison`, a
    (component, tensor name) pair, puts a NaN into that tensor (the pin is taken after, so only
    the converter's own check can refuse it)."""
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    snap = os.path.join(out, "snapshots", REVISION)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    text = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
        text_config=TEXT, vision_config=VISION, image_token_id=250, video_token_id=251,
        vision_start_token_id=252, vision_end_token_id=253, tie_word_embeddings=False))
    dit = QwenImage21Transformer2DModel(**DIT)
    vae = AutoencoderKLQwenImage21(**VAE)
    models = {"text_encoder": text, "transformer": dit, "vae": vae}
    for comp, model in models.items():
        _randomize(model, g)
        if poison and poison[0] == comp:
            dict(model.named_parameters())[poison[1]].data.view(-1)[0] = float("nan")
    text.save_pretrained(os.path.join(snap, "text_encoder"))
    # two shards and an index, as the real transformer
    dit.save_pretrained(os.path.join(snap, "transformer"), max_shard_size="150KB")
    vae.save_pretrained(os.path.join(snap, "vae"))
    _write(os.path.join(snap, "model_index.json"), json.dumps(
        {"_class_name": "QwenImage21Pipeline", "sample_sigmas": SAMPLE_SIGMAS}, indent=2) + "\n")
    _write(os.path.join(snap, "scheduler/scheduler_config.json"), json.dumps(
        {"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 1.0, "use_dynamic_shifting": False}, indent=2) + "\n")
    _write(os.path.join(snap, "processor/processor_config.json"), json.dumps(
        {"image_processor": {"patch_size": VISION["patch_size"], "merge_size": 2, "temporal_patch_size": 2,
                             "image_mean": [0.5] * 3, "image_std": [0.5] * 3}}, indent=2) + "\n")
    for path, text_ in STAND_INS.items():
        _write(os.path.join(snap, path), text_)
    paths = sorted(os.path.relpath(os.path.join(root, f), snap).replace(os.sep, "/")
                   for root, _, files in os.walk(snap) for f in files)
    p = pin.from_checkpoint(snap, REPO, REVISION, paths)
    return snap, p
