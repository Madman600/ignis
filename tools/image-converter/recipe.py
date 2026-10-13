"""The tensor map: each checkpoint tensor's container object, encoding and stored shape (spec 01's
encoding table).

| part | encoding |
|---|---|
| DiT block linears (to_q/k/v, to_out.0, img_mlp proj/gate_layer/out), img_in, txt_in's two linears, proj_out | fp8 |
| text encoder decoder-layer linears (q/k/v/o, gate/up/down) | fp8 |
| DiT modulation, timestep embedder, norm_out's linear (its modulation), every norm | bf16 |
| token embedding (read on the host) | bf16 |
| vision tower, merger, DeepStack mergers | w8 for every matrix (the patch embedding, qkv/output/fc1/fc2, the mergers), bf16 biases, norms, positions |
| VAE, quant_conv, post_quant_conv | bf16 |

Not stored: `lm_head` and the text model's final norm (the encoder's output is taken before it).

Object names: `text_encoder/` + the checkpoint name without `model.language_model.`;
`transformer/` and `vae/` + the checkpoint name; the vision tower under the 27B artifact's own
names (`vision/layers/{b}/attention/qkv`, ...: crates/artifact/src/inventory.rs
`vision_scope_27b`), though not in its formats: W8 everywhere instead of the 27B's Q4/Q5/Q6
blocks (the owner's call of 2026-10-13, on the tower's measured error); the DeepStack mergers sit
beside the merger as `vision/deepstack/{i}/...`. A tensor the map does not name is refused, by
name.
"""
import math
import re
from collections import namedtuple

Item = namedtuple("Item", "object encoding shape")

# encoding -> (container format, layout)
FORMATS = {
    "fp8": ("FP8_E4M3FN_ROW_BF16S", "row-scale-v1"),
    "bf16": ("BF16", "contiguous-le-v1"),
    "q4": ("Q4G64_F16S", "row-split-k128-v1"),
    "q5": ("Q5G64_F16S", "row-split-k128-v1"),
    "q6": ("Q6G64_F16S", "row-split-k128-v1"),
    "w8": ("W8G32_F16S", "row-split-k128-v1"),
}

_TEXT = "model.language_model."
_TEXT_LINEAR = re.compile(r"layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight")
_TEXT_NORM = re.compile(r"layers\.\d+\.(input_layernorm|post_attention_layernorm|self_attn\.[qk]_norm)\.weight")
_TEXT_DROPPED = ("lm_head.weight", _TEXT + "norm.weight")

_VISION = "model.visual."
_VISION_BLOCK = re.compile(r"blocks\.(\d+)\.(.+)")
_VISION_BLOCK_PARTS = {
    "attn.qkv.weight": ("attention/qkv", "w8"),
    "attn.qkv.bias": ("attention/qkv_bias", "bf16"),
    "attn.proj.weight": ("attention/output", "w8"),
    "attn.proj.bias": ("attention/output_bias", "bf16"),
    "mlp.linear_fc1.weight": ("mlp/fc1", "w8"),
    "mlp.linear_fc1.bias": ("mlp/fc1_bias", "bf16"),
    "mlp.linear_fc2.weight": ("mlp/fc2", "w8"),
    "mlp.linear_fc2.bias": ("mlp/fc2_bias", "bf16"),
    "norm1.weight": ("norm1/weight", "bf16"),
    "norm1.bias": ("norm1/bias", "bf16"),
    "norm2.weight": ("norm2/weight", "bf16"),
    "norm2.bias": ("norm2/bias", "bf16"),
}
_MERGER = re.compile(r"(merger|deepstack_merger_list\.(\d+))\.(.+)")
_MERGER_PARTS = {
    "linear_fc1.weight": ("fc1", "w8"),
    "linear_fc1.bias": ("fc1_bias", "bf16"),
    "linear_fc2.weight": ("fc2", "w8"),
    "linear_fc2.bias": ("fc2_bias", "bf16"),
    "norm.weight": ("norm/weight", "bf16"),
    "norm.bias": ("norm/bias", "bf16"),
}

_DIT_LINEAR = re.compile(r"transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0)|img_mlp\.(proj|gate_layer|out))\.weight"
                         r"|img_in\.weight|txt_in\.(in_layer|out_layer)\.weight|proj_out\.weight")
_DIT_BF16 = re.compile(r"transformer_blocks\.\d+\.attn\.norm_[qk]\.weight|modulation\.1\.weight"
                       r"|time_text_embed\.timestep_embedder\.linear_[12]\.weight|norm_out\.linear\.weight"
                       r"|txt_in\.text_norm\.weight")


def _unknown(comp, name):
    return ValueError(f"{comp} tensor {name} is not in the image converter's tensor map")


def _vision(name, shape):
    rest = name[len(_VISION):]
    if rest == "patch_embed.proj.weight":
        # the 27B's patch embedding: [hidden, in_channels * temporal * patch * patch]
        return Item("vision/patch_embedding", "w8", [shape[0], math.prod(shape[1:])])
    if rest == "patch_embed.proj.bias":
        return Item("vision/patch_embedding_bias", "bf16", shape)
    if rest == "pos_embed.weight":
        return Item("vision/position_embedding", "bf16", shape)
    m = _VISION_BLOCK.fullmatch(rest)
    if m and m.group(2) in _VISION_BLOCK_PARTS:
        suffix, enc = _VISION_BLOCK_PARTS[m.group(2)]
        return Item(f"vision/layers/{m.group(1)}/{suffix}", enc, shape)
    m = _MERGER.fullmatch(rest)
    if m and m.group(3) in _MERGER_PARTS:
        suffix, enc = _MERGER_PARTS[m.group(3)]
        where = "merger" if m.group(2) is None else f"deepstack/{m.group(2)}"
        return Item(f"vision/{where}/{suffix}", enc, shape)
    return None


def plan(comp, name, shape):
    """The Item a checkpoint tensor becomes, or None when it is not stored. Raises ValueError,
    naming the tensor, for a tensor the map does not know."""
    shape = list(shape)
    if comp == "text_encoder":
        if name in _TEXT_DROPPED:
            return None
        if name.startswith(_VISION):
            item = _vision(name, shape)
            if item is None:
                raise _unknown(comp, name)
            return item
        if not name.startswith(_TEXT):
            raise _unknown(comp, name)
        rest = name[len(_TEXT):]
        if rest == "embed_tokens.weight" or _TEXT_NORM.fullmatch(rest):
            return Item("text_encoder/" + rest, "bf16", shape)
        if _TEXT_LINEAR.fullmatch(rest):
            return Item("text_encoder/" + rest, "fp8", shape)
        raise _unknown(comp, name)
    if comp == "transformer":
        if _DIT_LINEAR.fullmatch(name):
            return Item("transformer/" + name, "fp8", shape)
        if _DIT_BF16.fullmatch(name):
            return Item("transformer/" + name, "bf16", shape)
        raise _unknown(comp, name)
    if comp == "vae":
        return Item("vae/" + name, "bf16", shape)
    raise _unknown(comp, name)
