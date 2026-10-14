"""A probe for the owner (#318 M3, informational): how far do the artifact's vision-tower formats
move the tower's outputs on the edit case's encoder image, next to plain bf16? It measured the
27B's formats first (Q4 tensors with 10-16% weight error: 42% at the merger), on which the owner
made the tower W8 throughout; it measures whatever recipe.py stores.

Three runs of the vision tower, its merger and its three DeepStack mergers, on the CPU:
- `fp32`: the checkpoint's weights, computing in fp32 (the reference);
- `artifact`: the weights the artifact holds (recipe.py's grouped formats, `row-split-k128-v1`,
  decoded to fp32; bf16 biases, norms and positions), in fp32;
- `bf16`: the checkpoint's weights, computing in bf16.
It reports the relative L2 of the merger output and of each DeepStack feature against `fp32`.

    python vision_probe.py --checkpoint SNAPSHOT_DIR --oracle ORACLE_DIR [--conversion SIDECAR]

The image is the recording's: ORACLE_DIR/edit/pixel_values.npy and image_grid_thw.npy. With
--conversion (the artifact's .conversion.json), every quantized payload's sha256 is checked against
the artifact's own record first.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

import recipe
import rowsplit

_PREFIX = "model.visual."


def vision_state(checkpoint):
    """(the vision config, {name under model.visual.: the checkpoint's bf16 tensor})."""
    from safetensors import safe_open
    from transformers import Qwen3VLConfig

    directory = os.path.join(checkpoint, "text_encoder")
    config = Qwen3VLConfig.from_pretrained(directory).vision_config
    state = {}
    files = sorted(f for f in os.listdir(directory) if f.endswith(".safetensors"))
    for name in files:
        with safe_open(os.path.join(directory, name), framework="pt") as f:
            for key in f.keys():
                if key.startswith(_PREFIX):
                    state[key[len(_PREFIX):]] = f.get_tensor(key)
    return config, state


def artifact_weights(state, conversion=None):
    """The state with every grouped-format tensor replaced by what the artifact decodes to (the
    converter's own encode, then decode): (state, the number of payloads checked against
    `conversion`'s sha256s). A payload that differs from the artifact's record is refused."""
    records = {} if conversion is None else {t["source"]: t for t in conversion["tensors"]}
    out, checked = dict(state), 0
    for name, t in state.items():
        item = recipe.plan("text_encoder", _PREFIX + name, list(t.shape))
        if item is None or item.encoding in ("bf16", "fp8"):
            continue
        payload = rowsplit.encode(t.reshape(item.shape), item.encoding)
        record = records.get("text_encoder/" + _PREFIX + name)
        if conversion is not None:
            if record is None or record["sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"{name}: the payload is not the artifact's {item.object}")
            checked += 1
        out[name] = rowsplit.decode(payload, item.encoding, item.shape).reshape(t.shape)
    return out, checked


def outputs(config, state, pixel_values, grid, dtype):
    """(the merger output, [the DeepStack features]) of the tower computing in `dtype`, as fp32."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config._attn_implementation = "sdpa"
    model = Qwen3VLVisionModel(config)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    model.to(dtype).eval()
    with torch.no_grad():
        result = model(torch.as_tensor(pixel_values).to(dtype), grid_thw=torch.as_tensor(grid))
    return result.pooler_output.float(), [d.float() for d in result.deepstack_features]


def _rel(x, ref):
    return float(torch.linalg.vector_norm((x - ref).double()) / torch.linalg.vector_norm(ref.double()))


def probe(config, state, pixel_values, grid, conversion=None):
    """The relative L2 errors of `artifact` and `bf16` against `fp32` (module docstring)."""
    reference = outputs(config, state, pixel_values, grid, torch.float32)
    quantized, checked = artifact_weights(state, conversion)
    runs = {"artifact": outputs(config, quantized, pixel_values, grid, torch.float32),
            "bf16": outputs(config, state, pixel_values, grid, torch.bfloat16)}
    result = {"payloads_checked": checked, "patches": int(np.prod(np.asarray(grid)[0])),
              "deepstack_layers": list(config.deepstack_visual_indexes)}
    for run, (merger, deepstack) in runs.items():
        result[run] = {"merger": _rel(merger, reference[0]),
                       "deepstack": [_rel(d, r) for d, r in zip(deepstack, reference[1])]}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--oracle", required=True, help="the recordings' directory (its edit case)")
    ap.add_argument("--conversion", help="the artifact's .conversion.json, to check the payloads against")
    a = ap.parse_args(argv)
    conversion = None
    if a.conversion:
        with open(a.conversion, encoding="utf-8") as f:
            conversion = json.load(f)
    config, state = vision_state(a.checkpoint)
    pixel_values = np.load(os.path.join(a.oracle, "edit", "pixel_values.npy"))
    grid = np.load(os.path.join(a.oracle, "edit", "image_grid_thw.npy"))
    print(json.dumps(probe(config, state, pixel_values, grid, conversion), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
