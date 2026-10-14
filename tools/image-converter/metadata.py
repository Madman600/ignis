"""What the artifact carries beside the weights (spec 01): the converter-written resource
`frontend/image_model.json`, and the VAE's convolution shapes for `conversion.json`.

The prompt templates are not in the checkpoint: they are diffusers' `QwenImage21Pipeline.__init__`
(diffusers 0.41.0, `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`), copied here
verbatim; test_metadata.py checks them against the installed diffusers source. The traps the
recorder reproduces (spec 01) are the pipeline's, not this module's: the prompt is the raw
template string, not `apply_chat_template`; `drop_idx` is the system message's length through
the chat template; an empty prompt becomes " ".
"""
import json
import os

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
TEMPLATE_TI2I = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
# condition image i (from 1) is "<image{i}><|vision_start|><|image_pad|><|vision_end|>", the
# slots joined by one space, in place of the first slot of TEMPLATE_TI2I
IMAGE_SLOT = "<image{}><|vision_start|><|image_pad|><|vision_end|>"
IMAGE_SLOT_SEPARATOR = " "
EMPTY_PROMPT = " "


def _json(directory, path):
    with open(os.path.join(directory, path), encoding="utf-8") as f:
        return json.load(f)


def image_model(checkpoint, pin, diffusers_version):
    """frontend/image_model.json: the sampler, the processor's image config, the latent
    normalization, the prompt templates and the source, for the image leaf and the server."""
    vae = _json(checkpoint, "vae/config.json")
    return {
        "schema": "ignis-image-model-v1",
        "source": {"repo": pin["repo"], "revision": pin["revision"]},
        "sample_sigmas": _json(checkpoint, "model_index.json")["sample_sigmas"],
        "scheduler": _json(checkpoint, "scheduler/scheduler_config.json"),
        "image_processor": _json(checkpoint, "processor/processor_config.json")["image_processor"],
        "latents_mean": vae["latents_mean"],
        "latents_std": vae["latents_std"],
        "prompt": {
            "system": SYSTEM_PROMPT,
            "template_t2i": TEMPLATE_T2I,
            "template_ti2i": TEMPLATE_TI2I,
            "image_slot": IMAGE_SLOT,
            "image_slot_separator": IMAGE_SLOT_SEPARATOR,
            "empty_prompt": EMPTY_PROMPT,
            "from": f"diffusers {diffusers_version} QwenImage21Pipeline.__init__",
        },
    }


def vae_convs(vae_config):
    """Every convolution of the VAE as spec 02 implements it: the weight's object name, channels,
    kernel, stride, the zero padding applied before it as [left, right, top, bottom], and the 2x
    nearest-exact upsampling that precedes the decoder's upsampler convolutions. Read off
    diffusers' own modules (built on the meta device): QwenImage21CausalConv3d pads in forward
    (`_padding`) and the downsamplers pad (0, 1, 0, 1) with a ZeroPad2d before a stride-2 conv."""
    import torch
    from torch import nn
    from diffusers import AutoencoderKLQwenImage21
    from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as vae_module

    config = {k: v for k, v in vae_config.items() if not k.startswith("_")}
    with torch.device("meta"):
        vae = AutoencoderKLQwenImage21.from_config(config)
    before = {}
    for name, module in vae.named_modules():
        if isinstance(module, nn.Sequential):
            children = list(module.named_children())
            for (_, first), (second_name, second) in zip(children, children[1:]):
                key = f"{name}.{second_name}"
                if isinstance(first, nn.ZeroPad2d):
                    before[key] = ("pad", list(first.padding))
                elif isinstance(first, vae_module.QwenImage21Upsample):
                    before[key] = ("upsample", f"{first.mode} x{first.scale_factor[0]:g}")
    convs = []
    for name, module in vae.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if isinstance(module, vae_module.QwenImage21CausalConv3d):
            padding = list(module._padding)
        else:
            ph, pw = module.padding
            padding = [pw, pw, ph, ph]
        kind, value = before.get(name, (None, None))
        if kind == "pad":
            padding = [a + b for a, b in zip(padding, value)]
        convs.append({"object": f"vae/{name}.weight", "in_channels": module.in_channels,
                      "out_channels": module.out_channels, "kernel": list(module.kernel_size),
                      "stride": list(module.stride), "padding": padding,
                      "upsample_before": value if kind == "upsample" else None})
    return convs
