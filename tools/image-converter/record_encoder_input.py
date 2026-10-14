"""The CPU fixture of spec 03's front end (ACs 1, 2 and 6): what the pipeline hands its text
encoder, recorded from the pipeline's own code on the pinned processor, without weights that
matter. `crates/server/tests/image_encoder_input.rs` checks the server's tokenizer, template,
image preprocessing and mrope positions against it.

A `QwenImage21Pipeline` is built around the pinned checkpoint's processor and a tiny, seeded
Qwen3-VL (one text layer, one vision block: its outputs are not recorded). Each case calls the
pipeline's `_get_qwen_prompt_embeds`, which builds the raw template, composites RGBA on white,
runs the processor and the text encoder; hooks on the encoder record its inputs and the positions
its rotary embedding reads (as record.py's `_Capture` does). So the template with N condition
images, `drop_idx` and the mrope rule are diffusers' and transformers' own, not a copy of them.

    F:/ai/venvs/ignis-image/Scripts/python.exe tools/image-converter/record_encoder_input.py \
        [--snapshot SNAPSHOT_DIR] [--out crates/server/tests/fixtures/image_encoder_input.json]

The fixture holds the processor's config (`image_processor`, as `processor_config.json` has it)
and the image-pad id, then:
- `prompts`: per case the prompt, its condition images' sizes, `input_ids`, `drop_idx`, each
  image's `image_grid_thw` and the positions, (3, L) (t, h, w);
- `preprocess`: per case an image's size and its `image_grid_thw`, with the sha256 of its
  `pixel_values` (float32, little-endian, row-major) and its first and last eight values;
- `composite`: Pillow's paste of value v at alpha a over white, as the pipeline flattens a
  condition image: the rows for alpha 0, 128 and 255, and the sha256 of the whole table
  (`table[a][v]`, 65,536 bytes), which is checked against the integer formula here.

Every image is `pattern(width, height)`, RGBA, so the fixture carries no pixels: the test builds
the same image.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SNAPSHOT = os.environ.get(
    "IGNIS_IMAGE_SNAPSHOT",
    "F:/ai/models/hf/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba1986fcb905854dc8de2f52eee395f49b5b")
OUT = os.path.join(REPO, "crates", "server", "tests", "fixtures", "image_encoder_input.json")

# the real checkpoint's token ids, tiny everything else (test_record.py's, one layer and block)
TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, vocab_size=151936, rms_norm_eps=1e-6, rope_theta=5000000,
            max_position_embeddings=65536,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True})
VISION = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=16, temporal_patch_size=2,
              in_channels=3, spatial_merge_size=2, out_hidden_size=64, num_position_embeddings=16,
              deepstack_visual_indexes=[0], hidden_act="gelu_pytorch_tanh")

CJK_PROMPT = "一只狐狸站在雪地里，举着写有“ignis”的木牌。夕焼けの光、柔らかい影。눈 내리는 숲."
TEN = [(256, 256), (512, 288), (288, 512), (256, 448), (320, 256), (256, 320), (384, 384), (448, 256),
       (256, 512), (352, 288)]
PROMPT_CASES = [
    ("empty", "", []),
    ("cjk", CJK_PROMPT, []),
    ("2000 tokens", None, []),
    ("1 image", "Turn the shapes into polished glass", [(256, 256)]),
    ("1 image, empty prompt", "", [(288, 512)]),
    ("2 images", "Put the object from image 2 into image 1", [(512, 288), (288, 512)]),
    ("3 images", "Combine the three pictures into one scene", [(288, 512), (256, 448), (256, 256)]),
    ("10 images", "A collage of all ten images", TEN),
]
# 1:1, 16:9 and 9:16 at multiples of 32 (no resample), a 2-pixel-wide image (smart_resize scales
# it up to the minimum area) and an odd size (rounded to multiples of 32: one side up, one down)
PREPROCESS_CASES = [("1:1", 256, 256), ("16:9", 512, 288), ("9:16", 288, 512), ("2 pixels wide", 2, 64),
                    ("odd size", 450, 250)]
COMPOSITE_ALPHAS = (0, 128, 255)


def pattern(width, height):
    """The fixture's RGBA image: r = 3x + 5y, g = 7x + 2y + 85, b = xy + 170, a = 11x + 13y, each
    mod 256 (every alpha occurs, so the white composite changes most pixels)."""
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    rgba = np.stack([3 * x + 5 * y, 7 * x + 2 * y + 85, x * y + 170, 11 * x + 13 * y], axis=-1) % 256
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def long_prompt(tokenizer, tokens=2000):
    """A deterministic prompt of at least `tokens` tokens on its own."""
    subjects = ["a lighthouse", "an old fox", "a glass teapot", "a red bicycle", "a sleeping cat", "a paper lantern"]
    places = ["on a rocky coast", "in a snowy forest", "on a wooden table", "under neon rain", "beside a river"]
    styles = ["painted in oils", "as a pencil sketch", "in soft morning light", "with long shadows",
              "in watercolor", "photographed at dusk", "in the style of a woodcut"]
    sentences = []
    while len(tokenizer.encode(" ".join(sentences))) < tokens:
        i = len(sentences)
        sentences.append(f"{subjects[i % 6].capitalize()} {places[i % 5]}, {styles[i % 7]}, detail {i}.")
    return " ".join(sentences)


def pipeline(snapshot):
    from diffusers import QwenImage21Pipeline
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(os.path.join(snapshot, "processor"))
    with open(os.path.join(snapshot, "text_encoder", "config.json"), encoding="utf-8") as f:
        real = json.load(f)
    ids = {k: real[k] for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")}
    torch.manual_seed(0)
    encoder = Qwen3VLForConditionalGeneration(Qwen3VLConfig(text_config=TEXT, vision_config=VISION,
                                                            tie_word_embeddings=False, **ids))
    return QwenImage21Pipeline(vae=None, text_encoder=encoder, processor=processor, transformer=None,
                               scheduler=None)


def encode(pipe, prompt, images):
    """The encoder's inputs and rotary positions for one `_get_qwen_prompt_embeds` call."""
    captured = {}

    def inputs(module, args, kwargs):
        for key in ("input_ids", "mm_token_type_ids", "pixel_values", "image_grid_thw"):
            if kwargs.get(key) is not None:
                captured[key] = kwargs[key].detach().clone()

    def positions(module, args):
        captured["positions"] = args[1].detach().clone()

    text_model = pipe.text_encoder.model.language_model
    handles = [pipe.text_encoder.register_forward_pre_hook(inputs, with_kwargs=True),
               text_model.rotary_emb.register_forward_pre_hook(positions)]
    try:
        with torch.no_grad():
            pipe._get_qwen_prompt_embeds(prompt, images or None, device="cpu")
    finally:
        for h in handles:
            h.remove()
    return captured


def composite_table():
    """Pillow's paste over white for every (alpha, value), checked against the integer formula."""
    a, v = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    rgba = np.stack([v, v, v, a], axis=-1).astype(np.uint8)
    image = Image.fromarray(rgba, "RGBA")
    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    table = np.asarray(white)[..., 0]
    t = 255 * (255 - a) + v * a + 128
    if not np.array_equal(table, ((t >> 8) + t) >> 8):
        raise AssertionError("Pillow's paste is not ((t >> 8) + t) >> 8, t = 255(255 - a) + va + 128")
    return table


def record(snapshot):
    import diffusers
    import PIL
    import torchvision
    import transformers

    pipe = pipeline(snapshot)
    prompts = []
    for name, prompt, sizes in PROMPT_CASES:
        if prompt is None:
            prompt = long_prompt(pipe.processor.tokenizer)
        got = encode(pipe, prompt, [pattern(w, h) for w, h in sizes])
        grids = got["image_grid_thw"].tolist() if "image_grid_thw" in got else []
        prompts.append({"name": name, "prompt": prompt, "images": [list(s) for s in sizes],
                        "drop_idx": pipe._drop_idx, "input_ids": got["input_ids"][0].tolist(),
                        "image_grid_thw": grids, "positions": got["positions"][:, 0].tolist()})
    preprocess = []
    for name, width, height in PREPROCESS_CASES:
        got = encode(pipe, "x", [pattern(width, height)])
        values = got["pixel_values"].numpy().astype("<f4")
        preprocess.append({"name": name, "width": width, "height": height,
                           "image_grid_thw": got["image_grid_thw"][0].tolist(), "shape": list(values.shape),
                           "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                           "first": values.reshape(-1)[:8].tolist(), "last": values.reshape(-1)[-8:].tolist()})
    table = composite_table()
    composite = {"alphas": {str(a): table[a].tolist() for a in COMPOSITE_ALPHAS},
                 "table_sha256": hashlib.sha256(table.astype(np.uint8).tobytes()).hexdigest()}
    with open(os.path.join(snapshot, "processor", "processor_config.json"), encoding="utf-8") as f:
        image_processor = json.load(f)["image_processor"]
    return {
        "recipe": {
            "script": "tools/image-converter/record_encoder_input.py",
            "snapshot": os.path.basename(os.path.normpath(snapshot)),
            "versions": {"diffusers": diffusers.__version__, "transformers": transformers.__version__,
                         "torch": torch.__version__, "torchvision": torchvision.__version__,
                         "pillow": PIL.__version__, "numpy": np.__version__},
            "image_processor": type(pipe.processor.image_processor).__name__,
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        },
        "image_processor": image_processor,
        "image_token_id": pipe._img_token_id,
        "prompts": prompts,
        "preprocess": preprocess,
        "composite": composite,
    }


def dumps(fixture):
    """JSON with every list of numbers on one line."""
    text = json.dumps(fixture, indent=1, ensure_ascii=False)
    return re.sub(r"\[\s*((?:-?[\d.e+-]+,\s*)*-?[\d.e+-]+)\s*\]",
                  lambda m: "[" + ", ".join(n.strip() for n in m.group(1).split(",")) + "]", text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot", default=SNAPSHOT)
    p.add_argument("--out", default=OUT)
    args = p.parse_args(argv)
    fixture = record(args.snapshot)
    text = dumps(fixture)
    if json.loads(text) != json.loads(json.dumps(fixture)):
        raise AssertionError("the compact JSON does not round-trip")
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    print(f"wrote {args.out}: {len(fixture['prompts'])} prompts, {len(fixture['preprocess'])} images")


if __name__ == "__main__":
    sys.exit(main())
