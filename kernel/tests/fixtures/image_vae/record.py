"""Record the VAE shortcut and upsample fixture (spec image-model/02 acceptance 6, GitHub #319):
diffusers' own QwenImage21AvgDown3D, QwenImage21DupUp3D (first_chunk, one frame) and
QwenImage21Upsample (nearest-exact 2x) at every channel configuration the VAE uses, on small
spatial sizes, CPU only.

Run from the repository root with the image venv (no GPU is touched):

    CUDA_VISIBLE_DEVICES= F:/ai/venvs/ignis-image/Scripts/python.exe kernel/tests/fixtures/image_vae/record.py

It writes, next to itself:

  shortcuts.bin (IGNFX001, kernel/tests/moe_fixture.h reads it), per case NAME:
      NAME.geometry  i32  [7]          kind (0 AvgDown, 1 DupUp, 2 upsample), in, out, factor_t,
                                       factor_s, height, width (the input's)
      NAME.x         f32  [H][W][in]   the input, NHWC
      NAME.y         f32  [Ho][Wo][out] the module on x in fp32, NHWC
      NAME.y_bf16    bf16 [Ho][Wo][out] the module on x rounded to bf16, in bf16
  provenance.json
      versions and the cases.

The CTest (kernel/tests/test_image_vae.cu) holds the ops to these outputs: bit for bit in fp32,
and in bf16 bit for bit. The inputs span a wide dynamic range (so a different
summation order shows) and hold signed zeros and subnormals.
"""

from __future__ import annotations

import datetime
import json
import os
import struct

import torch

from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import (
    QwenImage21AvgDown3D,
    QwenImage21DupUp3D,
    QwenImage21Upsample,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, kind, in, out, factor_t, factor_s, height, width): the encoder's ResidualDownBlock
# shortcuts, the decoder's ResidualUpBlock shortcuts (diffusers' config: temperal_downsample
# [False, True, True, True], dim_mult [1, 2, 4, 8, 8], base 96 / decoder 144) and the resample's
# upsample. Heights and widths differ (and AvgDown has two output rows, DupUp two input rows), so
# a row stride mixed up with a column stride, or a row index never above 0, shows.
CASES = [
    ("avg_96_96_t1_s2", 0, 96, 96, 1, 2, 4, 6),
    ("avg_96_192_t2_s2", 0, 96, 192, 2, 2, 4, 6),
    ("avg_192_384_t2_s2", 0, 192, 384, 2, 2, 4, 6),
    ("avg_384_768_t2_s2", 0, 384, 768, 2, 2, 4, 6),
    ("avg_768_768_t1_s1", 0, 768, 768, 1, 1, 2, 3),
    ("dup_1152_1152_t2_s2", 1, 1152, 1152, 2, 2, 2, 3),
    ("dup_1152_576_t2_s2", 1, 1152, 576, 2, 2, 2, 3),
    ("dup_576_288_t1_s2", 1, 576, 288, 1, 2, 2, 3),
    ("upsample_8", 2, 8, 8, 1, 2, 3, 5),
]


def make_input(seed: int, c: int, h: int, w: int) -> torch.Tensor:
    """[1, c, 1, h, w] fp32 over a wide range, with signed zeros and subnormals."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(c * h * w, generator=g) * torch.exp(2.0 * torch.randn(c * h * w, generator=g))
    i = torch.arange(c * h * w)
    x = torch.where(i % 97 == 0, torch.tensor(-0.0), x)
    x = torch.where(i % 89 == 0, torch.tensor(0.0), x)
    x = torch.where(i % 83 == 0, torch.sign(x) * 3e-39, x)
    return x.view(c, h, w).unsqueeze(0).unsqueeze(2).contiguous()


def run(kind: int, cin: int, cout: int, ft: int, fs: int, x: torch.Tensor) -> torch.Tensor:
    if kind == 0:
        return QwenImage21AvgDown3D(cin, cout, ft, fs)(x)
    if kind == 1:
        return QwenImage21DupUp3D(cin, cout, ft, fs)(x, first_chunk=True)
    return QwenImage21Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact")(x[:, :, 0]).unsqueeze(2)


def nhwc(t: torch.Tensor) -> torch.Tensor:
    assert t.shape[0] == 1 and t.shape[2] == 1, t.shape
    return t[0, :, 0].permute(1, 2, 0).contiguous()


def main() -> None:
    torch.set_num_threads(1)
    with open(os.path.join(HERE, "shortcuts.bin"), "wb") as f:
        f.write(b"IGNFX001")

        def put(name: str, code: int, t: torch.Tensor) -> None:
            nb = name.encode("utf-8")
            payload = t.contiguous().numpy().tobytes()
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<II", code, t.dim()))
            for d in t.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<Q", len(payload)))
            f.write(payload)

        for seed, (name, kind, cin, cout, ft, fs, h, w) in enumerate(CASES):
            x = make_input(1000 + seed, cin, h, w)
            y = run(kind, cin, cout, ft, fs, x)
            y16 = run(kind, cin, cout, ft, fs, x.to(torch.bfloat16))
            assert y.dtype == torch.float32 and y16.dtype == torch.bfloat16
            put(name + ".geometry", 3, torch.tensor([kind, cin, cout, ft, fs, h, w], dtype=torch.int32))
            put(name + ".x", 6, nhwc(x))
            put(name + ".y", 6, nhwc(y))
            put(name + ".y_bf16", 5, nhwc(y16).view(torch.int16))

    import diffusers

    provenance = {
        "recorded": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "device": "cpu, torch.set_num_threads(1)",
        "oracle": "diffusers.models.autoencoders.autoencoder_kl_qwenimage21: QwenImage21AvgDown3D, "
        "QwenImage21DupUp3D(first_chunk=True), QwenImage21Upsample(nearest-exact 2x), one frame",
        "cases": [dict(zip(("name", "kind", "in", "out", "factor_t", "factor_s", "height", "width"), c)) for c in CASES],
    }
    with open(os.path.join(HERE, "provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(provenance, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
