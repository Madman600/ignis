"""Row-split payloads (Q4/Q5/Q6 G64, W8 G32): the bytes the 27B's vision tower is stored in."""
import json
import os
import struct
import sys

import pytest
import torch

import rowsplit

# the owner's layout; the tests that need them skip without them
NINFER = os.environ.get("IGNIS_NINFER_SOURCE", "F:/ai/q38/ninfer")
ARTIFACT_27B = os.environ.get("IGNIS_ARTIFACT_27B", "F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer")


def test_payload_geometry_is_the_readers():
    # crates/artifact/src/lib.rs row_split_geometry_q4: [128, 128] -> 8704 bytes
    assert rowsplit.geometry("q4", (128, 128)).payload_bytes == 8704
    # K padded to 128; each plane starts on a multiple of 256
    assert rowsplit.geometry("q5", (3, 100)).payload_bytes == 256 + 256 + 12      # codes 192, high 48, scales 12
    assert rowsplit.geometry("w8", (3, 100)).payload_bytes == 512 + 24            # codes 384, scales 24
    assert rowsplit.geometry("q6", (2, 128)).payload_bytes == 256 + 256 + 8       # codes 128, high 64, scales 8
    for enc, shape in (("q4", (37, 200)), ("q5", (5, 64)), ("q6", (9, 300)), ("w8", (7, 33))):
        assert len(rowsplit.encode(torch.randn(shape), enc)) == rowsplit.geometry(enc, shape).payload_bytes


def test_codes_and_planes_are_laid_out_as_the_kernels_read_them():
    s = 2.0 ** -6
    w = torch.zeros(1, 128)
    w[0, 0], w[0, 1], w[0, 2] = 1 * s, -1 * s, 7 * s          # group 0's amax 7s: the scale is s
    p = rowsplit.encode(w, "q4")
    assert p[0] == 0xF1 and p[1] == 0x07                      # value 2j low nibble, 2j+1 high nibble
    assert torch.equal(rowsplit.decode(p, "q4", (1, 128)), w)
    w5 = torch.zeros(1, 128)
    w5[0, 1], w5[0, 2] = -9 * s, 15 * s                       # -9 = 0b10111: low nibble 7, high bit 1
    p5 = rowsplit.encode(w5, "q5")
    assert p5[0] == 0x70 and p5[256] == 0x02                  # the high plane starts at 256, bit i = value i
    assert torch.equal(rowsplit.decode(p5, "q5", (1, 128)), w5)
    w6 = torch.zeros(1, 128)
    w6[0, 3] = -31 * s
    w6[0, 0] = 31 * s
    p6 = rowsplit.encode(w6, "q6")
    # value 3's code -31 is 0b100001 in 6 bits: its high bits 0b10 sit at bits 6-7 of high byte 0
    assert p6[256] == 0b10 << 6 | 0b01 and torch.equal(rowsplit.decode(p6, "q6", (1, 128)), w6)
    w8 = torch.zeros(1, 32)
    w8[0, 5] = -127 * s
    p8 = rowsplit.encode(w8, "w8")
    assert p8[5] == 0x81 and torch.equal(rowsplit.decode(p8, "w8", (1, 32)), w8)


def test_the_round_trip_is_within_half_a_step():
    g = torch.Generator().manual_seed(0)
    # scales in fp16's normal range: a scale rounds by at most 2^-11, which moves the largest code
    # of a group (qmax) by qmax * 2^-11 steps
    w = torch.randn(48, 300, generator=g) * torch.logspace(-2, 1, 48)[:, None]
    for enc in rowsplit.SPECS:
        spec = rowsplit.SPECS[enc]
        back = rowsplit.decode(rowsplit.encode(w, enc), enc, w.shape)
        k_pad = rowsplit.geometry(enc, w.shape).k_pad
        padded = torch.nn.functional.pad(w, (0, k_pad - w.shape[1])).reshape(48, -1, spec.group_size)
        step = (padded.abs().amax(-1) / spec.qmax).repeat_interleave(spec.group_size, 1)[:, :w.shape[1]]
        bound = step * (0.5 * (1 + 2 ** -11) + spec.qmax * 2 ** -11)
        assert torch.all((back - w).abs() <= bound), enc


def test_an_all_zero_group_decodes_to_zero_and_nan_is_refused():
    p = rowsplit.encode(torch.zeros(2, 64), "q5")
    assert torch.equal(rowsplit.decode(p, "q5", (2, 64)), torch.zeros(2, 64))
    with pytest.raises(ValueError, match="NaN or infinity"):
        rowsplit.encode(torch.tensor([[1.0, float("nan")] + [0.0] * 62]), "q4")


@pytest.mark.skipif(not os.path.isdir(NINFER), reason=f"{NINFER} (the reference converter) is not here")
def test_the_bytes_are_the_reference_converters():
    sys.path.insert(0, NINFER)
    try:
        from tools.convert.common.quantize import quantize_and_encode
    finally:
        sys.path.remove(NINFER)
    g = torch.Generator().manual_seed(1)
    for enc, fmt in (("q4", "Q4G64_F16S"), ("q5", "Q5G64_F16S"), ("q6", "Q6G64_F16S"), ("w8", "W8G32_F16S")):
        for shape in ((37, 200), (64, 1152)):
            w = (torch.randn(shape, generator=g) * 0.02).to(torch.bfloat16)
            assert rowsplit.encode(w, enc) == quantize_and_encode(w, fmt, device="cpu"), (enc, shape)


@pytest.mark.skipif(not os.path.isfile(ARTIFACT_27B), reason=f"{ARTIFACT_27B} is not here")
def test_the_27b_vision_objects_have_these_sizes():
    with open(ARTIFACT_27B, "rb") as f:
        n = struct.unpack("<Q", f.read(16)[8:])[0]
        objects = json.loads(f.read(n))["objects"]
    enc_of = {"Q4G64_F16S": "q4", "Q5G64_F16S": "q5", "Q6G64_F16S": "q6", "W8G32_F16S": "w8"}
    checked = 0
    for o in objects:
        if o["name"].startswith("vision/") and o["kind"] == "tensor" and o["format"] in enc_of:
            assert rowsplit.geometry(enc_of[o["format"]], o["shape"]).payload_bytes == o["bytes"], o["name"]
            checked += 1
    assert checked == 1 + 27 * 4 + 2
