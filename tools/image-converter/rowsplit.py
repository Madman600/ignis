"""Grouped symmetric quantization in the `row-split-k128-v1` layout: Q4G64_F16S, Q5G64_F16S,
Q6G64_F16S and W8G32_F16S, the formats the 27B's vision tower is stored in. The image model's
tower is W8 throughout (recipe.py).

Adapted op (ADR 0051), copied from the reference converter so the image model's vision tower is
encoded exactly as the 27B's formats are:
  upstream  gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1 (kernel/vendor's pin)
  paths     tools/convert/common/quantize.py (`_canonical_scale_words`, `quantize_matrix`)
            tools/artifact/layouts.py (`row_split_geometry`, `_pack_low_nibbles`, `_pack_high_bits`,
            `assemble_row_planes`, `_unpack_codes`, `dequantize_row_split`)
            tools/artifact/numeric.py (the four format specs)
  licence   Apache-2.0
Changes: one module, no format registry, CPU only, no chunking; the decoder returns fp32.

A row of K values is zero-padded to a multiple of 128 and cut in groups (64, or 32 for W8). A
group's scale is binary16(amax / qmax) (the smallest positive binary16 when that underflows), its
codes round(v / scale) clamped to [qmin, qmax]. The payload is the code plane (low nibbles, two
per byte, the even index in the low nibble; W8: one int8 per byte), then from the next multiple
of 256 the high-bit plane (Q5: one bit per value, Q6: two, value i at bit i*(bits-4)), then from
the next multiple of 256 the fp16 scale plane, group-major ([row][group]).
"""
from collections import namedtuple

import numpy as np
import torch

PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128
_FP16_MIN_SUBNORMAL = 2.0 ** -24

Spec = namedtuple("Spec", "bits group_size qmin qmax")
SPECS = {
    "q4": Spec(4, 64, -8, 7),
    "q5": Spec(5, 64, -16, 15),
    "q6": Spec(6, 64, -32, 31),
    "w8": Spec(8, 32, -127, 127),
}

Geometry = namedtuple("Geometry", "n k k_pad groups_per_row base_bytes high_offset high_bytes scale_offset "
                                  "scale_bytes payload_bytes")


def _align(value, alignment):
    return -(-value // alignment) * alignment


def geometry(encoding, shape):
    spec = SPECS[encoding]
    n, k = shape
    k_pad = _align(k, K_ALIGNMENT)
    groups = k_pad // spec.group_size
    base_per_group = spec.group_size if spec.bits == 8 else spec.group_size // 2
    high_per_group = 0 if spec.bits in (4, 8) else spec.group_size * (spec.bits - 4) // 8
    base_bytes = n * groups * base_per_group
    high_bytes = n * groups * high_per_group
    scale_bytes = n * groups * 2
    high_offset = _align(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + _align(high_bytes, PLANE_ALIGNMENT)
    return Geometry(n, k, k_pad, groups, base_bytes, high_offset, high_bytes, scale_offset, scale_bytes,
                    scale_offset + scale_bytes)


def _scale_words(max_abs, qmax):
    """binary16 scales and their binary32 reciprocals, divided in binary64 (ninfer's
    `_canonical_scale_words`: CUDA division is not correctly rounded at every binary16 boundary)."""
    host_max = max_abs.numpy().astype(np.float32, copy=False)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = (host_max.astype(np.float64) / float(qmax)).astype(np.float32).astype(np.float16)
    underflow = (scale == 0) & (host_max > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.float16(_FP16_MIN_SUBNORMAL)
    if np.any((host_max > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("grouped quantization scale is not finite and positive")
    reciprocal = np.zeros(host_max.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def quantize(weight, encoding):
    """(codes int8 [N, groups, group_size], scales float16 [N, groups]) of a 2-D weight."""
    spec = SPECS[encoding]
    if weight.dim() != 2:
        raise ValueError(f"row-split quantization needs a 2-D weight, got {tuple(weight.shape)}")
    g = geometry(encoding, tuple(weight.shape))
    logical = weight.detach().to(device="cpu", dtype=torch.float32)
    if g.k_pad != g.k:
        physical = torch.zeros((g.n, g.k_pad), dtype=torch.float32)
        physical[:, :g.k] = logical
        logical = physical
    grouped = logical.reshape(g.n, g.groups_per_row, spec.group_size)
    scales, reciprocal = _scale_words(grouped.abs().amax(dim=2), spec.qmax)
    codes = torch.clamp(torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax).to(torch.int8)
    return codes, scales


def _pack(codes, spec):
    flat = codes.reshape(-1, spec.group_size)
    if spec.bits == 8:
        return flat.contiguous().view(torch.uint8).reshape(-1), torch.empty(0, dtype=torch.uint8)
    unsigned = flat.to(torch.int16) & 0x0F
    base = (unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)).to(torch.uint8).reshape(-1)
    if spec.bits == 4:
        return base, torch.empty(0, dtype=torch.uint8)
    high_bits = spec.bits - 4
    upper = ((flat.to(torch.int32) & ((1 << spec.bits) - 1)) >> 4) & ((1 << high_bits) - 1)
    shifts = torch.arange(high_bits, dtype=torch.int32)
    bits = ((upper.unsqueeze(-1) >> shifts) & 1).reshape(flat.shape[0], -1, 8)
    weights = 1 << torch.arange(8, dtype=torch.int32)
    return base, (bits * weights).sum(-1).to(torch.uint8).reshape(-1)


def encode(weight, encoding):
    """The row-split-k128-v1 payload of a 2-D weight (any float dtype)."""
    spec = SPECS[encoding]
    g = geometry(encoding, tuple(weight.shape))
    codes, scales = quantize(weight, encoding)
    base, high = _pack(codes, spec)
    payload = bytearray(g.payload_bytes)
    payload[:g.base_bytes] = base.numpy().tobytes()
    payload[g.high_offset:g.high_offset + g.high_bytes] = high.numpy().tobytes()
    payload[g.scale_offset:] = scales.contiguous().view(torch.uint8).numpy().tobytes()
    return bytes(payload)


def decode(payload, encoding, shape):
    """The fp32 [N, K] weight a payload stands for: code * fp16 scale."""
    spec = SPECS[encoding]
    g = geometry(encoding, tuple(shape))
    if len(payload) != g.payload_bytes:
        raise ValueError(f"row-split payload has {len(payload)} bytes, expected {g.payload_bytes}")
    buf = np.frombuffer(payload, dtype=np.uint8)
    scales = torch.from_numpy(buf[g.scale_offset:].copy()).view(torch.float16).reshape(g.n, g.groups_per_row)
    groups = g.n * g.groups_per_row
    base = torch.from_numpy(buf[:g.base_bytes].copy())
    if spec.bits == 8:
        codes = base.view(torch.int8).reshape(groups, spec.group_size).to(torch.int16)
    else:
        packed = base.reshape(groups, spec.group_size // 2).to(torch.int16)
        unsigned = torch.empty((groups, spec.group_size), dtype=torch.int16)
        unsigned[:, 0::2] = packed & 0x0F
        unsigned[:, 1::2] = (packed >> 4) & 0x0F
        if spec.bits > 4:
            high_bits = spec.bits - 4
            high = torch.from_numpy(buf[g.high_offset:g.high_offset + g.high_bytes].copy())
            high = high.reshape(groups, -1).to(torch.int16)
            position = torch.arange(spec.group_size) * high_bits
            upper = (high.index_select(1, position // 8) >> (position % 8)) & ((1 << high_bits) - 1)
            unsigned = unsigned | (upper << 4)
        sign = 1 << (spec.bits - 1)
        codes = torch.where((unsigned & sign) != 0, unsigned - (1 << spec.bits), unsigned)
    w = (codes.float().reshape(g.n, g.groups_per_row, spec.group_size) * scales.float().unsqueeze(-1))
    return w.reshape(g.n, g.k_pad)[:, :g.k]
