"""FP8 E4M3FN weights with one bf16 scale per output row: the container's
`FP8_E4M3FN_ROW_BF16S` / `row-scale-v1` payload (docs/specs/flash-next/layout.md §6.1).

Copied from tools/flash-next-converter/fp8.py (same repository), unchanged but for this header
and `rel_error`: one FP8 encoder for both converters. Keep the two in step.

The payload is the u8 code plane `rows x cols`, zero padding to the next multiple of
256, then the bf16 scale plane `rows`. A row's scale is amax/448 rounded up to the next
bf16, so no code saturates; codes are round-to-nearest-even against that stored scale.
"""
import torch

ALIGN = 256
E4M3_MAX = 448.0


def _bf16_round_up(x):
    """Smallest bf16 >= x (x positive fp32), as a bf16 tensor."""
    b = x.to(torch.bfloat16)
    low = b.float() < x
    bits = b.view(torch.int16)
    bits = torch.where(low, bits + 1, bits)
    return bits.view(torch.bfloat16)


def scales(w):
    amax = w.float().abs().amax(1)
    s = _bf16_round_up(amax / E4M3_MAX)
    return torch.where(amax > 0, s, torch.ones_like(s))


def encode(w):
    """bytes of the row-scale-v1 payload of a 2-D weight (any float dtype, any device)."""
    if w.dim() != 2:
        raise ValueError(f"FP8 row scale needs a 2-D weight, got {tuple(w.shape)}")
    rows, cols = w.shape
    s = scales(w)
    q = (w.float() / s.float()[:, None]).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    codes = q.view(torch.uint8).cpu().contiguous().numpy().tobytes()
    pad = -len(codes) % ALIGN
    return codes + bytes(pad) + s.cpu().contiguous().view(torch.int16).numpy().tobytes()


def decode(payload, shape, device="cpu"):
    """The bf16 weight a payload stands for: e4m3fn(code) * bf16(scale), rounded to bf16."""
    rows, cols = shape
    n = rows * cols
    off = n + (-n % ALIGN)
    buf = bytearray(payload) if not isinstance(payload, bytearray) else payload
    codes = torch.frombuffer(buf, dtype=torch.uint8, count=n).view(torch.float8_e4m3fn).view(rows, cols)
    s = torch.frombuffer(buf, dtype=torch.bfloat16, count=rows, offset=off)
    return (codes.to(device).float() * s.to(device).float()[:, None]).to(torch.bfloat16)


def payload_bytes(shape):
    rows, cols = shape
    n = rows * cols
    return n + (-n % ALIGN) + 2 * rows


def decode_fp32(payload, shape):
    """e4m3fn(code) * bf16(scale) in fp32, not rounded to bf16: what an FP8 GEMM's epilogue
    multiplies (the image converter's round-trip error is measured on this)."""
    rows, cols = shape
    n = rows * cols
    buf = bytearray(payload)
    codes = torch.frombuffer(buf, dtype=torch.uint8, count=n).view(torch.float8_e4m3fn).view(rows, cols)
    s = torch.frombuffer(buf, dtype=torch.bfloat16, count=rows, offset=n + (-n % ALIGN))
    return codes.float() * s.float()[:, None]


def rel_error(w, w_hat, rows_per_block=2048):
    """||w - w_hat||_F / ||w||_F, accumulated in float64 (0 for an all-zero w)."""
    num = den = 0.0
    for a in range(0, w.shape[0], rows_per_block):
        x = w[a:a + rows_per_block].double()
        num += float(((x - w_hat[a:a + rows_per_block].double()) ** 2).sum())
        den += float((x ** 2).sum())
    return (num / den) ** 0.5 if den > 0 else 0.0
