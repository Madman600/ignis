"""FP8 row scaling: the row-scale-v1 payload and the round-trip error the converter bounds."""
import numpy as np
import torch

import fp8


def test_payload_geometry_is_the_readers():
    # crates/artifact row_scale_geometry_fp8: [256, 512] -> codes 131072, scales at 131072, 131584 bytes
    assert len(fp8.encode(torch.randn(256, 512, dtype=torch.bfloat16))) == 131584
    assert len(fp8.encode(torch.randn(3, 100, dtype=torch.bfloat16))) == 512 + 6   # 300 code bytes padded to 512


def test_every_row_has_its_own_scale_rounded_up_and_no_code_saturates():
    w = torch.tensor([[448.0 * 3.0, -1.0, 0.5], [1e-3, 2e-3, -3e-3]], dtype=torch.bfloat16)
    p = fp8.encode(w)
    assert not np.any((np.frombuffer(p[:6], dtype=np.uint8) & 0x7F) == 0x7F)     # E4M3FN's NaN codes
    s = torch.frombuffer(bytearray(p[256:260]), dtype=torch.bfloat16).float()
    assert s[0] == 3.0 and torch.all(s * 448 >= w.float().abs().amax(1))
    assert s[1] > 0 and s[1] < 1e-5                                               # rows scale apart


def test_an_all_zero_row_has_scale_one_and_zero_codes():
    p = fp8.encode(torch.zeros(2, 32, dtype=torch.bfloat16))
    assert p[:64] == bytes(64)
    assert torch.frombuffer(bytearray(p[256:260]), dtype=torch.bfloat16).tolist() == [1.0, 1.0]


def test_the_round_trip_error_is_the_relative_frobenius_norm():
    w = torch.tensor([[3.0, 4.0]])
    assert fp8.rel_error(w, torch.tensor([[3.0, 4.0]])) == 0.0
    assert abs(fp8.rel_error(w, torch.tensor([[3.0, 3.5]])) - 0.1) < 1e-12     # |(0, .5)| / |(3, 4)|
    assert fp8.rel_error(torch.zeros(2, 2), torch.zeros(2, 2)) == 0.0
    # E4M3's 3-bit mantissa on Gaussian rows: ~2.6% (the 27B's FP8 linears measured 2.64-2.66%,
    # tools/image-converter/fp8_bound.py)
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(512, 1024, generator=g) * 0.02).to(torch.bfloat16)
    err = fp8.rel_error(w.float(), fp8.decode_fp32(fp8.encode(w), w.shape))
    assert 0.024 < err < 0.029
