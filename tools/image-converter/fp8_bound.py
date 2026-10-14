"""FP8 round-trip relative error of the 27B's row-scaled FP8 linears (ticket #318).

For each FP8 linear of unsloth/Qwen3.8-27B-NVFP4 (F8_E4M3 `X.weight` + `X.weight_scale`) it
measures, against the BF16 original Qwen/Qwen3.8-27B (same name, resolved by suffix),
  rel = ||W - W_hat||_F / ||W||_F     (float64, W = BF16 upcast)
for (a) the source codes * source scale, (b) the same with the scale rounded to bf16 (only if
the source scale is fp32), (c) W re-encoded by the flash-next-converter `fp8.encode`
(codes * bf16 scale in fp32, no bf16 rounding of the product).
Only the measured tensors' byte ranges are fetched (HTTP range requests, into RAM).

    F:/ai/venvs/ignis-image/Scripts/python.exe tools/image-converter/fp8_bound.py [--lm-head]

CPU and network only. --lm-head adds lm_head (~3.7 GB of traffic); the default stays ~1.5 GB.

Measured 2026-10-13 (54 s): the source scales are BF16; on all ten tensors the source FP8 error
is 2.659-2.663% (max 2.6626e-02) and this converter's encoder 2.643-2.648% (max 2.6475e-02).
convert.py's --fp8-bound default, 3%, is that maximum rounded up (README, "The FP8 bound").
"""
import argparse
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flash-next-converter"))
import fp8  # noqa: E402
import fetch  # noqa: E402

SRC = ("unsloth/Qwen3.8-27B-NVFP4", "7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108")
REF = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
CHUNK = 2048  # rows per float64 block
fetch.DT["F8_E4M3"] = torch.float8_e4m3fn

# (layer, projection) picks: layer 3 full attention, layer 0 GDN, layer 60 MLP (FP8 from 56).
PICKS = [(3, "self_attn.q_proj"), (3, "self_attn.k_proj"), (3, "self_attn.v_proj"), (3, "self_attn.o_proj"),
         (0, "linear_attn.in_proj_qkv"), (0, "linear_attn.in_proj_z"), (0, "linear_attn.out_proj"),
         (60, "mlp.gate_proj"), (60, "mlp.up_proj"), (60, "mlp.down_proj")]


def pick(src):
    out = []
    for layer, proj in PICKS:
        n = f"model.language_model.layers.{layer}.{proj}.weight"
        if n in src.index and src.meta(n)["dtype"] == "F8_E4M3" and n + "_scale" in src.index:
            out.append(n)
        else:
            print(f"  skipped (not FP8 with scale in the source): {n}")
    return out


def resolve(ref, name):
    if name in ref.index:
        return name
    tail = name.split("language_model.", 1)[-1]
    hits = [k for k in ref.index if k.endswith(tail)]
    if len(hits) != 1:
        raise SystemExit(f"cannot resolve {name} in the BF16 repo: {hits}")
    return hits[0]


def measure(codes, scale, w):
    """(source, source-with-bf16-scale, ignis encoder) relative errors, row-chunked float64."""
    den = 0.0
    num = [0.0, 0.0, 0.0]
    for a in range(0, w.shape[0], CHUNK):
        wc = w[a:a + CHUNK]
        sc = scale[a:a + CHUNK].reshape(-1).float()
        cc = codes[a:a + CHUNK].float()
        p = fp8.encode(wc)
        n = wc.numel()
        off = n + (-n % fp8.ALIGN)
        buf = bytearray(p)
        ec = torch.frombuffer(buf, dtype=torch.uint8, count=n).view(torch.float8_e4m3fn).view(wc.shape).float()
        es = torch.frombuffer(buf, dtype=torch.bfloat16, count=wc.shape[0], offset=off).float()
        w64 = wc.double()
        den += float((w64 ** 2).sum())
        hats = [cc * sc[:, None], cc * sc.bfloat16().float()[:, None], ec * es[:, None]]
        for i, h in enumerate(hats):
            num[i] += float(((w64 - h.double()) ** 2).sum())
    return [(x / den) ** 0.5 for x in num]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-head", action="store_true")
    args = ap.parse_args()
    cache = tempfile.mkdtemp(prefix="fp8_bound_")
    src = fetch.Source(os.path.join(cache, "src"), *SRC)
    ref = fetch.Source(os.path.join(cache, "ref"), *REF)
    names = pick(src)
    if args.lm_head and src.meta("lm_head.weight")["dtype"] == "F8_E4M3":
        names.append("lm_head.weight")
    rows = []
    print("tensor (source) -> BF16 tensor")
    for n in names:
        r = resolve(ref, n)
        print(f"  {n} -> {r}")
        sd = src.meta(n + "_scale")["dtype"]
        w = ref.get(r)
        e = measure(src.get(n), src.get(n + "_scale"), w)
        rows.append((n.replace("model.language_model.", ""), tuple(w.shape), sd, e))
        print(f"    done {rows[-1][0]}", flush=True)
    print(f"\n{'tensor':44} {'shape':16} {'scale':5} {'src fp8':>10} {'src bf16s':>10} {'ignis enc':>10}")
    for n, sh, sd, e in rows:
        b = f"{e[1]:.4e}" if sd == "F32" else "n/a"
        print(f"{n:44} {str(sh):16} {sd:5} {e[0]:10.4e} {b:>10} {e[2]:10.4e}")
    print(f"{'max':44} {'':16} {'':5} {max(r[3][0] for r in rows):10.4e} "
          f"{max((r[3][1] for r in rows if r[2] == 'F32'), default=float('nan')):10.4e} "
          f"{max(r[3][2] for r in rows):10.4e}")


if __name__ == "__main__":
    main()
