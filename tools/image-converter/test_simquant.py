"""The simulated W8A8 yardstick's parts (spec 01, AC 6), CPU only: #319's per-token E4M3
activations, the FP8 linear built from a payload, the container read back, the swap of a model's
stored tensors for the artifact's (on the synthetic checkpoint, packed by the Rust packer when it
is built), and thresholds.json's bound."""
import copy
import hashlib
import json
import os
import subprocess

import numpy as np
import pytest
import torch

import convert
import fp8
import record
import rowsplit
import synthetic

F32 = np.float32


def test_fake_quant_rounds_to_nearest_even_and_keeps_subnormals():
    # amax 448: scale and inverse are 1, so the codes are the values themselves
    x = torch.tensor([[448.0, 1.0625, 1.1875, 2.0 ** -9, 1.5 * 2.0 ** -9, 0.5 * 2.0 ** -9, -3.0, 0.0]])
    want = torch.tensor([[448.0, 1.0, 1.25, 2.0 ** -9, 2.0 ** -8, 0.0, -3.0, 0.0]])
    assert torch.equal(record.fake_quant_e4m3(x), want)


def test_fake_quant_is_per_token():
    x = torch.randn(3, 64, generator=torch.Generator().manual_seed(0))
    y = x.clone()
    y[1] *= 2.0 ** 5
    a, b = record.fake_quant_e4m3(x), record.fake_quant_e4m3(y)
    assert torch.equal(a[[0, 2]], b[[0, 2]]) and torch.equal(a[1] * 2.0 ** 5, b[1])


def test_the_scale_and_its_inverse_are_two_correctly_rounded_fp32_divisions():
    rng = np.random.default_rng(1)
    amax = (rng.random(256, dtype=F32) * F32(1000) + F32(1e-3)).astype(F32)
    x = torch.from_numpy(amax[:, None])
    # a row of its amax alone has one code, E4M3(amax * inverse), 448 unless the product rounds below
    codes = (torch.from_numpy(amax) * torch.from_numpy(F32(448) / amax)).clamp(-448, 448).to(torch.float8_e4m3fn)
    want = codes.float().numpy() * (amax / F32(448))
    assert np.array_equal(record.fake_quant_e4m3(x)[:, 0].numpy(), want)


def test_the_amax_element_saturates_to_448_never_nan():
    a = np.arange(1, 200001, dtype=F32) * F32(0.0137)
    over = a[a * (F32(448) / a) > F32(448)]
    assert over.size
    out = record.fake_quant_e4m3(torch.from_numpy(over[:, None]))[:, 0].numpy()
    assert np.isfinite(out).all() and np.array_equal(out, F32(448) * (over / F32(448)))


def test_a_zero_row_and_a_row_under_the_floor_have_zero_codes():
    floor = F32(record.E4M3_MIN_AMAX)
    assert floor == F32(448) * F32(2.0 ** -126)
    below = np.nextafter(floor, F32(0))
    x = torch.from_numpy(np.array([[0, 0], [floor, floor / 2], [below, below / 2]], dtype=F32))
    out = record.fake_quant_e4m3(x)
    assert torch.equal(out[0], torch.zeros(2)) and torch.equal(out[2], torch.zeros(2))
    # at the floor the scale is 2^-126, fp32's smallest normal: quantized as any other row
    assert torch.equal(out[1], x[1])


def test_a_sim_linear_is_its_payloads_weight_on_a_fake_quantized_input():
    g = torch.Generator().manual_seed(2)
    w = (torch.randn(48, 80, generator=g) * 0.02).to(torch.bfloat16)
    payload = fp8.encode(w)
    linear = record.sim_linear(payload, [48, 80], "cpu")
    weight = fp8.decode_fp32(payload, [48, 80])
    assert torch.equal(linear.codes.float() * linear.scales.float()[:, None], weight)
    assert not list(linear.parameters())
    x = torch.randn(2, 5, 80, generator=g)
    assert torch.equal(linear(x), torch.nn.functional.linear(record.fake_quant_e4m3(x), weight))
    linear.quantize_input = False
    assert torch.equal(linear(x), torch.nn.functional.linear(x, weight))


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """The synthetic checkpoint, its work tree and the container the Rust packer makes of it."""
    if not os.path.isfile(convert.PACKER):
        pytest.skip(f"{convert.PACKER} is not built")
    root = tmp_path_factory.mktemp("packed")
    snap, p = synthetic.make(str(root / "checkpoint"))
    work = str(root / "work")
    record_ = convert.convert(snap, p, work, status="fixture", log=lambda *a: None)
    artifact = str(root / "synthetic-v2.ninfer")
    subprocess.run([convert.PACKER, "--family", "qwen-image", "--work", work, "--out", artifact,
                    "--accept-status", "fixture", "--keep-work"], check=True, capture_output=True)
    return snap, work, artifact, record_


def test_the_container_reads_back_every_payload(packed):
    _, work, artifact, record_ = packed
    container = record.Artifact(artifact)
    for r in record_["tensors"]:
        payload = container.payload(r["object"])
        assert hashlib.sha256(payload).hexdigest() == r["sha256"], r["object"]
        with open(os.path.join(work, convert.unit_of(r["object"]), convert.file_name(r["object"])), "rb") as f:
            assert payload == f.read()


def _models(snap):
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration
    return {"text_encoder": Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(snap, "text_encoder"),
                                                                            dtype=torch.bfloat16),
            "transformer": QwenImage21Transformer2DModel.from_pretrained(os.path.join(snap, "transformer"),
                                                                         dtype=torch.bfloat16),
            "vae": AutoencoderKLQwenImage21.from_pretrained(os.path.join(snap, "vae"), dtype=torch.bfloat16)}


def test_simquant_swaps_in_the_artifacts_payloads(packed):
    snap, work, artifact, record_ = packed
    records = {r["source"]: r for r in record_["tensors"]}
    models = _models(snap)
    counts = {comp: record.simquant(m, comp, record.Artifact(artifact), records) for comp, m in models.items()}
    # 2 text layers and 2 DiT blocks of 7 FP8 linears, img_in, txt_in's two, proj_out; the
    # vision tower's 3 blocks of 4 matrices, the patch embedding, the merger and 3 DeepStack mergers
    assert counts["text_encoder"]["fp8"] + counts["transformer"]["fp8"] == 2 * 7 + 2 * 7 + 4
    assert counts["text_encoder"]["w8"] == 1 + 3 * 4 + 2 + 3 * 2
    assert sum(n for c in counts.values() for n in c.values()) == len(records)

    def payload(r):
        with open(os.path.join(work, convert.unit_of(r["object"]), convert.file_name(r["object"])), "rb") as f:
            return f.read()
    modules = {comp: dict(m.named_modules()) for comp, m in models.items()}
    params = {comp: dict(m.named_parameters()) for comp, m in models.items()}
    for r in record_["tensors"]:
        comp, name = r["source"].split("/", 1)
        if r["encoding"] == "fp8":
            linear = modules[comp][name[:-len(".weight")]]
            assert isinstance(linear, record._SimQuantLinear), name
            assert torch.equal(linear.codes.float() * linear.scales.float()[:, None],
                               fp8.decode_fp32(payload(r), r["shape"])), name
        elif r["encoding"] == "w8":
            p = params[comp][name]
            assert torch.equal(p.data, rowsplit.decode(payload(r), "w8", r["shape"]).reshape(p.shape)), name


def test_simquant_refuses_a_tensor_that_is_not_the_artifacts(packed):
    snap, _, artifact, record_ = packed
    for encoding in ("fp8", "w8", "bf16"):
        tampered = copy.deepcopy(record_)
        r = next(r for r in tampered["tensors"] if r["encoding"] == encoding and r["source"].startswith("text_encoder/"))
        r["sha256"] = "0" * 64
        records = {t["source"]: t for t in tampered["tensors"]}
        with pytest.raises(ValueError, match=f"{r['source']}: not the artifact's {r['object']}"):
            record.simquant(_models(snap)["text_encoder"], "text_encoder", record.Artifact(artifact), records)


def test_simquant_refuses_a_record_whose_encoding_is_not_the_plans(packed):
    snap, _, artifact, record_ = packed
    tampered = copy.deepcopy(record_)
    r = next(r for r in tampered["tensors"] if r["encoding"] == "fp8" and r["source"].startswith("text_encoder/"))
    r["encoding"] = "w8"
    records = {t["source"]: t for t in tampered["tensors"]}
    with pytest.raises(ValueError, match=f"{r['source']}: the artifact records w8, recipe.py plans fp8"):
        record.simquant(_models(snap)["text_encoder"], "text_encoder", record.Artifact(artifact), records)


def test_the_bound_is_the_larger_of_both_factors_per_quantity(tmp_path):
    path = str(tmp_path / "thresholds.json")
    # binary fractions, so the products are exact
    sim = {"t2i": {"velocity": [0.25, 0.015625], "encoder": 0.03125, "encoder_row0": 0.5},
           "edit": {"velocity": [0.0625, 0.125], "encoder": 0.5, "encoder_after_vision": 0.25, "encoder_row0": 0.5,
                    "condition_latents": 0.0}}
    bf16 = {"t2i": {"velocity": [0.0625, 0.015625], "encoder": 0.0078125, "encoder_row0": 0.25},
            "edit": {"velocity": [0.03125, 0.0078125], "encoder": 0.015625, "encoder_after_vision": 0.0625,
                     "encoder_row0": 0.25, "condition_latents": 0.015625}}
    artifact = {"file": "a.ninfer", "bytes": 1}
    record.write_bounds(path, sim, bf16, 2.0, 5.0, {"id": "oracle"}, {"id": "yardstick"}, artifact)
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    assert t["schema"] == "ignis-image-thresholds-v4" and t["factors"] == {"sim": 2.0, "bf16": 5.0}
    assert t["oracle_session"] == {"id": "oracle"} and t["session"] == {"id": "yardstick"}
    assert t["artifact"] == artifact and set(t["metrics"]) == {"velocity", "encoder", "encoder_after_vision",
                                                               "encoder_row0", "condition_latents"}
    assert "text rows" in t["metrics"]["encoder"] and set(t["yardsticks"]) == {"bf16", "sim"}
    assert "own encoder output" in t["metrics"]["velocity"]
    assert t["cases"]["t2i"]["velocity"] == {"sim_error": [0.25, 0.015625], "bf16_error": [0.0625, 0.015625],
                                             "bound": [0.5, 0.078125]}
    assert t["cases"]["t2i"]["encoder"]["bound"] == 0.0625
    assert t["cases"]["edit"]["velocity"]["bound"] == [0.15625, 0.25]
    assert t["cases"]["edit"]["encoder"] == {"sim_error": 0.5, "bf16_error": 0.015625, "bound": 1.0}
    assert t["cases"]["edit"]["encoder_after_vision"] == {"sim_error": 0.25, "bf16_error": 0.0625, "bound": 0.5}
    assert t["cases"]["edit"]["condition_latents"]["bound"] == 0.078125
    # the first row is information, never bounded
    assert t["cases"]["edit"]["encoder_row0"] == {"sim_error": 0.5, "bf16_error": 0.25}
    assert "condition_latents" not in t["cases"]["t2i"] and "encoder_after_vision" not in t["cases"]["t2i"]
    with pytest.raises(FileExistsError):
        record.write_bounds(path, sim, bf16, 2.0, 5.0, {"id": "oracle"}, {"id": "yardstick"}, artifact)


def test_simquant_frees_each_replaced_weight_as_it_goes(packed, monkeypatch):
    """On the GPU a model's bf16 linears and their codes do not fit together: each bf16 weight is
    released when its linear is swapped, not when the swap of the whole model ends."""
    import gc
    import weakref

    import recipe
    snap, _, artifact, record_ = packed
    model = _models(snap)["transformer"]
    refs = [weakref.ref(p) for name, p in model.named_parameters()
            if recipe.plan("transformer", name, list(p.shape)).encoding == "fp8"]
    alive = []
    real = record.sim_linear

    def counting(*args):
        gc.collect()
        alive.append(sum(r() is not None for r in refs))
        return real(*args)
    monkeypatch.setattr(record, "sim_linear", counting)
    record.simquant(model, "transformer", record.Artifact(artifact), {r["source"]: r for r in record_["tensors"]})
    assert alive == [len(refs) - k for k in range(len(refs))]
