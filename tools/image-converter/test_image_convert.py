"""The converter on a synthetic checkpoint: the work tree it writes, its record, its refusals."""
import copy
import json
import os

import pytest

import convert
import pin
import synthetic


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    snap, p = synthetic.make(str(root / "checkpoint"))
    record = convert.convert(snap, p, str(root / "work"), status="fixture", log=lambda *a: None)
    return snap, p, str(root / "work"), record


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_every_unit_is_complete_and_lists_its_tensors(converted):
    _, _, work, record = converted
    names = {}
    for unit in convert.UNITS[1:]:
        done = _json(os.path.join(work, unit, "DONE"))["files"]
        tensors = _json(os.path.join(work, unit, "tensors.json"))["tensors"]
        assert set(done) == {"tensors.json"} | {t["file"] for t in tensors}
        for t in tensors:
            assert done[t["file"]]["bytes"] == t["bytes"] == os.path.getsize(os.path.join(work, unit, t["file"]))
            assert t["name"].split("/", 1)[0] == unit
        names[unit] = len(tensors)
    # 2 text layers of 7 linears and 4 norms + the embedding; the vision tower (3 blocks of 12,
    # patch/bias/positions, merger, 3 DeepStack mergers of 6); 2 DiT blocks of 9 + 9 globals; the VAE
    assert names == {"text_encoder": 2 * 11 + 1, "vision": 3 + 3 * 12 + 6 + 3 * 6, "transformer": 2 * 9 + 9,
                     "vae": 238}
    frontend = _json(os.path.join(work, "frontend", "DONE"))["files"]
    assert set(frontend) == set(convert.FRONTEND.values()) | {convert.IMAGE_MODEL}
    assert record["objects"]["not_stored"] == ["text_encoder/lm_head.weight", "text_encoder/model.language_model.norm.weight"]


def test_the_record_holds_every_tensor_with_its_error_and_the_source(converted):
    _, p, work, record = converted
    assert _json(os.path.join(work, "converter.json")) == record
    assert record["status"] == "fixture" and record["source"]["revision"] == synthetic.REVISION
    assert record["source"]["files"] == p["files"]
    assert record["converter"]["commit"] and record["versions"]["diffusers"] == "0.41.0"
    stored = {r["source"]: r for r in record["tensors"]}
    assert len(stored) == len(pin.tensors(p)) - 2
    for r in stored.values():
        if r["encoding"] == "bf16":
            assert r["rel_error"] is None
        else:
            assert 0 < r["rel_error"] < 0.3, r
        assert r["max_abs"] > 0
    fp8 = [r for r in stored.values() if r["encoding"] == "fp8"]
    assert len(fp8) == 2 * 7 + 2 * 7 + 4
    assert max(r["rel_error"] for r in fp8) == record["summary"]["fp8"]["max_rel_error"] < convert.FP8_BOUND
    assert stored["transformer/img_in.weight"]["format"] == "FP8_E4M3FN_ROW_BF16S"
    assert stored["text_encoder/model.visual.patch_embed.proj.weight"]["shape"] == [32, 3 * 2 * 4 * 4]


def test_the_vae_convolutions_are_recorded_with_their_padding(converted):
    _, p, _, record = converted
    convs = {c["object"]: c for c in record["vae_convs"]}
    vae_4d = {f"vae/{n}" for n, (_, _, shape) in p["tensors"]["vae"].items() if len(shape) == 4}
    assert set(convs) == vae_4d - {n for n in vae_4d if ".gamma" in n}
    assert convs["vae/decoder.conv_in.weight"] == {
        "object": "vae/decoder.conv_in.weight", "in_channels": 16, "out_channels": 3 * 8, "kernel": [3, 3],
        "stride": [1, 1], "padding": [1, 1, 1, 1], "upsample_before": None}
    down = convs["vae/encoder.down_blocks.0.downsampler.resample.1.weight"]
    assert (down["stride"], down["padding"]) == ([2, 2], [0, 1, 0, 1])
    up = convs["vae/decoder.up_blocks.0.upsampler.resample.1.weight"]
    assert (up["stride"], up["padding"], up["upsample_before"]) == ([1, 1], [1, 1, 1, 1], "nearest-exact x2")
    assert convs["vae/encoder.down_blocks.1.downsampler.time_conv.weight"]["kernel"] == [1, 1]
    assert convs["vae/quant_conv.weight"]["padding"] == [0, 0, 0, 0]


def test_another_revision_or_tensor_list_is_refused_naming_the_difference(converted, tmp_path):
    snap, p, _, _ = converted
    out = str(tmp_path / "work")
    other = dict(p, revision="f" * 40)
    with pytest.raises(pin.Refused, match=f"revision {synthetic.REVISION} is not the pinned {'f' * 40}"):
        convert.convert(snap, other, out)
    fewer = copy.deepcopy(p)
    del fewer["tensors"]["transformer"]["proj_out.weight"]
    with pytest.raises(pin.Refused, match=r"transformer: tensors not in the pin \(1\): proj_out.weight"):
        convert.convert(snap, fewer, out)
    more = copy.deepcopy(p)
    more["tensors"]["vae"]["decoder.extra.weight"] = ["vae/diffusion_pytorch_model.safetensors", "BF16", [4]]
    with pytest.raises(pin.Refused, match=r"vae: tensors missing \(1\): decoder.extra.weight"):
        convert.convert(snap, more, out)
    reshaped = copy.deepcopy(p)
    reshaped["tensors"]["text_encoder"]["lm_head.weight"][2] = [255, 64]
    with pytest.raises(pin.Refused, match=r"lm_head.weight BF16\[256, 64\] \(pinned BF16\[255, 64\]\)"):
        convert.convert(snap, reshaped, out)
    rehashed = copy.deepcopy(p)
    rehashed["files"]["model_index.json"]["sha256"] = "0" * 64
    with pytest.raises(pin.Refused, match="file model_index.json does not hash to the pinned sha256"):
        convert.convert(snap, rehashed, out)
    assert not os.path.exists(out)


def test_a_nan_an_fp8_error_above_the_bound_and_an_old_work_tree_are_refused(converted, tmp_path):
    snap, p, work, _ = converted
    poisoned, pp = synthetic.make(str(tmp_path / "poisoned"), poison=("transformer", "transformer_blocks.1.attn.to_k.weight"))
    with pytest.raises(pin.Refused, match=r"transformer/transformer_blocks.1.attn.to_k.weight holds a NaN"):
        convert.convert(poisoned, pp, str(tmp_path / "w1"))
    with pytest.raises(pin.Refused, match=r"FP8 round-trip relative error .* above the bound 0.001 \(--fp8-bound\)"):
        convert.convert(snap, p, str(tmp_path / "w2"), fp8_bound=0.001)
    with pytest.raises(pin.Refused, match="is not empty"):
        convert.convert(snap, p, work)
