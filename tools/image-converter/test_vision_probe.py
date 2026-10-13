"""The vision-tower probe (#318 M3) on the synthetic checkpoint: the weights it calls the artifact's
are the converter's payloads, and it measures the three runs."""
import copy
import os

import pytest
import torch

import convert
import rowsplit
import synthetic
import vision_probe


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    snap, p = synthetic.make(str(root / "checkpoint"))
    record = convert.convert(snap, p, str(root / "work"), status="fixture", log=lambda *a: None)
    return snap, str(root / "work"), record


def test_the_artifact_weights_are_the_converters_payloads(converted):
    snap, work, record = converted
    _, state = vision_probe.vision_state(snap)
    weights, checked = vision_probe.artifact_weights(state, record)
    grouped = [r for r in record["tensors"] if r["encoding"] in rowsplit.SPECS]
    assert checked == len(grouped) > 0
    for r in grouped:
        with open(os.path.join(work, "vision", convert.file_name(r["object"])), "rb") as f:
            want = rowsplit.decode(f.read(), r["encoding"], r["shape"])
        name = r["source"][len("text_encoder/model.visual."):]
        assert torch.equal(weights[name], want.reshape(state[name].shape)), name
    untouched = [n for n in state if n not in {r["source"][len("text_encoder/model.visual."):] for r in grouped}]
    assert untouched and all(weights[n] is state[n] for n in untouched)


def test_a_payload_that_is_not_the_artifacts_is_refused(converted):
    snap, _, record = converted
    _, state = vision_probe.vision_state(snap)
    tampered = copy.deepcopy(record)
    for r in tampered["tensors"]:
        if r["encoding"] == "w8":
            r["sha256"] = "0" * 64
            break
    with pytest.raises(ValueError, match="not the artifact's"):
        vision_probe.artifact_weights(state, tampered)


def test_the_probe_measures_the_artifact_and_bf16_runs_against_fp32(converted):
    snap, _, record = converted
    config, state = vision_probe.vision_state(snap)
    side = 4
    patch = config.in_channels * config.temporal_patch_size * config.patch_size ** 2
    pixels = torch.rand(side * side, patch, generator=torch.Generator().manual_seed(0)) * 2 - 1
    result = vision_probe.probe(config, state, pixels.numpy(), [[1, side, side]], record)
    assert result["patches"] == side * side and result["deepstack_layers"] == [0, 1, 2]
    for run in ("artifact", "bf16"):
        assert 0 < result[run]["merger"] < 1 and len(result[run]["deepstack"]) == 3
        assert all(0 < e < 1 for e in result[run]["deepstack"])
