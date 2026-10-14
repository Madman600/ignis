"""The recorder (spec 01, AC 5-6) on a tiny Qwen-Image-2.1 checkpoint: the real processor (the
pinned checkpoint's tokenizer, chat template and image config) with tiny diffusers and
transformers modules, CPU only. The tiny pipeline runs the same code as the real one, so the
precision scheme, every recorded field and every trap are checked here; only the size, the device
and the attention backend wait for the GPU run. The tests that need the pinned checkpoint skip
without it (IGNIS_IMAGE_SNAPSHOT, default the owner's layout)."""
import inspect
import json
import os
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

import record

SNAPSHOT = os.environ.get(
    "IGNIS_IMAGE_SNAPSHOT",
    "F:/ai/models/hf/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba1986fcb905854dc8de2f52eee395f49b5b")

# the real checkpoint's token ids, tiny everything else
TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, vocab_size=151936, rms_norm_eps=1e-6, rope_theta=5000000,
            max_position_embeddings=4096,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True})
VISION = dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=16, temporal_patch_size=2,
              in_channels=3, spatial_merge_size=2, out_hidden_size=64, num_position_embeddings=16,
              deepstack_visual_indexes=[0, 1, 2], hidden_act="gelu_pytorch_tanh")
DIT = dict(num_layers=2, num_attention_heads=4, attention_head_dim=16, mlp_ratio=3, in_channels=16, out_channels=16,
           patch_size=1, axes_dims_rope=[4, 6, 6], context_in_dim=64, causal_condition=True)
VAE = dict(base_dim=2, decoder_base_dim=3, z_dim=16, dim_mult=[1, 2, 4, 8, 8], num_res_blocks=2, in_channels=4,
           out_channels=4, is_residual=True, temperal_downsample=[False, True, True, True],
           latents_mean=[0.1 * i for i in range(16)], latents_std=[1.0 + 0.1 * i for i in range(16)])
T2I = record.Case("t2i", "A red fox in the snow, holding a sign that reads \"ignis\"", 64, seed=1)
BLOCKS = (0, 1)


def _randomize(module, g):
    for p in module.parameters():
        p.data = (torch.randn(p.shape, generator=g) * 0.05).to(torch.bfloat16)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A bf16 checkpoint in the diffusers layout: the pinned processor, scheduler and sigmas, tiny
    seeded-random modules."""
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    if not os.path.isdir(SNAPSHOT):
        pytest.skip(f"{SNAPSHOT} (the pinned checkpoint) is not here")
    out = tmp_path_factory.mktemp("tiny")
    with open(os.path.join(SNAPSHOT, "text_encoder", "config.json"), encoding="utf-8") as f:
        real = json.load(f)
    ids = {k: real[k] for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")}
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    text = Qwen3VLForConditionalGeneration(Qwen3VLConfig(text_config=TEXT, vision_config=VISION,
                                                         tie_word_embeddings=False, **ids))
    dit = QwenImage21Transformer2DModel(**DIT)
    vae = AutoencoderKLQwenImage21(**VAE)
    for name, model in (("text_encoder", text), ("transformer", dit), ("vae", vae)):
        _randomize(model, g)
        model.save_pretrained(str(out / name))
    for sub in ("processor", "scheduler"):
        shutil.copytree(os.path.join(SNAPSHOT, sub), str(out / sub))
    shutil.copy(os.path.join(SNAPSHOT, "model_index.json"), str(out / "model_index.json"))
    return str(out)


@pytest.fixture(scope="module")
def edit(tiny):
    """The edit case at the processor's minimum size, 256x256: the checked-in image's pattern."""
    path = os.path.join(tiny, "edit-condition-256.png")
    record.edit_image(256).save(path)
    return record.Case("edit", "Make the shapes look like glass", 256, seed=2, image=path)


@pytest.fixture(params=["t2i", "edit"])
def case(request, edit):
    return T2I if request.param == "t2i" else edit


def _reference(tiny, dtype):
    """diffusers' own load: what the recorder must compute."""
    from diffusers import QwenImage21Pipeline
    return QwenImage21Pipeline.from_pretrained(tiny, dtype=dtype)


def _latents(pipe, case):
    h, w = case.size // 16, case.size // 16
    noise = torch.from_numpy(record.noise(case, pipe.transformer.config.in_channels))
    image = record.condition_image(case)
    return pipe(prompt=case.prompt, image=image, height=case.size, width=case.size,
                output_resolution=case.size, latents=pipe._pack_latents(noise, 1, noise.shape[1], h, w),
                output_type="latent", use_kv_cache=True).images


def test_fp32_compute_with_bf16_weights_is_diffusers_fp32_bit_for_bit(tiny, case):
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "fp32")
    got = _latents(pipe, case)
    want = _latents(_reference(tiny, torch.float32), case)
    assert got.dtype == torch.float32 and torch.equal(got, want)
    # at rest the repeated blocks hold bf16, as before the run
    assert {p.dtype for p in pipe.transformer.transformer_blocks.parameters()} == {torch.bfloat16}
    assert {p.dtype for p in pipe.text_encoder.model.language_model.layers.parameters()} == {torch.bfloat16}


def test_the_vae_keeps_no_single_frame_cache_and_computes_the_same(tiny):
    from diffusers import AutoencoderKLQwenImage21
    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(tiny, "vae"), dtype=torch.float32)
    retained = []
    vae.decoder.up_blocks[-1].register_forward_hook(
        lambda m, a, o: retained.append(sum(isinstance(x, torch.Tensor) for x in vae._feat_map)))
    image = torch.rand(1, 4, 1, 64, 64, generator=torch.Generator().manual_seed(0)) * 2 - 1

    def round_trip():
        with torch.no_grad():
            z = vae.encode(image).latent_dist.mode()
            return z, vae.decode(z).sample
    z, decoded = round_trip()
    record._no_retain(vae)
    z2, decoded2 = round_trip()
    assert torch.equal(z, z2) and torch.equal(decoded, decoded2)
    assert retained[0] > 0 and retained[1] == 0


def test_bf16_mode_is_a_plain_bf16_load_bit_for_bit(tiny, case):
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "fp32")
    record.set_precision(pipe, "bf16")
    got = _latents(pipe, case)
    want = _latents(_reference(tiny, torch.bfloat16), case)
    assert got.dtype == torch.bfloat16 and torch.equal(got, want)


@pytest.fixture(scope="module")
def recorded(tiny, edit, tmp_path_factory):
    """Both cases recorded by the fp32 pipeline: {case name: (case, its directory, case.json)}."""
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "fp32")
    out = {}
    for case in (T2I, edit):
        directory = str(tmp_path_factory.mktemp(case.name))
        out[case.name] = (case, directory, record.record_case(pipe, case, directory, blocks=BLOCKS))
    return out


def _npy(directory, name):
    return np.load(os.path.join(directory, name + ".npy"))


def test_every_field_is_written_and_indexed(recorded):
    hidden, latent = TEXT["hidden_size"], DIT["in_channels"]
    for name, (case, directory, info) in recorded.items():
        with open(os.path.join(directory, "case.json"), encoding="utf-8") as f:
            assert json.load(f) == info
        for field, entry in info["files"].items():
            a = _npy(directory, field)
            assert list(a.shape) == entry["shape"] and str(a.dtype) == entry["dtype"], field
        n = info["sequence"]["text_encoder"]
        drop = info["drop_idx"]
        side = case.size // 16
        target = side * side
        assert _npy(directory, "input_ids").shape == (1, n)
        assert _npy(directory, "encoder_hidden_states").shape == (1, n, hidden)
        assert _npy(directory, "prompt_embeds").shape == (1, n - drop, hidden)
        assert _npy(directory, "image_pad_mask").shape == (1, n - drop)
        assert _npy(directory, "mrope_position_ids").shape == (3, 1, n)
        assert _npy(directory, "noise").shape == (1, latent, side, side)
        joint = info["sequence"]["dit"]
        assert _npy(directory, "rope").shape == (joint, sum(DIT["axes_dims_rope"]) // 2, 2)
        assert _npy(directory, "image_ids").shape == _npy(directory, "target_token_mask").shape == (joint,)
        assert info["segments"] and info["segments"][-1][1] == joint - target
        steps = len(info["sigmas"]) - 1
        assert steps == 8 and _npy(directory, "timesteps").shape == (steps,)
        for i in range(steps):
            for field in ("latents_in", "velocity", "latents_out"):
                assert _npy(directory, f"step{i}/{field}").shape == (1, target, latent)
        # step 0 runs the whole joint sequence (the prefix is extracted), step 1 the target only
        for b in BLOCKS:
            assert _npy(directory, f"step0/block{b:02d}").shape == (1, joint, DIT["num_attention_heads"] * 16)
            assert _npy(directory, f"step1/block{b:02d}").shape == (1, target, DIT["num_attention_heads"] * 16)
        assert info["kv_cache_modes"] == ["extract"] + ["cached"] * (steps - 1)
        from PIL import Image
        with Image.open(os.path.join(directory, "image.png")) as img:
            assert img.mode == "RGBA" and img.size == (case.size, case.size)
        if case.image:
            condition = side * side  # the condition image is the target's size here
            assert _npy(directory, "condition_latents").shape == (1, condition, latent)
            assert _npy(directory, "condition_vae_input").shape == (1, 4, 1, case.size, case.size)
            grid = _npy(directory, "image_grid_thw")
            assert grid.tolist() == [[1, case.size // 16, case.size // 16]]
            assert _npy(directory, "pixel_values").shape == (case.size // 16 * case.size // 16, 3 * 2 * 16 * 16)
            assert joint == n - drop + 3 * condition // 4 + target


def test_each_step_is_the_euler_update_of_its_velocity(recorded):
    """Spec 04's sampler: sigmas are the checkpoint's sample_sigmas then 0, x += (s_next - s) * v."""
    with open(os.path.join(SNAPSHOT, "model_index.json"), encoding="utf-8") as f:
        sigmas = json.load(f)["sample_sigmas"] + [0.0]
    for case, directory, info in recorded.values():
        assert np.array_equal(_npy(directory, "sigmas"), np.array(sigmas, dtype=np.float32))
        assert np.array_equal(_npy(directory, "timesteps"), np.array(sigmas[:-1], dtype=np.float32) * 1000)
        side = case.size // 16
        noise = _npy(directory, "noise")
        assert np.array_equal(_npy(directory, "step0/latents_in")[0], noise[0].reshape(noise.shape[1], -1).T)
        for i in range(8):
            x, v = _npy(directory, f"step{i}/latents_in"), _npy(directory, f"step{i}/velocity")
            want = x + (np.float32(sigmas[i + 1]) - np.float32(sigmas[i])) * v
            np.testing.assert_allclose(_npy(directory, f"step{i}/latents_out"), want, rtol=0, atol=1e-6)
            if i:
                assert np.array_equal(x, _npy(directory, f"step{i - 1}/latents_out"))


def test_the_recording_is_diffusers_fp32(tiny, recorded):
    reference = _reference(tiny, torch.float32)
    for case, directory, info in recorded.values():
        assert np.array_equal(_latents(reference, case).numpy(), _npy(directory, "step7/latents_out"))
        with torch.no_grad():
            embeds, _, pad = reference.encode_prompt(case.prompt, image=None if case.image is None else [
                reference.image_processor.resize(record.condition_image(case), width=case.size, height=case.size)])
        assert np.array_equal(embeds.numpy(), _npy(directory, "prompt_embeds"))
        assert np.array_equal(pad.numpy(), _npy(directory, "image_pad_mask").astype(bool))
        # the decode phase is the pipeline's own decode
        h = case.size // 16
        noise = torch.from_numpy(record.noise(case, reference.transformer.config.in_channels))
        image = reference(prompt=case.prompt, image=record.condition_image(case), height=case.size, width=case.size,
                          output_resolution=case.size, latents=reference._pack_latents(noise, 1, noise.shape[1], h, h),
                          output_type="pil", use_kv_cache=True).images[0]
        with Image.open(os.path.join(directory, "image.png")) as recorded_image:
            assert np.array_equal(np.asarray(recorded_image), np.asarray(image))


# The traps spec 01 lists, each pinned on the recording.

def test_trap_the_prompt_is_the_raw_template_not_apply_chat_template(tiny, recorded):
    """The two differ where an image is: the raw template names it `<image1>` before
    `<|vision_start|>`, the chat template does not. Without an image they coincide."""
    import metadata
    from PIL import Image
    from transformers import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(tiny, "processor"))
    case, directory, _ = recorded["t2i"]
    assert _npy(directory, "input_ids")[0].tolist() == processor.tokenizer(
        metadata.TEMPLATE_T2I.format(case.prompt))["input_ids"]
    case, directory, _ = recorded["edit"]
    rgba = record.condition_image(case)
    on_white = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    ids = _npy(directory, "input_ids")[0].tolist()
    raw = processor(text=[metadata.TEMPLATE_TI2I.format(case.prompt)], images=[on_white])["input_ids"][0]
    chat = processor.apply_chat_template(
        [{"role": "system", "content": [{"type": "text", "text": metadata.SYSTEM_PROMPT}]},
         {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": case.prompt}]}],
        tokenize=False, add_generation_prompt=True)
    assert ids == list(raw)
    assert ids != list(processor(text=[chat], images=[on_white])["input_ids"][0])


def test_trap_drop_idx_is_the_chat_templated_system_message(tiny, recorded):
    import metadata
    from transformers import Qwen3VLProcessor
    tokenizer = Qwen3VLProcessor.from_pretrained(os.path.join(tiny, "processor")).tokenizer
    for case, directory, info in recorded.values():
        ids = _npy(directory, "input_ids")[0].tolist()
        drop = info["drop_idx"]
        assert tokenizer.decode(ids[:drop]) == f"<|im_start|>system\n{metadata.SYSTEM_PROMPT}<|im_end|>\n"
        assert tokenizer.decode(ids[drop:drop + 2]) == "<|im_start|>user"


def test_trap_an_empty_prompt_becomes_a_space(tiny, tmp_path):
    import metadata
    from transformers import Qwen3VLProcessor
    tokenizer = Qwen3VLProcessor.from_pretrained(os.path.join(tiny, "processor")).tokenizer
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "fp32")
    record.record_case(pipe, record.Case("empty", "", 64, seed=3), str(tmp_path), blocks=BLOCKS)
    ids = _npy(str(tmp_path), "input_ids")[0].tolist()
    assert ids == tokenizer(metadata.TEMPLATE_T2I.format(" "))["input_ids"]
    assert ids != tokenizer(metadata.TEMPLATE_T2I.format(""))["input_ids"]


def test_trap_the_encoder_output_is_taken_before_the_final_norm(tiny, recorded):
    norm = _reference(tiny, torch.float32).text_encoder.model.language_model.norm
    for case, directory, info in recorded.values():
        hidden = _npy(directory, "encoder_hidden_states")
        assert np.array_equal(_npy(directory, "prompt_embeds"), hidden[:, info["drop_idx"]:])
        with torch.no_grad():
            normed = norm(torch.from_numpy(hidden)).numpy()
        assert not np.allclose(normed, hidden, atol=1e-3)


def test_trap_the_encoder_sees_the_image_on_white_and_the_vae_reads_rgba(recorded):
    from PIL import Image
    from transformers import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(SNAPSHOT, "processor")).image_processor
    case, directory, _ = recorded["edit"]
    rgba = record.condition_image(case)
    alpha = np.asarray(rgba)[..., 3]
    assert {0, 255} <= set(np.unique(alpha).tolist()) and ((alpha > 0) & (alpha < 255)).any()
    on_white = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    dropped = rgba.convert("RGB")
    pixels = _npy(directory, "pixel_values")
    # one 8-bit step is 2/255 after the processor's (x/255 - 0.5)/0.5
    assert np.abs(pixels - processor(images=[on_white], return_tensors="np")["pixel_values"]).max() <= 2 / 255 + 1e-6
    assert np.abs(pixels - processor(images=[dropped], return_tensors="np")["pixel_values"]).max() > 1
    vae = _npy(directory, "condition_vae_input")[0, :, 0]
    np.testing.assert_allclose(vae[3], alpha / 127.5 - 1, atol=1e-6)
    hidden_rgb = np.asarray(rgba)[alpha == 0][:, :3]
    np.testing.assert_allclose(vae[:3].transpose(1, 2, 0)[alpha == 0], hidden_rgb / 127.5 - 1, atol=1e-6)
    assert (hidden_rgb != 255).any()


def test_a_run_that_breaks_a_trap_is_refused(tiny, tmp_path):
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "fp32")
    pipe._drop_idx += 1
    with pytest.raises(AssertionError, match="drop_idx"):
        record.record_case(pipe, T2I, str(tmp_path), blocks=BLOCKS)


# The bf16 yardstick (AC 6).

def _copy_case(directory, tmp_path):
    copy = str(tmp_path / "case")
    shutil.copytree(directory, copy)
    return copy


def test_the_yardstick_is_teacher_forced_step_by_step(tiny, recorded, tmp_path):
    """Fed each step's recorded input, the fp32 pipeline reproduces each recorded velocity: a
    perturbed input at step 3 changes step 3's velocity only."""
    case, directory, _ = recorded["t2i"]
    copy = _copy_case(directory, tmp_path)
    x = _npy(copy, "step3/latents_in")
    np.save(os.path.join(copy, "step3", "latents_in.npy"), x * np.float32(1.25))
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "fp32")
    errors = record.yardstick(pipe, case, copy)
    assert errors["encoder"] == 0
    assert [e == 0 for e in errors["velocity"]] == [i != 3 for i in range(8)]


def test_the_bf16_yardstick_measures_each_step_against_fp32(tiny, recorded):
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "bf16")
    for case, directory, _ in recorded.values():
        errors = record.yardstick(pipe, case, directory)
        assert len(errors["velocity"]) == 8 and all(0 < e < 0.1 for e in errors["velocity"]), errors
        assert 0 < errors["encoder"] < 0.1
        assert (errors.get("condition_latents") is not None) == (case.image is not None)


def test_thresholds_hold_5x_each_step_and_are_written_once(tmp_path):
    path = str(tmp_path / "thresholds.json")
    # binary fractions, so 5x is exact
    errors = {"t2i": {"velocity": [0.015625, 0.03125], "encoder": 0.00390625},
              "edit": {"velocity": [0.0625, 0.125], "encoder": 0.0078125, "condition_latents": 0.001953125}}
    record.write_thresholds(path, errors, factor=5.0, session={"id": "s"})
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    assert t["factor"] == 5.0 and t["session"] == {"id": "s"}
    assert t["cases"]["t2i"]["velocity"] == {"bf16_error": [0.015625, 0.03125], "threshold": [0.078125, 0.15625]}
    assert t["cases"]["edit"]["encoder"] == {"bf16_error": 0.0078125, "threshold": 0.0390625}
    assert t["cases"]["edit"]["condition_latents"] == {"bf16_error": 0.001953125}
    with pytest.raises(FileExistsError):
        record.write_thresholds(path, errors, factor=5.0, session={"id": "s"})


# The GPU run's guards, checked where they can be on the CPU.

def test_fp32_ieee_turns_tf32_off_for_matmul_and_cudnn():
    torch.backends.cudnn.allow_tf32 = True  # torch's default: cuDNN convolutions in TF32
    with pytest.raises(AssertionError, match="TF32"):
        record.check_fp32_ieee()
    flags = record.fp32_ieee()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert "tf32" not in (flags["cudnn.conv.fp32_precision"], flags["matmul.fp32_precision"])
    assert flags["matmul.allow_tf32"] is False and flags["cudnn.allow_tf32"] is False


def test_without_a_visible_gpu_the_cudnn_precisions_are_not_read(monkeypatch):
    class Unreadable:
        @property
        def fp32_precision(self):
            raise AssertionError("cuDNN read without a GPU")
    record.fp32_ieee()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.backends.cudnn, "conv", Unreadable())
    monkeypatch.setattr(torch.backends.cudnn, "rnn", Unreadable())
    flags = record.check_fp32_ieee()
    assert flags["cudnn.conv.fp32_precision"] is None and flags["cudnn.rnn.fp32_precision"] is None
    assert flags["cudnn.allow_tf32"] is False

    def unreadable():
        raise AssertionError("cuDNN version read without a GPU")
    monkeypatch.setattr(torch.backends.cudnn, "version", unreadable)
    assert record._versions()["cudnn"] is None


def test_the_dit_guard_changes_nothing(tiny, recorded, tmp_path):
    """The attention guard around the DiT's forward (on the GPU, memory-efficient SDPA only), with
    a guard the CPU allows."""
    import contextlib
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "fp32")
    entered = []

    def guard():
        entered.append(1)
        return contextlib.nullcontext()
    record._guard_dit(pipe, True, guard)
    case, directory, info = recorded["t2i"]
    assert record.record_case(pipe, case, str(tmp_path), blocks=BLOCKS) == info
    assert np.array_equal(_npy(str(tmp_path), "step7/latents_out"), _npy(directory, "step7/latents_out"))
    assert len(entered) == 8
    record._guard_dit(pipe, False)
    assert "forward" not in pipe.transformer.__dict__


def test_the_attention_guard_refuses_a_fallback_to_math():
    q = torch.randn(1, 2, 8, 16)
    with record.efficient_attention_only():
        with pytest.raises(RuntimeError):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)  # no memory-efficient kernel on the CPU
    torch.nn.functional.scaled_dot_product_attention(q, q, q)


# The real run's cases and the whole run.

def test_the_three_cases_are_the_specs():
    t2i, t2i_2k, edit = (record.CASES[n] for n in ("t2i", "t2i-2k", "edit"))
    assert (t2i.size, t2i_2k.size, edit.size) == (1024, 2048, 1024)
    assert t2i.prompt == t2i_2k.prompt and t2i.image is None and t2i_2k.image is None
    assert edit.image == os.path.join(record.HERE, "fixtures", "edit-condition.png")
    assert len({c.seed for c in record.CASES.values()}) == 3


def test_the_checked_in_condition_image_is_the_procedural_one():
    with Image.open(record.CASES["edit"].image) as checked_in:
        assert checked_in.mode == "RGBA" and checked_in.size == (1024, 1024)
        pixels = np.asarray(checked_in)
    assert np.array_equal(pixels, np.asarray(record.edit_image(1024)))
    alpha = pixels[..., 3]
    assert (alpha == 0).any() and (alpha == 255).any() and ((alpha > 0) & (alpha < 255)).any()
    assert os.path.getsize(record.CASES["edit"].image) < 32 * 1024


def _run(tiny, edit, out):
    return record.run(tiny, out, "cpu", [T2I, edit], blocks=BLOCKS, factor=5.0, min_free_disk_gb=0,
                      disk_budget_gb=0, min_free_ram_gb=0, min_free_vram_gb=0, max_vram_gib=0)


def test_a_run_records_every_case_then_the_yardstick_in_one_session(tiny, edit, recorded, tmp_path):
    out = str(tmp_path / "oracle")
    manifest = _run(tiny, edit, out)
    with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
        assert json.load(f) == manifest
    with open(os.path.join(out, "thresholds.json"), encoding="utf-8") as f:
        thresholds = json.load(f)
    assert thresholds["session"] == manifest["session"] and set(thresholds["cases"]) == {"t2i", "edit"}
    fresh = record.load_pipeline(tiny, "cpu")
    record.set_precision(fresh, "bf16")
    for name in ("t2i", "edit"):
        entry = manifest["cases"][name]
        for phase in ("fp32_encoder", "bf16_encoder", "fp32_dit", "bf16_dit", "fp32_decode"):
            assert entry[phase + "_wall_s"] >= 0
        # the phased run (all prompts, the encoder dropped, then all DiT work) records what one
        # pipeline call per case records, bit for bit
        case, directory, info = recorded[name]
        with open(os.path.join(out, name, "case.json"), encoding="utf-8") as f:
            assert json.load(f) == info
        for field in info["files"]:
            assert np.array_equal(_npy(os.path.join(out, name), field), _npy(directory, field)), field
        with Image.open(os.path.join(out, name, "image.png")) as a, Image.open(os.path.join(directory, "image.png")) as b:
            assert np.array_equal(np.asarray(a), np.asarray(b))
        errors = record.yardstick(fresh, case, directory)
        assert thresholds["cases"][name]["velocity"]["bf16_error"] == errors["velocity"]
        assert thresholds["cases"][name]["encoder"]["bf16_error"] == errors["encoder"]
        assert len(thresholds["cases"][name]["velocity"]["threshold"]) == 8
    assert os.path.isfile(os.path.join(out, "edit", "condition.png"))
    assert manifest["use_kv_cache"] is True and manifest["tf32"]["cudnn.allow_tf32"] is False
    for lib in ("torch", "diffusers", "transformers"):
        assert manifest["versions"][lib]
    assert manifest["checkpoint"]["revision"] == os.path.basename(tiny)
    with pytest.raises(FileExistsError):
        _run(tiny, edit, out)


def test_the_preflight_refuses_a_full_disk_low_ram_and_an_old_recording(tmp_path):
    cpu, out = torch.device("cpu"), str(tmp_path / "oracle")
    with pytest.raises(RuntimeError, match="GB free on .* disk"):
        record._preflight(out, cpu, min_free_disk_gb=1e9, disk_budget_gb=0, min_free_ram_gb=0, min_free_vram_gb=0)
    with pytest.raises(RuntimeError, match="GB of host RAM free, below"):
        record._preflight(out, cpu, min_free_disk_gb=0, disk_budget_gb=0, min_free_ram_gb=1e9, min_free_vram_gb=0)
    measured = record._preflight(out, cpu, 0, 0, 0, 0)
    assert measured["free_disk_gb"] > 0 and measured["free_ram_gb"] > 0 and measured["free_vram_gb"] is None
    open(os.path.join(out, "run.log"), "w").close()
    with pytest.raises(FileExistsError, match="recorded once"):
        record._preflight(out, cpu, 0, 0, 0, 0)


def _main(monkeypatch, argv):
    """record.main(argv) with `run` and `run_yardstick` replaced: (positional args, keyword args)."""
    calls = []
    monkeypatch.setattr(record, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(record, "run_yardstick", lambda *a, **k: calls.append((a, k)))
    assert record.main(argv) == 0
    (args, kwargs), = calls
    kwargs.pop("log")
    return args, kwargs


def test_main_hands_every_flag_to_its_function_and_shares_its_defaults(monkeypatch):
    for fn, argv in ((record.run, ["run"]), (record.run_yardstick, ["yardstick"])):
        defaults = {k: p.default for k, p in inspect.signature(fn).parameters.items() if k != "log"
                    and p.default is not inspect.Parameter.empty}
        args, kwargs = _main(monkeypatch, argv + ["--checkpoint", "C"])
        assert kwargs == defaults
    args, kwargs = _main(monkeypatch, [
        "run", "--checkpoint", "C", "--out", "O", "--device", "cpu", "--cases", "t2i,edit", "--blocks", "1,2",
        "--block-steps", "0", "--factor", "3", "--min-free-disk-gb", "1", "--disk-budget-gb", "2",
        "--min-free-ram-gb", "3", "--min-free-vram-gb", "4", "--max-vram-gib", "5", "--host-decode-from", "0"])
    assert args == ("C", "O", "cpu", [record.CASES["t2i"], record.CASES["edit"]])
    assert kwargs == dict(blocks=(1, 2), block_steps=(0,), factor=3.0, min_free_disk_gb=1.0, disk_budget_gb=2.0,
                          min_free_ram_gb=3.0, min_free_vram_gb=4.0, max_vram_gib=5.0, host_decode_from=0)
    args, kwargs = _main(monkeypatch, [
        "yardstick", "--checkpoint", "C", "--oracle", "O", "--artifact", "A", "--device", "cpu", "--cases", "edit",
        "--sim-factor", "3", "--bf16-factor", "4", "--replace-thresholds", "--supersede-bound", "--report", "R",
        "--top-rows", "2", "--min-free-ram-gb", "1", "--min-free-vram-gb", "2", "--max-vram-gib", "3"])
    assert args == ("C", "O", "A", "cpu", [record.CASES["edit"]])
    assert kwargs == dict(sim_factor=3.0, bf16_factor=4.0, replace=True, supersede=True, report="R", top=2,
                          min_free_ram_gb=1.0, min_free_vram_gb=2.0, max_vram_gib=3.0)


# The encoder metric, DeepStack off and the simulated W8A8 yardstick (AC 6).

def _scaled(embeds, rows):
    """prompt_embeds with `rows` 1.5 times their value: each of them off by half its own norm."""
    out = embeds.astype(np.float64)
    out[0, rows] *= 1.5
    return out


def test_the_encoder_metric_is_the_rms_of_each_text_rows_relative_error(recorded):
    case, directory, _ = recorded["t2i"]
    embeds = _npy(directory, "prompt_embeds")
    assert not _npy(directory, "image_pad_mask").any()
    n = embeds.shape[1]
    # one row off by half its norm, out of the n - 1 rows from row 1 on: 0.5 / sqrt(n - 1), whatever
    # that row's norm; row 0 is information only and moves no bounded quantity
    for row in (1, n - 1):
        errors = record.encoder_errors(_scaled(embeds, [row]), directory)
        assert errors["encoder"] == pytest.approx(0.5 / np.sqrt(n - 1), rel=1e-12)
        assert errors["encoder_row0"] == 0
        assert "encoder_after_vision" not in errors
    errors = record.encoder_errors(_scaled(embeds, [0]), directory)
    assert errors["encoder"] == 0 and errors["encoder_row0"] == pytest.approx(0.5, abs=1e-15)
    assert record.rel_l2(_scaled(embeds, [1]), embeds) != pytest.approx(0.5 / np.sqrt(n - 1))


def test_the_encoder_metric_refuses_an_empty_row_set_and_a_zero_reference_row():
    ref = np.ones((3, 4))
    with pytest.raises(ValueError, match="nonzero reference norm"):
        record.row_rel_l2(ref[:0], ref[:0])
    ref[1] = 0
    with pytest.raises(ValueError, match="nonzero reference norm"):
        record.row_rel_l2(ref + 1, ref)


def test_the_rows_after_the_first_image_pad_include_the_text_between_images():
    # <im_start> text <pad> <pad> text <pad> text text: two images, the text between them is reached
    # by the first image's vision features
    pad = np.array([[0, 0, 1, 1, 0, 1, 0, 0]])
    rows = record.encoder_rows(pad)
    assert np.flatnonzero(rows["encoder"]).tolist() == [1, 4, 6, 7]
    assert np.flatnonzero(rows["encoder_row0"]).tolist() == [0]
    assert np.flatnonzero(rows["encoder_after_vision"]).tolist() == [4, 6, 7]
    assert "encoder_after_vision" not in record.encoder_rows(np.zeros((1, 5)))


def test_the_rows_after_the_vision_span_are_the_ones_the_image_reaches(recorded):
    case, directory, _ = recorded["edit"]
    embeds, pad = _npy(directory, "prompt_embeds"), _npy(directory, "image_pad_mask")[0].astype(bool)
    text = np.flatnonzero(~pad)
    before, after = text[text < np.flatnonzero(pad)[0]], text[text > np.flatnonzero(pad)[0]]
    assert before[0] == 0 and before.size > 1 and after.size
    assert record.encoder_errors(_scaled(embeds, np.flatnonzero(pad)), directory) == {
        "encoder": 0, "encoder_row0": 0, "encoder_after_vision": 0}
    errors = record.encoder_errors(_scaled(embeds, [before[-1]]), directory)
    assert errors["encoder"] == pytest.approx(0.5 / np.sqrt(text.size - 1), rel=1e-12)
    assert errors["encoder_after_vision"] == 0 and errors["encoder_row0"] == 0
    errors = record.encoder_errors(_scaled(embeds, [after[0]]), directory)
    assert errors["encoder"] == pytest.approx(0.5 / np.sqrt(text.size - 1), rel=1e-12)
    assert errors["encoder_after_vision"] == pytest.approx(0.5 / np.sqrt(after.size), rel=1e-12)


def test_deepstack_off_moves_the_edit_cases_rows_after_the_image_only_while_on(tiny, recorded):
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "fp32")
    case, directory, _ = recorded["edit"]
    with record.deepstack_off(pipe):
        off = record.encode(pipe, case)
        t2i, t2i_directory, _ = recorded["t2i"]
        assert record.encoder_errors(record._numpy(record.encode(pipe, t2i).embeds), t2i_directory)["encoder"] == 0
    errors = record.encoder_errors(record._numpy(off.embeds), directory)
    assert errors["encoder_after_vision"] > 0
    # it cannot reach the rows before the image
    assert errors["encoder_row0"] == 0 and errors["encoder"] < errors["encoder_after_vision"]
    assert record.encoder_errors(record._numpy(record.encode(pipe, case).embeds), directory)["encoder"] == 0


@pytest.fixture(scope="module")
def tiny_artifact(tiny, edit, tmp_path_factory):
    """The tiny checkpoint converted and packed by the Rust packer: (artifact, {source: record})."""
    import subprocess

    import convert
    import pin
    if not os.path.isfile(convert.PACKER):
        pytest.skip(f"{convert.PACKER} is not built")
    root = tmp_path_factory.mktemp("tiny-artifact")
    paths = sorted(os.path.relpath(os.path.join(d, f), tiny).replace(os.sep, "/")
                   for d, _, files in os.walk(tiny) for f in files)
    p = pin.from_checkpoint(tiny, "tiny/qwen-image-2.1", os.path.basename(tiny), paths)
    work = str(root / "work")
    converted = convert.convert(tiny, p, work, status="fixture", log=lambda *a: None)
    artifact = str(root / "tiny-v2.ninfer")
    subprocess.run([convert.PACKER, "--family", "qwen-image", "--work", work, "--out", artifact,
                    "--accept-status", "fixture"], check=True, capture_output=True)
    return artifact, {r["source"]: r for r in converted["tensors"]}


def _sim(tiny, tiny_artifact, quantize_input=True):
    artifact, records = tiny_artifact
    pipe = record.load_pipeline(tiny, "cpu")
    record.set_precision(pipe, "fp32")
    for comp in ("text_encoder", "transformer", "vae"):
        record.simquant(getattr(pipe, comp), comp, record.Artifact(artifact), records)
    for m in pipe.text_encoder.modules():
        if isinstance(m, record._SimQuantLinear):
            m.quantize_input = quantize_input
    for m in pipe.transformer.modules():
        if isinstance(m, record._SimQuantLinear):
            m.quantize_input = quantize_input
    return pipe


def test_the_sim_pipeline_without_activation_quantization_is_fp32_on_the_artifacts_weights(tiny, tiny_artifact, case):
    """The swap changes nothing but the weights: with the inputs left unquantized, the sim pipeline is
    the fp32 pipeline whose FP8 and W8 weights are what their payloads decode to, bit for bit."""
    import convert
    import fp8
    import recipe
    import rowsplit
    artifact, records = tiny_artifact
    container = record.Artifact(artifact)
    want = record.load_pipeline(tiny, "cpu")
    record.set_precision(want, "fp32")
    for comp in ("text_encoder", "transformer"):
        for name, p in getattr(want, comp).named_parameters():
            item = recipe.plan(comp, name, list(p.shape))
            if item is not None and item.encoding == "fp8":
                p.data = fp8.decode_fp32(container.payload(item.object), item.shape)
            elif item is not None and item.encoding == "w8":
                p.data = rowsplit.decode(container.payload(item.object), "w8", item.shape).reshape(p.shape)
    got = _sim(tiny, tiny_artifact, quantize_input=False)
    assert torch.equal(_latents(got, case), _latents(want, case))


def test_the_sim_yardstick_measures_the_quantization(tiny, recorded, tiny_artifact):
    pipe = _sim(tiny, tiny_artifact)
    for case, directory, _ in recorded.values():
        errors = record.yardstick(pipe, case, directory)
        assert all(0 < e < 0.5 for e in errors["velocity"]) and 0 < errors["encoder"] < 0.5, errors
        # the VAE is the artifact's bf16, computing in fp32 as the oracle does
        assert errors.get("condition_latents", 0) == 0


def test_a_yardstick_session_supersedes_the_thresholds_with_the_bound(tiny, edit, recorded, tiny_artifact, tmp_path):
    import pin
    artifact, records = tiny_artifact
    out = str(tmp_path / "oracle")
    manifest = _run(tiny, edit, out)
    thresholds = os.path.join(out, "thresholds.json")
    with open(thresholds, encoding="utf-8") as f:
        bf16_only = json.load(f)

    def session(replace, supersede=False):
        return record.run_yardstick(tiny, out, artifact, "cpu", [T2I, edit], replace=replace, supersede=supersede,
                                    top=3, min_free_ram_gb=0, min_free_vram_gb=0, max_vram_gib=0, log=lambda *a: None)
    with pytest.raises(FileExistsError, match="--replace-thresholds"):
        session(False)
    value, report = session(True)
    kept = os.path.join(out, f"thresholds.{bf16_only['session']['id'].replace(':', '')}.json")
    with open(kept, encoding="utf-8") as f:
        assert json.load(f) == bf16_only
    with open(thresholds, encoding="utf-8") as f:
        assert json.load(f) == value
    archived = f"yardstick-report.{value['session']['id'].replace(':', '')}.json"
    for name in ("yardstick-report.json", archived):
        with open(os.path.join(out, name), encoding="utf-8") as f:
            assert json.load(f) == json.loads(json.dumps(report))
    # the new file was written under a temporary name, then renamed; the superseded one was copied
    assert sorted(os.listdir(out)) == sorted(["edit", "t2i", "manifest.json", "thresholds.json", os.path.basename(kept),
                                              "yardstick-report.json", archived])
    assert value["schema"] == "ignis-image-thresholds-v4" and report["replaced"] == bf16_only["schema"]
    assert value["oracle_session"] == manifest["session"] and value["session"]["id"]
    assert value["factors"] == {"sim": 2.0, "bf16": 5.0} and value["artifact"]["tensors_checked"] == len(records)
    assert value["artifact"]["sha256"] == pin.sha256_file(artifact)
    sim = _sim(tiny, tiny_artifact)
    for case, directory, _ in recorded.values():
        t = value["cases"][case.name]
        assert t["velocity"]["bf16_error"] == bf16_only["cases"][case.name]["velocity"]["bf16_error"]
        assert t["velocity"]["sim_error"] == record.yardstick(sim, case, directory)["velocity"]
        assert ("encoder_after_vision" in t) == ("condition_latents" in t) == (case.image is not None)
        for quantity in ("velocity", *record.BOUNDED):
            if quantity in t:
                s, b = np.atleast_1d(t[quantity]["sim_error"]), np.atleast_1d(t[quantity]["bf16_error"])
                assert np.array_equal(np.atleast_1d(t[quantity]["bound"]), np.maximum(2 * s, 5 * b))
        assert set(t["encoder_row0"]) == {"sim_error", "bf16_error"}
        checks = report["checks"][case.name]
        assert checks["fp32_replay_encoder"] == 0 and checks["velocity_vs_replaced"] == {"bf16": 0}
        breakdown = report["encoder_breakdown"][case.name]["sim"]
        assert breakdown["row_rel_l2"] == pytest.approx(t["encoder"]["sim_error"], rel=1e-12)
        assert breakdown["text_row_rel_l2"][0] == [0, pytest.approx(t["encoder_row0"]["sim_error"], rel=1e-12)]
        assert len(breakdown["top_rows_by_ref_norm"]) == 3
    off = report["checks"]["edit"]["deepstack_off"]
    assert off["encoder_after_vision"]["error"] > 0 and off["encoder_row0"]["error"] == 0
    for quantity in ("encoder", "encoder_after_vision"):
        assert off[quantity]["bound"] == value["cases"]["edit"][quantity]["bound"]
        assert off[quantity]["fails_the_bound"] == (off[quantity]["error"] > off[quantity]["bound"])
    assert set(report["encoder_breakdown"]["edit"]) == {"sim", "bf16", "deepstack_off"}
    assert set(report["encoder_breakdown"]["t2i"]) == {"sim", "bf16"}
    assert report["encoder_breakdown"]["edit"]["bf16"]["image_pad_rows"]["rows"] > 0

    # a bound is superseded only with --supersede-bound as well, and never with other velocity errors
    with pytest.raises(FileExistsError, match="--supersede-bound"):
        session(True)
    moved = json.loads(json.dumps(value))
    moved["cases"]["t2i"]["velocity"]["sim_error"][0] += 1e-3
    with open(thresholds, "w", encoding="utf-8") as f:
        json.dump(moved, f)
    with pytest.raises(AssertionError, match="errors moved"):
        session(True, supersede=True)
    with open(thresholds, encoding="utf-8") as f:
        assert json.load(f) == moved
    with open(os.path.join(out, "yardstick-report.json"), encoding="utf-8") as f:
        assert json.load(f) == json.loads(json.dumps(report))   # the report of a refused session is not "the" report
    refused = max(name for name in os.listdir(out) if name.startswith("yardstick-report.2"))
    with open(os.path.join(out, refused), encoding="utf-8") as f:
        assert json.load(f)["checks"]["t2i"]["velocity_vs_replaced"]["sim"] == pytest.approx(1e-3)
    # the errors after the vision span the replaced file holds must not move either
    moved["cases"]["t2i"]["velocity"]["sim_error"][0] -= 1e-3
    moved["cases"]["edit"]["encoder_after_vision"]["bf16_error"] += 1e-3
    with open(thresholds, "w", encoding="utf-8") as f:
        json.dump(moved, f)
    with pytest.raises(AssertionError, match="after_vision"):
        session(True, supersede=True)
    assert not [name for name in os.listdir(out) if name.endswith(".new.json")]
