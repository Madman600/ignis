"""The tensor map: spec 01's encoding table, tensor by tensor, on the pinned checkpoint's list."""
import collections

import pytest

import pin
import recipe


@pytest.fixture(scope="module")
def items():
    out = {}
    for comp, _, name, _, shape in pin.tensors(pin.load()):
        out[(comp, name)] = recipe.plan(comp, name, shape)
    return out


def test_the_spec_table_holds_on_named_tensors(items):
    def enc(comp, name):
        item = items[(comp, name)]
        return None if item is None else (item.object, item.encoding, list(item.shape))

    # DiT block linears, img_in, txt_in's two linears, proj_out: FP8 with a row scale
    for name in ("to_q", "to_k", "to_v", "to_out.0"):
        assert enc("transformer", f"transformer_blocks.7.attn.{name}.weight")[1] == "fp8"
    for name in ("proj", "gate_layer", "out"):
        assert enc("transformer", f"transformer_blocks.31.img_mlp.{name}.weight")[1] == "fp8"
    assert enc("transformer", "img_in.weight") == ("transformer/img_in.weight", "fp8", [4096, 64])
    assert enc("transformer", "txt_in.in_layer.weight")[1] == "fp8"
    assert enc("transformer", "txt_in.out_layer.weight")[1] == "fp8"
    assert enc("transformer", "proj_out.weight") == ("transformer/proj_out.weight", "fp8", [64, 4096])
    # the shared modulation, the timestep embedder and every norm: bf16 (norm_out's linear is
    # its modulation, read from the timestep embedding)
    assert enc("transformer", "modulation.1.weight") == ("transformer/modulation.1.weight", "bf16", [16384, 4096])
    assert enc("transformer", "time_text_embed.timestep_embedder.linear_1.weight")[1] == "bf16"
    assert enc("transformer", "time_text_embed.timestep_embedder.linear_2.weight")[1] == "bf16"
    assert enc("transformer", "norm_out.linear.weight")[1] == "bf16"
    assert enc("transformer", "txt_in.text_norm.weight")[1] == "bf16"
    assert enc("transformer", "transformer_blocks.0.attn.norm_k.weight")[1] == "bf16"
    # the text encoder's decoder linears FP8, its embedding bf16, lm_head and the final norm dropped
    assert enc("text_encoder", "model.language_model.layers.35.mlp.down_proj.weight") == \
        ("text_encoder/layers.35.mlp.down_proj.weight", "fp8", [4096, 12288])
    assert enc("text_encoder", "model.language_model.layers.0.self_attn.k_proj.weight")[1] == "fp8"
    assert enc("text_encoder", "model.language_model.layers.0.self_attn.q_norm.weight")[1] == "bf16"
    assert enc("text_encoder", "model.language_model.embed_tokens.weight") == \
        ("text_encoder/embed_tokens.weight", "bf16", [151936, 4096])
    assert enc("text_encoder", "lm_head.weight") is None
    assert enc("text_encoder", "model.language_model.norm.weight") is None
    # the vision tower under the 27B's names (crates/artifact/src/inventory.rs vision_scope_27b),
    # every matrix W8
    assert enc("text_encoder", "model.visual.patch_embed.proj.weight") == \
        ("vision/patch_embedding", "w8", [1152, 1536])
    assert enc("text_encoder", "model.visual.blocks.26.attn.qkv.weight") == \
        ("vision/layers/26/attention/qkv", "w8", [3456, 1152])
    assert enc("text_encoder", "model.visual.blocks.0.attn.proj.weight") == \
        ("vision/layers/0/attention/output", "w8", [1152, 1152])
    assert enc("text_encoder", "model.visual.blocks.0.mlp.linear_fc1.weight")[1] == "w8"
    assert enc("text_encoder", "model.visual.blocks.0.mlp.linear_fc2.weight")[1] == "w8"
    assert enc("text_encoder", "model.visual.blocks.3.norm2.bias") == ("vision/layers/3/norm2/bias", "bf16", [1152])
    assert enc("text_encoder", "model.visual.merger.linear_fc2.weight") == ("vision/merger/fc2", "w8", [4096, 4608])
    assert enc("text_encoder", "model.visual.deepstack_merger_list.2.linear_fc1.weight") == \
        ("vision/deepstack/2/fc1", "w8", [4608, 4608])
    assert enc("text_encoder", "model.visual.deepstack_merger_list.2.norm.weight") == \
        ("vision/deepstack/2/norm/weight", "bf16", [4608])
    # the VAE, quant_conv and post_quant_conv: bf16, their own shapes
    assert enc("vae", "decoder.conv_in.weight") == ("vae/decoder.conv_in.weight", "bf16", [1152, 64, 3, 3])
    assert enc("vae", "post_quant_conv.weight")[1] == "bf16"


def test_every_pinned_tensor_is_mapped_and_counted(items):
    counts = collections.Counter()
    for (comp, _), item in items.items():
        counts[(comp, None if item is None else item.encoding)] += 1
    # text encoder: 36 x 7 linears, the embedding and 36 x 4 norms, two dropped, and the vision
    # tower (27 blocks of 12, patch/bias/positions, the merger and 3 DeepStack mergers of 6)
    assert counts[("text_encoder", "fp8")] == 36 * 7
    assert counts[("text_encoder", None)] == 2
    assert counts[("text_encoder", "w8")] == 1 + 27 * 4 + 2 + 3 * 2
    assert not {enc for _, enc in counts} & {"q4", "q5", "q6"}
    assert counts[("text_encoder", "bf16")] == 1 + 36 * 4 + 2 + 27 * 8 + 4 + 3 * 4
    # transformer: 32 x 7 block linears + img_in, txt_in x 2, proj_out
    assert counts[("transformer", "fp8")] == 32 * 7 + 4
    assert counts[("transformer", "bf16")] == 32 * 2 + 5
    assert counts[("vae", "bf16")] == 238
    objects = [i.object for i in items.values() if i is not None]
    assert len(objects) == len(set(objects)) == 748 + 297 + 238


def test_an_unknown_tensor_is_refused_by_name():
    with pytest.raises(ValueError, match="transformer_blocks.0.attn.add_q_proj.weight"):
        recipe.plan("transformer", "transformer_blocks.0.attn.add_q_proj.weight", [4096, 4096])
    with pytest.raises(ValueError, match="model.visual.blocks.0.attn.rel_pos"):
        recipe.plan("text_encoder", "model.visual.blocks.0.attn.rel_pos", [7])
    with pytest.raises(ValueError, match="scheduler"):
        recipe.plan("scheduler", "x", [1])
