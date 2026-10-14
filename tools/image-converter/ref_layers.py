# transformers 5.17.0's Qwen3-VL text model in fp32 on the CPU, layer by layer, on a recorded case's
# input_ids and mrope positions: the per-layer reference for #320's GPU encoder (agent-320a).
# Weights: the checkpoint's bf16, upcast to fp32 per layer (as spec 01's recorder computes).
# Writes layer_in.npy and layer00.npy .. layer35.npy, fp32 [tokens][4096]: the inputs of the GPU test's
# per-layer diagnosis (IGNIS_IMAGE_ENCODER_LAYERS), which runs each layer from the previous one's file.
#   F:/ai/venvs/ignis-image/Scripts/python.exe tools/image-converter/ref_layers.py #       <checkpoint snapshot> <oracle>/t2i <out dir>
import json, sys, os, time
import numpy as np, torch
from safetensors import safe_open
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl as m

snap, case_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
torch.set_num_threads(8)
cfg = Qwen3VLConfig.from_pretrained(os.path.join(snap, "text_encoder")).text_config
cfg._attn_implementation = "eager"
ids = torch.from_numpy(np.load(os.path.join(case_dir, "input_ids.npy")))
pos = torch.from_numpy(np.load(os.path.join(case_dir, "mrope_position_ids.npy")))
want = np.load(os.path.join(case_dir, "encoder_hidden_states.npy"))
T = ids.shape[1]
f = safe_open(os.path.join(snap, "text_encoder", "model.safetensors"), "pt")
pre = "model.language_model."
emb = f.get_slice(pre + "embed_tokens.weight")
x = torch.stack([emb[int(i)] for i in ids[0]]).float()[None]
rot = m.Qwen3VLTextRotaryEmbedding(config=cfg)
cos, sin = rot(x, pos)
mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
np.save(os.path.join(out, "layer_in.npy"), x[0].numpy())
t0 = time.time()
for l in range(cfg.num_hidden_layers):
    layer = m.Qwen3VLTextDecoderLayer(cfg, l)
    sd = {k[len(pre + f"layers.{l}."):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(pre + f"layers.{l}.")}
    missing = layer.load_state_dict(sd, strict=True)
    layer = layer.float().eval()
    with torch.no_grad():
        x = layer(x, position_embeddings=(cos, sin), attention_mask=mask)
    np.save(os.path.join(out, f"layer{l:02d}.npy"), x[0].numpy())
    del layer, sd
print(f"{cfg.num_hidden_layers} layers in {time.time() - t0:.1f} s")
got = x[0].numpy().astype(np.float64)
ref = want[0].astype(np.float64)
rel = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
print("CPU fp32 vs the oracle, per-row rel L2: max %.3e, median %.3e" % (rel.max(), np.median(rel)))
