# transformers 5.17.0's Qwen3-VL vision tower on the CPU on the edit case's pixel_values: the
# per-stage references for #320's GPU tower (agent-320b). Writes, fp32 [1024][4096] each:
#   merger.npy, deepstack0..2.npy        the checkpoint's weights in fp32 (vision_probe's `fp32`)
#   w8_merger.npy, w8_deepstack0..2.npy  the artifact's W8 weights decoded, in fp32 (`artifact`)
# The GPU test reads them from IGNIS_IMAGE_VISION_REF.
#   F:/ai/venvs/ignis-image/Scripts/python.exe tools/image-converter/ref_vision.py #       <checkpoint snapshot> <oracle> <out dir>
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # vision_probe, beside this file
import vision_probe
snap, oracle, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(8)
config, state = vision_probe.vision_state(snap)
print("deepstack_visual_indexes", list(config.deepstack_visual_indexes), "depth", config.depth,
      "out_hidden", config.out_hidden_size, flush=True)
pv = np.load(os.path.join(oracle, "edit", "pixel_values.npy"))
grid = np.load(os.path.join(oracle, "edit", "image_grid_thw.npy"))
t0 = time.time()
merger, deep = vision_probe.outputs(config, state, pv, grid, torch.float32)
np.save(os.path.join(out, "merger.npy"), merger.numpy())
for i, d in enumerate(deep):
    np.save(os.path.join(out, f"deepstack{i}.npy"), d.numpy())
print(f"fp32 {time.time() - t0:.1f} s, merger {tuple(merger.shape)}", flush=True)
t0 = time.time()
q, _ = vision_probe.artifact_weights(state)
merger, deep = vision_probe.outputs(config, q, pv, grid, torch.float32)
np.save(os.path.join(out, "w8_merger.npy"), merger.numpy())
for i, d in enumerate(deep):
    np.save(os.path.join(out, f"w8_deepstack{i}.npy"), d.numpy())
print(f"w8 fp32 {time.time() - t0:.1f} s", flush=True)
