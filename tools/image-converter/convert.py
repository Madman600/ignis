"""Qwen/Qwen-Image-2.1-Turbo -> the ignis image artifact (spec image-model/01, GitHub #318).

    python convert.py run --checkpoint SNAPSHOT_DIR [--out DIR] [--fp8-bound 0.03] [--packer EXE]
    python convert.py fixture --out DIR     # a synthetic checkpoint and its work tree, CPU only

`run` checks the checkpoint against pin.json (revision, every file's sha256, the tensor list) and
refuses any difference by name. It then streams the tensors one at a time (safetensors mmap) into
the work tree DIR/work: one unit per part (frontend, text_encoder, vision, transformer, vae), each
a tensors.json, one payload file per tensor and a DONE marker (docs/specs/flash-next/layout.md §2,
§6.3), the encodings of recipe.py. DIR/work/converter.json records every tensor (source name,
encoding, bytes, sha256, max |value|, round-trip relative error), the VAE's convolution shapes,
the revision and this tool's commit. Last, the packer (`ignis-artifact-pack --family qwen-image`,
crates/artifact) assembles DIR/ignis_image_2_1_turbo-v2.ninfer from the work tree, deleting the
units as it appends them, and merges converter.json into the sidecar
DIR/ignis_image_2_1_turbo-v2.ninfer.conversion.json. Python never writes the container.

Exit codes: 0 done; 2 refused (not the pinned checkpoint, a NaN or an infinity, an FP8 tensor
above --fp8-bound, a work tree already there); 3 the packer failed; 1 error.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch

import fp8
import pin
import recipe
import rowsplit
import work

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ARTIFACT = "ignis_image_2_1_turbo-v2.ninfer"
PACKER = os.path.join(REPO, "target", "x86_64-pc-windows-msvc", "release", "ignis-artifact-pack.exe")
UNITS = ("frontend", "text_encoder", "vision", "transformer", "vae")
# The FP8 round-trip relative error (||W - decode(encode(W))||_F / ||W||_F) the 27B's FP8 linears
# carry: 2.64-2.66% on all ten measured (q/k/v/o, GDN in/out, MLP gate/up/down; fp8_bound.py,
# README "The FP8 bound"), rounded up to 3%.
FP8_BOUND = 0.03
# checkpoint file -> frontend resource name (`frontend/<name>` in the container)
FRONTEND = {
    "processor/tokenizer.json": "tokenizer.json",
    "processor/tokenizer_config.json": "tokenizer_config.json",
    "processor/chat_template.jinja": "chat_template.jinja",
    "processor/processor_config.json": "processor_config.json",
    "model_index.json": "model_index.json",
    "scheduler/scheduler_config.json": "scheduler_config.json",
    "text_encoder/config.json": "text_encoder_config.json",
    "text_encoder/generation_config.json": "text_encoder_generation_config.json",
    "transformer/config.json": "transformer_config.json",
    "vae/config.json": "vae_config.json",
}
IMAGE_MODEL = "image_model.json"

EXIT_REFUSED, EXIT_PACKER, EXIT_ERROR = 2, 3, 1


def unit_of(obj):
    return obj.split("/", 1)[0]


def file_name(obj):
    """The payload file of an object: its name with `/` as `.` (checked unique per unit)."""
    return obj.replace("/", ".") + ".bin"


def versions():
    import diffusers
    import numpy
    import safetensors
    import transformers
    return {"torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
            "safetensors": safetensors.__version__, "numpy": numpy.__version__, "python": platform.python_version()}


def tool_commit():
    def git(*args):
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--", "tools/image-converter"))}


def encode(item, source, t, fp8_bound):
    """(payload, record) of one tensor. Refuses a NaN or an infinity in the source or in what the
    payload decodes to, a non-bf16 source, and an FP8 round trip above `fp8_bound`."""
    if t.dtype != torch.bfloat16:
        raise pin.Refused(f"{source} is {t.dtype}, the converter takes the checkpoint's bf16")
    if not bool(torch.isfinite(t).all()):
        raise pin.Refused(f"{source} holds a NaN or an infinity")
    shape = list(item.shape)
    err = None
    if item.encoding == "bf16":
        payload = t.contiguous().view(torch.int16).numpy().tobytes()
    elif item.encoding == "fp8":
        payload = fp8.encode(t)
        decoded = fp8.decode_fp32(payload, shape)
        err = fp8.rel_error(t.float(), decoded)
    else:
        w = t.reshape(shape)
        payload = rowsplit.encode(w, item.encoding)
        decoded = rowsplit.decode(payload, item.encoding, shape)
        err = fp8.rel_error(w.float(), decoded)
    if err is not None and not bool(torch.isfinite(decoded).all()):
        raise pin.Refused(f"{source} decodes from its {item.encoding} payload to a NaN or an infinity")
    if item.encoding == "fp8" and err > fp8_bound:
        raise pin.Refused(f"{source}: FP8 round-trip relative error {err:.4e} is above the bound {fp8_bound:g} "
                          f"(--fp8-bound)")
    fmt, layout = recipe.FORMATS[item.encoding]
    record = {"source": source, "object": item.object, "encoding": item.encoding, "format": fmt, "layout": layout,
              "shape": shape, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
              "max_abs": float(t.float().abs().max()) if t.numel() else 0.0, "rel_error": err}
    return payload, record


def _copy(src, dst):
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data)


def convert(checkpoint, p, out_work, fp8_bound=FP8_BOUND, status="complete", log=print, command=None):
    """Writes the work tree `out_work` from a checkpoint matching the pin `p`; returns the
    converter.json record. Raises pin.Refused (nothing to pack) on any refusal."""
    import metadata
    from safetensors import safe_open

    t0 = time.time()
    pin.check(checkpoint, p)
    if os.path.isdir(out_work) and os.listdir(out_work):
        raise pin.Refused(f"{out_work} is not empty: remove it to convert again")
    vers = versions()

    plan = {u: [] for u in UNITS[1:]}
    not_stored = []
    for comp, path, name, dtype, shape in pin.tensors(p):
        item = recipe.plan(comp, name, shape)
        if item is None:
            not_stored.append(f"{comp}/{name}")
        else:
            plan[unit_of(item.object)].append((comp, path, name, item))

    d = os.path.join(out_work, "frontend")
    os.makedirs(d)
    for src, dst in FRONTEND.items():
        _copy(os.path.join(checkpoint, src), os.path.join(d, dst))
    work.write_json_atomic(os.path.join(d, IMAGE_MODEL), metadata.image_model(checkpoint, p, vers["diffusers"]))
    work.mark_done(d)
    log(f"frontend: {len(FRONTEND) + 1} resources")

    records = []
    for unit in UNITS[1:]:
        d = os.path.join(out_work, unit)
        os.makedirs(d)
        entries, files = [], set()
        handles = {}
        started = time.time()
        for comp, path, name, item in plan[unit]:
            if path not in handles:
                handles[path] = safe_open(os.path.join(checkpoint, path), framework="pt")
            payload, record = encode(item, f"{comp}/{name}", handles[path].get_tensor(name), fp8_bound)
            fname = file_name(item.object)
            if fname in files:
                raise ValueError(f"two objects of {unit} share the file name {fname}")
            files.add(fname)
            with open(os.path.join(d, fname), "wb") as f:
                f.write(payload)
            entries.append({"name": item.object, "file": fname, "format": record["format"],
                            "layout": record["layout"], "shape": record["shape"], "bytes": record["bytes"],
                            "sha256": record["sha256"]})
            records.append(record)
        handles.clear()
        work.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": entries})
        work.mark_done(d, {e["file"]: e["sha256"] for e in entries})
        log(f"{unit}: {len(entries)} tensors, {sum(e['bytes'] for e in entries)} bytes, "
            f"{time.time() - started:.1f} s")

    summary = {}
    for r in records:
        s = summary.setdefault(r["encoding"], {"count": 0, "bytes": 0, "max_rel_error": None})
        s["count"] += 1
        s["bytes"] += r["bytes"]
        if r["rel_error"] is not None:
            s["max_rel_error"] = max(s["max_rel_error"] or 0.0, r["rel_error"])
    record = {
        "schema": "image-converter-v1",
        "status": status,
        "verdict": "PASS",
        "source": {"repo": p["repo"], "revision": p["revision"], "files": p["files"]},
        "converter": dict(tool_commit(), command=command),
        "versions": vers,
        "fp8_bound": fp8_bound,
        "objects": {"tensors": len(records), "resources": len(FRONTEND) + 1, "not_stored": not_stored},
        "all_finite": True,
        "summary": summary,
        "vae_convs": metadata.vae_convs(json.load(open(os.path.join(checkpoint, "vae/config.json"), encoding="utf-8"))),
        "tensors": records,
        "time_s": round(time.time() - t0, 1),
    }
    work.write_json_atomic(os.path.join(out_work, "converter.json"), record)
    return record


def run_packer(packer, out_work, artifact, accept_status=None):
    cmd = [os.path.abspath(packer), "--family", "qwen-image", "--work", out_work, "--out", artifact]
    if accept_status:
        cmd += ["--accept-status", accept_status]
    return subprocess.run(cmd).returncode


def cmd_run(a):
    p = pin.load(a.pin)
    out_work = os.path.join(a.out, "work")
    if not a.no_pack and not os.path.isfile(a.packer):
        print(f"REFUSED: no packer at {a.packer}: cargo build --release -p ignis-artifact --bin ignis-artifact-pack",
              file=sys.stderr)
        return EXIT_REFUSED
    try:
        record = convert(a.checkpoint, p, out_work, a.fp8_bound, command=" ".join(sys.argv))
    except pin.Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    s = record["summary"]
    print(f"converted {record['objects']['tensors']} tensors in {record['time_s']} s; FP8 max error "
          f"{s['fp8']['max_rel_error']:.4e} (bound {a.fp8_bound:g})")
    if a.no_pack:
        return 0
    code = run_packer(a.packer, out_work, os.path.join(a.out, ARTIFACT))
    return 0 if code == 0 else EXIT_PACKER


def cmd_fixture(a):
    import synthetic
    snap, p = synthetic.make(os.path.join(a.out, "checkpoint"))
    with open(os.path.join(a.out, "pin.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(p, f, indent=1)
    record = convert(snap, p, os.path.join(a.out, "work"), status="fixture",
                     command="convert.py fixture", log=lambda *x: None)
    print(f"fixture work tree written to {a.out}/work: {record['objects']['tensors']} tensors")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="convert the pinned checkpoint and pack the artifact")
    r.add_argument("--checkpoint", required=True, help="the checkpoint (a Hugging Face cache snapshot directory)")
    r.add_argument("--out", default="F:/ai/models/ignis-image-2.1-turbo")
    r.add_argument("--pin", default=pin.PIN)
    r.add_argument("--fp8-bound", type=float, default=FP8_BOUND,
                   help="refuse an FP8 tensor whose round-trip relative error is above this "
                        "(default: the 27B's FP8 linears, 2.66%%, rounded up)")
    r.add_argument("--packer", default=PACKER)
    r.add_argument("--no-pack", action="store_true", help="stop after the work tree")
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("fixture", help="a synthetic checkpoint and its work tree (crates/artifact tests)")
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fixture)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
