"""The checkpoint the converter accepts: one revision, its files and its tensor list (spec 01, AC 1).

`pin.json` records `Qwen/Qwen-Image-2.1-Turbo` at the pinned revision: every file the converter
reads with its size and sha256 (a safetensors file's sha256 is its LFS oid), and every tensor of
every safetensors file with its dtype and shape, read from the file headers. `check` refuses a
checkpoint directory that differs from a pin in any of these, naming the difference.

A checkpoint directory is a diffusers tree (`model_index.json`, `transformer/`, ...). Its revision
is read from the path for a Hugging Face cache snapshot (`.../snapshots/<revision>/`) and from the
`.cache/huggingface/download/*.metadata` files of an `hf download --local-dir` copy; any other
directory has none, and a refusal says so. The file hashes decide either way.
"""
import hashlib
import json
import os
import struct

PIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pin.json")
SAFETENSORS = ".safetensors"


class Refused(Exception):
    """The checkpoint is not the pinned one; the message names every difference."""


def sha256_file(path, block=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_header(path):
    """A safetensors file's tensors: name -> (dtype, shape), without reading the data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: (v["dtype"], list(v["shape"])) for k, v in header.items()}


def component(path):
    """The diffusers component a file belongs to: its first path segment (`text_encoder`, ...)."""
    return path.split("/", 1)[0]


def build(repo, revision, files, headers):
    """A pin from `files` (relative path -> (bytes, sha256)) and `headers` (relative path of each
    safetensors file -> its read_header)."""
    tensors = {}
    for path in sorted(headers):
        for name, (dtype, shape) in sorted(headers[path].items()):
            tensors.setdefault(component(path), {})[name] = [path, dtype, shape]
    return {"repo": repo, "revision": revision,
            "files": {p: {"bytes": b, "sha256": s} for p, (b, s) in sorted(files.items())},
            "tensors": tensors}


def from_checkpoint(directory, repo, revision, paths):
    """The pin of a local checkpoint: `paths` are the files to record, relative, with `/`."""
    files = {p: (os.path.getsize(os.path.join(directory, p)), sha256_file(os.path.join(directory, p)))
             for p in paths}
    headers = {p: read_header(os.path.join(directory, p)) for p in paths if p.endswith(SAFETENSORS)}
    return build(repo, revision, files, headers)


def load(path=PIN):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def snapshot_revision(directory):
    """The revision of a Hugging Face cache snapshot directory, or None for any other directory."""
    parent, name = os.path.split(os.path.normpath(os.path.abspath(directory)))
    return name if os.path.basename(parent) == "snapshots" else None


def revisions(directory, paths):
    """The revisions a checkpoint directory names: a snapshot's, or the commits an
    `hf download --local-dir` copy records for `paths` (each file's `.metadata`, commit first)."""
    snapshot = snapshot_revision(directory)
    if snapshot is not None:
        return {snapshot}
    found = set()
    for path in paths:
        meta = os.path.join(directory, ".cache", "huggingface", "download", path + ".metadata")
        if os.path.isfile(meta):
            with open(meta, encoding="utf-8") as f:
                found.add(f.readline().strip())
    return found


def _named(kind, names, limit=8):
    names = sorted(names)
    more = f" and {len(names) - limit} more" if len(names) > limit else ""
    return f"{kind} ({len(names)}): {', '.join(names[:limit])}{more}"


def check(directory, pin, hashes=True):
    """Refuses (Refused) a checkpoint whose revision, files or tensor list differ from `pin`.

    `hashes=False` skips the sha256 of the files (sizes and tensors are still checked); the
    converter always hashes, the flag is for tests that only exercise the tensor list."""
    problems = []
    named = revisions(directory, pin["files"])
    for revision in sorted(named - {pin["revision"]}):
        problems.append(f"revision {revision} is not the pinned {pin['revision']}")
    for path, want in pin["files"].items():
        full = os.path.join(directory, path)
        if not os.path.isfile(full):
            problems.append(f"file {path} is missing")
            continue
        size = os.path.getsize(full)
        if size != want["bytes"]:
            problems.append(f"file {path} has {size} bytes, the pin {want['bytes']}")
        elif hashes and sha256_file(full) != want["sha256"]:
            problems.append(f"file {path} does not hash to the pinned sha256 {want['sha256']}")
    found = {}
    for path in pin["files"]:
        full = os.path.join(directory, path)
        if path.endswith(SAFETENSORS) and os.path.isfile(full):
            for name, (dtype, shape) in read_header(full).items():
                found.setdefault(component(path), {})[name] = [path, dtype, shape]
    for comp in sorted(set(pin["tensors"]) | set(found)):
        want, got = pin["tensors"].get(comp, {}), found.get(comp, {})
        missing = set(want) - set(got)
        extra = set(got) - set(want)
        changed = [f"{n} {got[n][1]}{got[n][2]} (pinned {want[n][1]}{want[n][2]})"
                   for n in sorted(set(want) & set(got)) if got[n][1:] != want[n][1:]]
        if missing:
            problems.append(f"{comp}: " + _named("tensors missing", missing))
        if extra:
            problems.append(f"{comp}: " + _named("tensors not in the pin", extra))
        if changed:
            problems.append(f"{comp}: " + _named("tensors with another dtype or shape", changed))
    if problems and not named:
        problems.append("its revision is unknown (neither a Hugging Face cache snapshot nor an "
                        "`hf download --local-dir` copy): the differences above stand for it")
    if problems:
        raise Refused(f"{directory} is not {pin['repo']}@{pin['revision']}:\n  " + "\n  ".join(problems))


def tensors(pin):
    """Every pinned tensor as (component, file, name, dtype, shape), in pin order."""
    return [(comp, path, name, dtype, shape)
            for comp, entries in pin["tensors"].items()
            for name, (path, dtype, shape) in entries.items()]
