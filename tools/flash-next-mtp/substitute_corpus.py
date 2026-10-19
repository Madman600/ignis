"""A calibration corpus for the MTP head when the compression study's token files are not at hand
(spec flash-next/08: re-calibrating the head on a re-tabled trunk).

    python substitute_corpus.py --sources <dir> --repo <ignis checkout> --out <calib>

The study's corpus (corpus_manifest.json) is pinned by sha256 to files that live outside the
repository. This builds one of the same shape from the same allowlisted sources, all public or
the owner's own: the repository's code and docs, the model's own sampled chat turns (`chats.json`,
sampled from the served model), wikitext-103 train, Wikipedia 20231101.it shard 0 (the manifest's
own hot-sample file), GSM8K train, the installed transformers' modeling sources, and MMLU-Pro.
Each source is tokenized with the checkpoint's tokenizer (`<sources>/tok`), concatenated and cut
into 2,048-token chunks; a seeded shuffle holds out the first few of each kind as test chunks.
The output is what `convert_head.py chunks` writes (`chunks.json`, `<name>.tokens.u32`), so the
rest of the pipeline is unchanged. A head calibrated on it is not the study's head: compare heads
built on the same corpus.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert_head import export_chunks  # noqa: E402

CH = 2048
SEED = 0
# kind: (calibration chunks, test chunks)
PLAN = {
    "code": (24, 2), "docs": (16, 2), "chat": (12, 2), "wikitext": (12, 2), "wikipedia": (12, 2),
    "gsm8k": (8, 2), "transformers": (12, 2), "mmlu": (16, 4),
}
CODE_EXT = (".rs", ".py", ".ts", ".tsx", ".toml", ".cu", ".h")


def repo_texts(repo, exts):
    files = subprocess.run(["git", "-C", repo, "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    files = sorted(f for f in files if f.endswith(exts))
    random.Random(SEED).shuffle(files)
    for f in files:
        try:
            yield open(os.path.join(repo, f), encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue


def stream(tok, texts, limit, per_text=8192):
    ids = []
    for t in texts:
        ids += tok.encode(t, add_special_tokens=False)[:per_text]
        if len(ids) >= limit:
            break
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(a.sources, "tok"))
    src = lambda name: os.path.join(a.sources, name)  # noqa: E731

    chats = json.load(open(src("chats.json")))
    chat_ids = []
    for c in chats:
        messages = [{"role": "user", "content": c["prompt"]},
                    {"role": "assistant", "content": c["content"], "reasoning_content": c["reasoning"]}]
        templated = tok.apply_chat_template(messages, tokenize=False)
        chat_ids += tok.encode(templated, add_special_tokens=False)
    wiki = pd.read_parquet(src("wikitext-train.parquet"))["text"]
    itwiki = pd.read_parquet(src("wikipedia-it.parquet"), columns=["text"])["text"]
    gsm = pd.read_parquet(src("gsm8k-train.parquet"))
    mmlu = pd.read_parquet(src("mmlu-pro-test.parquet"))
    import transformers
    modeling = sorted(glob.glob(os.path.join(os.path.dirname(transformers.__file__), "models", "*", "modeling_*.py")))
    random.Random(SEED).shuffle(modeling)

    def need(kind):
        cal, test = PLAN[kind]
        return (cal + test) * CH

    texts = {
        "code": lambda: repo_texts(a.repo, CODE_EXT),
        "docs": lambda: repo_texts(a.repo, (".md",)),
        "wikitext": lambda: ("".join(wiki[i:i + 200]) for i in range(0, len(wiki), 200)),
        "wikipedia": lambda: iter(itwiki),
        "gsm8k": lambda: (f"Question: {q}\nAnswer: {s}\n\n" for q, s in zip(gsm["question"], gsm["answer"])),
        "transformers": lambda: (open(f, encoding="utf-8").read() for f in modeling),
        "mmlu": lambda: (
            f"Question: {r.question}\n" + "".join(f"{chr(65 + i)}. {o}\n" for i, o in enumerate(r.options))
            + f"Answer: {r.answer}\n\n" for r in mmlu.itertuples()),
    }
    rng = random.Random(SEED)
    chunks = []
    for kind, (n_cal, n_test) in PLAN.items():
        ids = chat_ids if kind == "chat" else stream(tok, texts[kind](), need(kind))
        pieces = [ids[i:i + CH] for i in range(0, len(ids) - CH + 1, CH)][: n_cal + n_test]
        if len(pieces) < n_test + 1:
            raise SystemExit(f"{kind}: only {len(pieces)} chunks of {CH}")
        order = list(range(len(pieces)))
        rng.shuffle(order)
        test = set(order[:n_test])
        for i, piece in enumerate(pieces):
            chunks.append({"ids": piece, "valid": CH, "kind": kind, "cal": i not in test, "test": i in test,
                           "source": f"substitute:{kind}"})
        print(f"{kind}: {len(pieces) - len(test)} calibration + {len(test)} test chunks", file=sys.stderr)
    # The pipeline's order: calibration chunks first, then the held-out ones.
    chunks = [c for c in chunks if c["cal"]] + [c for c in chunks if c["test"]]
    export_chunks(chunks, a.out)
    with open(os.path.join(a.out, "substitute.json"), "w") as f:
        json.dump({"plan": PLAN, "seed": SEED, "chunk": CH, "sources": sorted(os.listdir(a.sources)),
                   "repo_head": subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True,
                                               text=True).stdout.strip()}, f, indent=1)


if __name__ == "__main__":
    main()
