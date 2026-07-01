"""Embed the declaration corpus with a Qwen3 embedding model (run on a CUDA GPU).

Reads corpus.jsonl.gz (from extract_corpus.py), embeds one field — "sig" (formal Lean signatures)
or "informal" (natural-language descriptions) — L2-normalizes, and writes:

    ids.npy          int64 declaration ids; row i of every embedding <-> ids[i]
    emb_<tag>.npy    float16 (N, dim) embeddings; tag e.g. "q3e4B_sig"

The centerpiece index in the PR is Qwen3-Embedding-4B over "sig" (formal, 2560-d). 0.6B is much
faster (1024-d). 4B over the ~8x-longer informalizations is compute-bound (~4 h on an RTX 3090) —
use a 4090/A100, or pass --max-seq-length to truncate. Build the FAISS index afterwards with
build_faiss.py (CPU).

    python embed.py --corpus corpus.jsonl.gz --model Qwen/Qwen3-Embedding-4B --field sig --batch 64
"""

import argparse
import gzip
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus.jsonl.gz")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--field", choices=["sig", "informal"], default="sig")
    ap.add_argument("--out", default=".")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=None, help="truncate inputs (speeds 4B up)")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    assert torch.cuda.is_available(), "no CUDA device — embedding the full corpus on CPU is too slow"

    ids, texts = [], []
    with gzip.open(args.corpus, "rt") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["id"])
            texts.append(r[args.field] or r["name"])  # fall back to the name if the field is empty
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "ids.npy"), np.array(ids, dtype=np.int64))

    tag = args.model.split("/")[-1].replace("Qwen3-Embedding-", "q3e") + "_" + args.field
    print(f"gpu={torch.cuda.get_device_name(0)} corpus={len(ids)} tag={tag}", flush=True)

    model = SentenceTransformer(
        args.model, device="cuda", trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16}
    )
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    t = time.time()
    emb = model.encode(
        texts,
        batch_size=args.batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float16")
    out = os.path.join(args.out, f"emb_{tag}.npy")
    np.save(out, emb)
    print(f"{tag}: {emb.shape} in {time.time() - t:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
