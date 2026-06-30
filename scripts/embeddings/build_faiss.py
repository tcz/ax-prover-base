"""Build FAISS cosine indexes from embed.py output (run on CPU).

One IndexFlatIP per emb_*.npy in the directory (vectors are L2-normalized, so inner product =
cosine). ids.npy aligns row i of each index to declaration id ids[i]. The HyDE tool
(tools/hyde_search.py) loads the index + ids.npy + the LeanExplore DB.

    python build_faiss.py --emb-dir ./embeddings
"""

import argparse
import glob
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dir", default=".")
    args = ap.parse_args()

    import faiss

    ids = np.load(os.path.join(args.emb_dir, "ids.npy"))
    print(f"ids: {len(ids)}")
    for f in sorted(glob.glob(os.path.join(args.emb_dir, "emb_*.npy"))):
        tag = os.path.basename(f)[4:-4]  # strip "emb_" prefix and ".npy" suffix
        emb = np.load(f).astype(np.float32)
        assert emb.shape[0] == len(ids), f"{tag}: {emb.shape[0]} rows vs {len(ids)} ids"
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        out = os.path.join(args.emb_dir, f"{tag}.faiss")
        faiss.write_index(index, out)
        print(f"{tag}: {emb.shape} -> {out}")


if __name__ == "__main__":
    main()
