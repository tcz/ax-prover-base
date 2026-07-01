"""Upload the HyDE retrieval artifacts to a HuggingFace **dataset** repo.

tools/hyde_search.py fetches these by basename from its configured ``hf_repo`` when they are not
present locally. Upload the FAISS index, the id map, and the LeanExplore DB (decl id ->
name/source_text, used to render results).

    huggingface-cli login            # or export HF_TOKEN=...
    python upload_hf.py --repo <user>/ax-prover-mathlib-hyde-4b \
        --files embeddings/q3e4B_sig.faiss embeddings/ids.npy \
                ~/.lean_explore/cache/*/lean_explore.db
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. user/ax-prover-mathlib-hyde-4b")
    ap.add_argument("--files", nargs="+", required=True, help="artifact paths to upload (by basename)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    for path in args.files:
        name = os.path.basename(path)
        print(f"uploading {name} ({os.path.getsize(path) / 1e9:.2f} GB) ...", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=args.repo, repo_type="dataset")
    print(f"done -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
