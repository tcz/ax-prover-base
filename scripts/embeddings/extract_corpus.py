"""Extract the Mathlib declaration corpus from a LeanExplore SQLite DB into corpus.jsonl.gz.

Each output line is one declaration: {"id", "name", "sig" (source_text), "informal"
(informalization)}. This corpus feeds embed.py.

The DB ships with LeanExplore (`pip install "lean-explore[local]"`, then it downloads an index
under ~/.lean_explore/cache/<timestamp>/lean_explore.db). The build used in the PR had 388,826
declarations.

    python extract_corpus.py --db ~/.lean_explore/cache/*/lean_explore.db --out corpus.jsonl.gz
"""

import argparse
import gzip
import json
import sqlite3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path to lean_explore.db")
    ap.add_argument("--out", default="corpus.jsonl.gz")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    n = 0
    with gzip.open(args.out, "wt") as f:
        for id_, name, sig, informal in con.execute(
            "SELECT id, name, source_text, informalization FROM declarations"
        ):
            f.write(
                json.dumps(
                    {"id": id_, "name": name, "sig": sig or "", "informal": informal or ""}
                )
                + "\n"
            )
            n += 1
    con.close()
    print(f"wrote {n} declarations -> {args.out}")


if __name__ == "__main__":
    main()
