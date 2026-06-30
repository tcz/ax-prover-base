"""Unit tests for the embedding-based HyDE retrieval tool.

The FAISS index, the embedder, and the hypothesis-drafting LLM are all faked, so these run with
no network, no faiss, and no sentence-transformers — only the pure fusion/formatting/search logic
is exercised.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from ax_prover.tools import hyde_search
from ax_prover.tools.hyde_search import (
    HYDE_TOOL_TYPE,
    HydeResources,
    SearchHydeConfig,
    _max_cosine_fuse,
    _parse_hypotheticals,
    create_search_hyde_tool,
)
from ax_prover.tools.registry import TOOL_REGISTRY


def test_tool_is_registered():
    assert HYDE_TOOL_TYPE in TOOL_REGISTRY


def test_max_cosine_fuse_ranks_by_best_similarity():
    # lemA appears in two queries; its fused score is the MAX of its cosines, not the sum.
    per_query = [
        [("lemA", 0.40), ("lemB", 0.30)],
        [("lemA", 0.90), ("lemC", 0.10)],
    ]
    fused = _max_cosine_fuse(per_query)
    assert fused == [("lemA", 0.90), ("lemB", 0.30), ("lemC", 0.10)]


def test_parse_hypotheticals_strips_fences_and_bullets():
    text = "```lean\n- theorem foo : a = b\n2. theorem bar : c = d\n```\n"
    assert _parse_hypotheticals(text, limit=5) == ["theorem foo : a = b", "theorem bar : c = d"]


def _fake_resources(monkeypatch):
    """Three lemmas; the fake index returns fixed (cosine, row) hits so fusion is deterministic."""
    row_name = ["lemA", "lemB", "lemC"]
    name_block = {n: f"\n• {n}\n  sig_{n}" for n in row_name}

    class _FakeIndex:
        def search(self, vecs, depth):
            n = vecs.shape[0]
            # every query ranks the same 3 rows; cosines differ only on the FIRST query so that
            # max-fusion has a clear winner (lemA=0.9 > lemB=0.5 > lemC=0.1).
            sims = np.tile(np.array([0.2, 0.2, 0.1], dtype="float32"), (n, 1))
            sims[0] = np.array([0.9, 0.5, 0.1], dtype="float32")
            idx = np.tile(np.array([0, 1, 2]), (n, 1))
            return sims[:, :depth], idx[:, :depth]

    def _embed(texts):
        return np.zeros((len(texts), 4), dtype="float32")

    monkeypatch.setattr(hyde_search, "LLMClient", lambda *a, **k: object())

    async def _fake_gen(query, client, config):
        return ["h1", "h2", "h3"][: config.num_hypotheses]

    monkeypatch.setattr(hyde_search, "_generate_hypotheticals", _fake_gen)

    res = HydeResources(
        index=_FakeIndex(),
        row_name=row_name,
        name_block=name_block,
        embed=_embed,
        search_lock=asyncio.Semaphore(1),
    )
    return SimpleNamespace(get_tool_resources=lambda _t: res)


def test_search_returns_top_k_ranked_by_cosine(monkeypatch):
    runtime = _fake_resources(monkeypatch)
    config = SearchHydeConfig(num_hypotheses=3, retrieval_depth=3, max_results=2)
    tool = create_search_hyde_tool(config, runtime)

    out = asyncio.run(tool.coroutine("commutativity of multiplication"))

    lines = [ln for ln in out.splitlines() if ln.startswith("• ")]
    assert len(lines) == 2  # max_results
    assert "lemA" in lines[0] and "lemB" in lines[1]  # ranked by fused cosine
    assert "[cos=0.900]" in out and "[cos=0.500]" in out


def test_search_cosine_cutoff_drops_low_hits(monkeypatch):
    runtime = _fake_resources(monkeypatch)
    config = SearchHydeConfig(num_hypotheses=3, retrieval_depth=3, max_results=6, cosine_cutoff=0.6)
    tool = create_search_hyde_tool(config, runtime)

    out = asyncio.run(tool.coroutine("query"))

    assert "lemA" in out  # 0.9 >= 0.6
    assert "lemB" not in out and "lemC" not in out  # 0.5, 0.1 < cutoff


def test_tool_disabled_when_resources_missing():
    runtime = SimpleNamespace(get_tool_resources=lambda _t: None)
    assert create_search_hyde_tool(SearchHydeConfig(), runtime) is None


def test_generate_hypotheticals_uses_structured_output():
    import json

    class _Client:
        async def ainvoke(self, messages, output_schema=None):
            assert output_schema is not None  # structured path is tried first
            return SimpleNamespace(text=json.dumps({"lemmas": ["theorem a", "theorem b"]}))

    config = SearchHydeConfig(num_hypotheses=2)
    hyps = asyncio.run(hyde_search._generate_hypotheticals("goal", _Client(), config))
    assert hyps == ["theorem a", "theorem b"]


def test_generate_hypotheticals_falls_back_to_raw_query_on_failure():
    class _Client:
        async def ainvoke(self, messages, output_schema=None):
            raise RuntimeError("llm down")

    config = SearchHydeConfig(num_hypotheses=3)
    hyps = asyncio.run(hyde_search._generate_hypotheticals("the goal", _Client(), config))
    assert hyps == ["the goal"]  # safety net: never crash the search


@pytest.mark.parametrize(
    "mode,needle",
    [("formal", "Lean 4 theorem signature"), ("informal", "natural-language")],
)
def test_prompt_mode_selects_template(mode, needle):
    from ax_prover.tools.hyde_search import HYDE_PROMPT_FORMAL, HYDE_PROMPT_INFORMAL

    template = HYDE_PROMPT_FORMAL if mode == "formal" else HYDE_PROMPT_INFORMAL
    assert needle in template
