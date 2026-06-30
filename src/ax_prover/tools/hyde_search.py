"""HyDE retrieval over a local Qwen3-embedding FAISS index of Mathlib declarations.

A cheap LLM drafts hypothetical lemmas — *formal* Lean signatures (to match a formal-signature
index) or *natural-language* descriptions (to match an informalization index) — which are
embedded with the same sentence-transformer used to build the index and searched against it by
cosine similarity. Results are fused across hypotheticals by **max cosine** and the top-k are
returned. No external search service: the index, id->declaration map, and embedder are loaded
once in the tool lifespan.

Runtime artifacts (configurable paths): a FAISS `IndexFlatIP` over L2-normalized embeddings, an
aligned `ids.npy`, and a SQLite DB mapping declaration id -> (name, signature). The embedder
model must match the one the index was built with.
"""

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..config import LLMConfig
from ..runtime import Runtime
from ..utils import get_logger
from ..utils.llm import LLMClient
from .registry import register_tool, tool_name_from_type

logger = get_logger(__name__)

HYDE_TOOL_TYPE = "search_hyde"

# Two prompt modes: the hypothetical must live in the SAME modality as the embedded corpus.
HYDE_PROMPT_FORMAL = (
    "You are helping find Mathlib lemmas to prove a Lean 4 goal.\n\n"
    "Goal:\n{q}\n\n"
    "Draft {n} DISTINCT hypothetical Mathlib lemmas that, if they existed, would directly help "
    "close this goal. Write each as a COMPLETE Lean 4 theorem signature — the binders AND the "
    "full proposition after the colon — NOT just a name. For example:\n"
    "theorem mul_comm (a b : ℕ) : a * b = b * a\n"
    "Use idiomatic Mathlib names and notation. It is fine if a lemma does not exist verbatim; "
    "we retrieve the closest real ones. Output only the signatures."
)
HYDE_PROMPT_INFORMAL = (
    "You are helping find Mathlib lemmas to prove a Lean 4 goal.\n\n"
    "Goal:\n{q}\n\n"
    "Draft {n} DISTINCT natural-language descriptions of Mathlib lemmas that, if they existed, "
    "would directly help close this goal. Each should read like a one-sentence mathematical "
    "statement of the lemma (as a textbook would phrase it), not Lean code and not a name. "
    "Output only the descriptions, one per line."
)


def _default_llm() -> dict:
    return {
        "model": "anthropic:claude-haiku-4-5-20251001",
        "provider_config": {"temperature": 0.7, "max_tokens": 512},
    }


@dataclass
class SearchHydeConfig:
    """Configuration for the embedding-based HyDE retrieval tool.

    ``embed_model`` MUST match the sentence-transformer the index was built with.
    ``max_results`` mirrors the baseline retriever so result count never skews comparisons.
    ``cosine_cutoff`` (default off) drops fused hits below an absolute cosine; calibration on the
    4B-signature index showed cosine does not cleanly separate used premises from noise, so it is
    disabled by default.

    Artifact paths (``index_path``/``ids_path``/``db_path``) are local file paths. If a path does
    not exist and ``hf_repo`` is set, the file is downloaded once from that HuggingFace **dataset**
    repo (by basename) and cached — so the multi-GB index need not live in the git repo.

    The hyperparameters here (``num_hypotheses``, ``retrieval_depth``, ``max_results``,
    ``cosine_cutoff``) were set by hand / mirrored from the LeanSearch baseline, **not swept** —
    they are a first guess, and tuning them is expected to move results.
    """

    index_path: str = ""
    ids_path: str = ""
    db_path: str = ""
    hf_repo: str | None = None  # optional HF dataset repo to fetch missing artifacts from
    embed_model: str = "Qwen/Qwen3-Embedding-4B"
    embed_device: str = "cpu"
    prompt_mode: str = "formal"  # "formal" | "informal"
    llm: dict = field(default_factory=_default_llm)
    num_hypotheses: int = 3
    include_raw_query: bool = True
    max_results: int = 6  # final count returned (mirror the LeanSearch baseline)
    retrieval_depth: int = 50  # per-hypothesis FAISS depth before fusion (internal)
    cosine_cutoff: float | None = None  # optional absolute-cosine floor; off by default


class HydeSearchInput(BaseModel):
    query: str = Field(..., description="The goal or a description of what to prove")


class _Hypotheticals(BaseModel):
    lemmas: list[str] = Field(
        default_factory=list,
        description="Hypothetical lemmas — complete Lean signatures (formal mode) or "
        "one-sentence descriptions (informal mode), per the prompt; never bare names.",
    )


@dataclass
class HydeResources:
    """Loaded once in the lifespan: FAISS index, id map, declaration metadata, embedder."""

    index: Any  # faiss.Index (IndexFlatIP over normalized vectors)
    row_name: list[str]  # index row -> declaration name
    name_block: dict[str, str]  # declaration name -> formatted display block (name + signature)
    embed: Any  # callable: list[str] -> np.ndarray (n, d), L2-normalized float32
    search_lock: asyncio.Semaphore  # serialize embed+FAISS (encode/FAISS are not thread-safe)


def _parse_hypotheticals(text: str, limit: int) -> list[str]:
    """Fallback parser (used only if structured output is unavailable): strip fences/bullets."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            continue
        s = re.sub(r"^[\s\-*•]*(?:\d+[.)]\s*)?", "", line).strip()
        if s:
            out.append(s)
    return out[:limit]


async def _generate_hypotheticals(query: str, client: LLMClient, config: SearchHydeConfig) -> list[str]:
    """Draft hypothetical lemmas in the configured modality; fall back to the raw query."""
    import json

    template = HYDE_PROMPT_INFORMAL if config.prompt_mode == "informal" else HYDE_PROMPT_FORMAL
    prompt = template.replace("{q}", query).replace("{n}", str(config.num_hypotheses))
    try:
        response = await client.ainvoke([HumanMessage(content=prompt)], output_schema=_Hypotheticals)
        lemmas = _Hypotheticals(**json.loads(response.text)).lemmas
        hyps = [h.strip() for h in lemmas if h.strip()][: config.num_hypotheses]
        if hyps:
            return hyps
    except Exception as e:
        logger.debug(f"HyDE structured generation failed ({type(e).__name__}: {e}); trying text")
    try:
        response = await client.ainvoke([HumanMessage(content=prompt)])
        return _parse_hypotheticals(response.text, config.num_hypotheses) or [query]
    except Exception as e:
        logger.warning(f"HyDE generation failed ({type(e).__name__}: {e}); falling back to raw query")
        return [query]


def _max_cosine_fuse(per_query_hits: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    """Fuse ranked (name, cosine) lists by MAX cosine across queries; rank by fused cosine.

    Unlike RRF this keeps the actual cosine values (so ranking is by cosine and an absolute
    cutoff is meaningful). A lemma's score is its best similarity to any hypothetical.
    """
    best: dict[str, float] = {}
    for hits in per_query_hits:
        for name, cos in hits:
            if cos > best.get(name, -2.0):
                best[name] = cos
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def _resolve_artifact(path: str, hf_repo: str | None) -> str:
    """Return a local path to ``path``, downloading it from the HF dataset repo if missing.

    Falls back to ``path`` unchanged if it exists locally or no ``hf_repo`` is configured (so the
    subsequent open raises a clear FileNotFoundError rather than this helper hiding the cause).
    """
    import os

    if os.path.exists(path) or not hf_repo:
        return path
    from huggingface_hub import hf_hub_download

    logger.info(f"HyDE: fetching {os.path.basename(path)} from HF dataset {hf_repo}")
    return hf_hub_download(repo_id=hf_repo, filename=os.path.basename(path), repo_type="dataset")


@asynccontextmanager
async def _hyde_lifespan(config: SearchHydeConfig) -> AsyncIterator[HydeResources | None]:
    """Load the FAISS index, id->declaration map, and embedder once.

    Yields None if optional deps (faiss / sentence-transformers) or the artifacts are missing,
    so the tool is skipped rather than aborting the run.
    """
    try:
        import sqlite3

        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger.warning(f"HyDE deps unavailable ({e}); tool disabled")
        yield None
        return

    try:
        index_path = await asyncio.to_thread(_resolve_artifact, config.index_path, config.hf_repo)
        ids_path = await asyncio.to_thread(_resolve_artifact, config.ids_path, config.hf_repo)
        db_path = await asyncio.to_thread(_resolve_artifact, config.db_path, config.hf_repo)
        logger.info(f"HyDE: loading FAISS index {index_path}")
        index = await asyncio.to_thread(faiss.read_index, index_path)
        ids = np.load(ids_path)
        con = sqlite3.connect(db_path)
        id_meta = {r[0]: (r[1], r[2]) for r in con.execute("SELECT id, name, source_text FROM declarations")}
        con.close()

        def _block(name: str, src: str) -> str:
            sig = (src or "").strip()
            for marker in (":=", "\n"):
                i = sig.find(marker)
                if i > 0:
                    sig = sig[:i]
                    break
            sig = sig.strip()[:600]
            return f"\n• {name}\n  {sig}" if sig else f"\n• {name}"

        row_name, name_block = [], {}
        for x in ids:
            name, src = id_meta.get(int(x), ("?", ""))
            row_name.append(name)
            name_block.setdefault(name, _block(name, src))

        logger.info(f"HyDE: loading embedder {config.embed_model} on {config.embed_device}")
        model = await asyncio.to_thread(
            SentenceTransformer, config.embed_model, device=config.embed_device, trust_remote_code=True
        )

        def embed(texts: list[str]):
            return model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")

        logger.info(f"HyDE ready: {index.ntotal} declarations indexed")
    except Exception as e:
        logger.warning(f"HyDE initialization failed: {e}")
        yield None
        return

    yield HydeResources(
        index=index, row_name=row_name, name_block=name_block, embed=embed,
        search_lock=asyncio.Semaphore(1),
    )


@register_tool(HYDE_TOOL_TYPE, SearchHydeConfig, _hyde_lifespan)
def create_search_hyde_tool(config: SearchHydeConfig, runtime: Runtime) -> StructuredTool | None:
    """Create the HyDE embedding-search tool, or None if its index/embedder failed to load."""
    res: HydeResources | None = runtime.get_tool_resources(HYDE_TOOL_TYPE)
    if res is None:
        return None

    client = LLMClient(LLMConfig(**dict(config.llm)))

    def _faiss_search(vecs) -> list[list[tuple[str, float]]]:
        sims, idx = res.index.search(vecs, config.retrieval_depth)
        return [[(res.row_name[i], float(s)) for s, i in zip(srow, irow)] for srow, irow in zip(sims, idx)]

    async def _search(query: str) -> str:
        hyps = await _generate_hypotheticals(query, client, config)
        queries = list(hyps) + ([query] if config.include_raw_query else [])
        async with res.search_lock:  # encode + FAISS are not concurrency-safe
            vecs = await asyncio.to_thread(res.embed, queries)
            per_query = _faiss_search(vecs)

        fused = _max_cosine_fuse(per_query)
        if config.cosine_cutoff is not None:
            fused = [(n, c) for n, c in fused if c >= config.cosine_cutoff]
        fused = fused[: config.max_results]
        if not fused:
            return f"No results found for: {query}"

        header = f"=== HyDE results for: {query} ({len(fused)} lemmas) ==="
        lines = [f"{res.name_block.get(n, chr(10) + '• ' + n)}  [cos={c:.3f}]" for n, c in fused]
        return "\n".join([header] + lines)

    return StructuredTool(
        name=tool_name_from_type(HYDE_TOOL_TYPE),
        description="""Find Mathlib lemmas via HyDE (hypothetical-lemma) embedding retrieval.

Best when you know WHAT you need but not the exact lemma name. Describe the goal or the kind of
fact you want; the tool drafts hypothetical lemmas, embeds them, and returns the closest real
Mathlib lemmas by semantic similarity, fused across several drafts.""",
        coroutine=_search,
        args_schema=HydeSearchInput,
    )
