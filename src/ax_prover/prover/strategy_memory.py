"""Cross-run strategy memory: distill failed runs into a per-theorem strategy ledger.

The within-run `ExperienceProcessor` intentionally preserves tactical lessons for the
attempt in flight, which anchors the proposer to its current proof skeleton. Across runs
that anchoring is harmful: independent restarts rediscover and re-drown in the same
strategies. This module keeps the complementary memory: after a failed run, an LLM
distills the attempt history into a STRATEGY-level ledger (framings tried, where each
died, what is untried), persisted per theorem. A new run on the same theorem starts with
its `experience` prepopulated from the ledger, so restarts explore instead of resample.

The ledger deliberately excludes tactic-level and lemma-level detail: those are the
anchoring poison the restart is meant to escape.
"""

import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import StrategyMemoryConfig
from ..models.messages import FeedbackMessage, ProposalMessage
from ..models.proving import ProverAgentState
from ..utils import get_logger
from ..utils.llm import LLMClient
from .memory import PROPOSER_EXPERIENCE_USER_PROMPT

logger = get_logger(__name__)

DISTILLER_SYSTEM_PROMPT = """You maintain a STRATEGY LEDGER for an LLM agent that repeatedly attempts a Lean 4 theorem across independent runs. After each failed run you receive: the previous ledger (possibly empty), a sample of the run's attempts (reasoning, code, feedback), and the run's final self-summary. You write the updated ledger.

The ledger's purpose is to make the NEXT run explore differently, not to help it patch the last proof. Rules:

1. Organize by PROOF STRATEGY (the mathematical framing, e.g. "quadratic discriminant argument on k", "case comparison of n with m(m+1)", "induction on n", "direct construction via geometric series"). For each strategy record: how many runs tried it (cumulative), the furthest it got, and the concrete reason it stalled (e.g. "case bookkeeping for even/odd m never closed", "key inequality resisted linear arithmetic").
2. Give a VERDICT per strategy: DEAD END (tried thoroughly, structural blocker), STALLED (failed on execution, possibly viable), or UNTRIED.
3. End with 2-4 specific UNTRIED DIRECTIONS: alternative mathematical framings, not tactic advice.
4. STRICTLY FORBIDDEN anywhere in the ledger, including blocker descriptions: tactic names, lemma names, code snippets, error-message minutiae, line-level fixes. Describe blockers in mathematical terms ("the floor characterization resisted", not lemma identifiers). Strategy level only; this is the difference between this ledger and the within-run notes.
5. Merge with the previous ledger: update counts, never lose a strategy entry, revise verdicts on new evidence.
6. Stay under {max_chars} characters. Be direct; the reader is the prover itself.

Output only the ledger text."""

DISTILLER_USER_PROMPT = """<previous-ledger>
{previous}
</previous-ledger>

<run-attempts>
{attempts}
</run-attempts>

<run-final-summary>
{summary}
</run-final-summary>"""

ATTEMPT_SNIPPET = """<attempt n="{i}">
<reasoning>{reasoning}</reasoning>
<code>{code}</code>
<feedback>{feedback}</feedback>
</attempt>"""


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


class StrategyMemory:
    """Per-theorem cross-run strategy ledger, distilled and consumed automatically."""

    def __init__(self, config: StrategyMemoryConfig, llm_client: LLMClient):
        self.config = config
        self.llm = llm_client
        self.store = Path(config.store_dir)

    def _path(self, item_name: str) -> Path:
        return self.store / f"{_slug(item_name)}.md"

    def load(self, item_name: str) -> str | None:
        """Return the ledger for this theorem, or None if absent."""
        p = self._path(item_name)
        if p.exists():
            text = p.read_text().strip()
            return text or None
        return None

    def preload_experience(self, state: ProverAgentState) -> None:
        """Prepopulate the run's experience with the stored ledger, if any."""
        ledger = self.load(state.item.name)
        if ledger:
            state.experience = PROPOSER_EXPERIENCE_USER_PROMPT.format(experience=ledger)
            logger.info(
                f"Strategy memory: preloaded ledger for {state.item.name} ({len(ledger)} chars)"
            )

    def _format_attempts(self, state: ProverAgentState) -> str:
        """Pair proposals with their feedback, most recent last, size-capped."""
        pairs = []
        proposal = None
        for msg in state.messages:
            if isinstance(msg, ProposalMessage):
                proposal = msg
            elif isinstance(msg, FeedbackMessage) and proposal is not None:
                pairs.append((proposal, msg))
                proposal = None
        pairs = pairs[-self.config.max_attempts_in_prompt :]
        return "\n".join(
            ATTEMPT_SNIPPET.format(
                i=i + 1,
                reasoning=(p.reasoning or "")[:400],
                code=(p.code or "")[:1200],
                feedback=(f.content or "")[:400],
            )
            for i, (p, f) in enumerate(pairs)
        )

    async def update(self, state: ProverAgentState) -> None:
        """After a run, update the ledger (LLM distillation on failure, marker on success)."""
        name = state.item.name
        self.store.mkdir(parents=True, exist_ok=True)
        if state.approved:
            # Success needs no distillation; record it so future readers know.
            prev = self.load(name) or ""
            self._path(name).write_text(
                (prev + "\n\n" if prev else "") + "RESOLVED: a later run proved this theorem.\n"
            )
            return
        attempts = self._format_attempts(state)
        if not attempts:
            logger.warning(f"Strategy memory: no attempts to distill for {name}")
            return
        prompt = DISTILLER_USER_PROMPT.format(
            previous=self.load(name) or "(empty: this was the first recorded run)",
            attempts=attempts,
            summary=(state.experience or "")[: self.config.max_chars],
        )
        system = DISTILLER_SYSTEM_PROMPT.replace("{max_chars}", str(self.config.max_chars))
        response = await self.llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        ledger = response.text.strip()[: self.config.max_chars]
        if ledger:
            self._path(name).write_text(ledger + "\n")
            logger.info(f"Strategy memory: updated ledger for {name} ({len(ledger)} chars)")
