"""Unit tests for cross-run strategy memory (ledger distillation + preload)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ax_prover.config import StrategyMemoryConfig
from ax_prover.models import ProverAgentState, TargetItem
from ax_prover.models.files import Location
from ax_prover.models.messages import BuildFailedFeedback, ProposalMessage
from ax_prover.prover.strategy_memory import StrategyMemory


@pytest.fixture
def item():
    return TargetItem(location=Location(name="putnam_9999_z9", module_path="Test.Module"))


def failed_state(item, n_attempts=2):
    msgs = []
    for i in range(n_attempts):
        msgs.append(
            ProposalMessage(
                reasoning=f"try strategy {i}",
                location=item.location,
                imports=[],
                opens=[],
                code=f"theorem putnam_9999_z9 : True := by attempt{i}",
            )
        )
        msgs.append(BuildFailedFeedback(error_output=f"error {i}"))
    return ProverAgentState(item=item, messages=msgs, experience="within-run summary")


def make_sm(tmp_path, ledger_text="LEDGER v1", **cfg_kwargs):
    cfg = StrategyMemoryConfig(enabled=True, store_dir=str(tmp_path / "sm"), **cfg_kwargs)
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(text=ledger_text))
    return StrategyMemory(cfg, llm), llm


@pytest.mark.asyncio
async def test_failed_run_writes_ledger_and_next_run_preloads(tmp_path, item):
    sm, llm = make_sm(tmp_path)
    await sm.update(failed_state(item))
    assert sm.load("putnam_9999_z9") == "LEDGER v1"

    fresh = ProverAgentState(item=item)
    sm.preload_experience(fresh)
    assert "LEDGER v1" in fresh.experience
    assert "<experience>" in fresh.experience  # wrapped like normal experience


@pytest.mark.asyncio
async def test_distiller_receives_previous_ledger_attempts_and_summary(tmp_path, item):
    sm, llm = make_sm(tmp_path)
    (tmp_path / "sm").mkdir()
    sm._path(item.name).write_text("OLD LEDGER: strategy A dead")

    await sm.update(failed_state(item))

    sent = "".join(m.content for m in llm.ainvoke.call_args.args[0])
    assert "OLD LEDGER: strategy A dead" in sent  # merge input
    assert "try strategy 0" in sent and "attempt1" in sent  # attempts
    assert "within-run summary" in sent  # final self-summary
    assert "STRATEGY LEDGER" in sent  # system prompt present


@pytest.mark.asyncio
async def test_success_writes_resolved_marker_without_llm_call(tmp_path, item):
    from ax_prover.models.messages import ReviewApprovedFeedback

    sm, llm = make_sm(tmp_path)
    state = failed_state(item)
    state.messages.append(ReviewApprovedFeedback(review_comment="looks right"))
    assert state.approved
    await sm.update(state)
    llm.ainvoke.assert_not_called()
    assert "RESOLVED" in sm.load("putnam_9999_z9")


@pytest.mark.asyncio
async def test_ledger_is_capped_and_attempts_are_windowed(tmp_path, item):
    sm, llm = make_sm(tmp_path, ledger_text="X" * 99999, max_chars=500, max_attempts_in_prompt=3)
    await sm.update(failed_state(item, n_attempts=10))
    assert len(sm.load("putnam_9999_z9")) <= 501
    sent = "".join(m.content for m in llm.ainvoke.call_args.args[0])
    assert "attempt9" in sent and "attempt0" not in sent  # only most recent 3 attempts


@pytest.mark.asyncio
async def test_no_attempts_means_no_write(tmp_path, item):
    sm, llm = make_sm(tmp_path)
    await sm.update(ProverAgentState(item=item))  # no messages
    llm.ainvoke.assert_not_called()
    assert sm.load("putnam_9999_z9") is None


def test_disabled_by_default_and_agent_gate():
    assert StrategyMemoryConfig().enabled is False


@pytest.mark.asyncio
async def test_prove_single_item_uses_strategy_memory(tmp_path, item):
    """prove_single_item preloads before chat and updates after."""
    from ax_prover.utils.proving import prove_single_item

    sm, llm = make_sm(tmp_path)
    (tmp_path / "sm").mkdir()
    sm._path(item.name).write_text("PRELOADED LEDGER")

    seen = {}

    async def fake_chat(state, run_name=None, thread_id=None):
        seen["experience_at_start"] = state.experience
        return failed_state(item)

    prover = SimpleNamespace(strategy_memory=sm, chat=fake_chat)
    await prove_single_item(prover, item)

    assert "PRELOADED LEDGER" in seen["experience_at_start"]
    llm.ainvoke.assert_called_once()  # update ran on the failed final state


@pytest.mark.asyncio
async def test_prove_single_item_without_strategy_memory_is_unchanged(item):
    from ax_prover.utils.proving import prove_single_item

    async def fake_chat(state, run_name=None, thread_id=None):
        assert state.experience == ""  # untouched
        return ProverAgentState(item=item)

    prover = SimpleNamespace(strategy_memory=None, chat=fake_chat)
    await prove_single_item(prover, item)
