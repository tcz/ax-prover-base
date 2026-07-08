#!/usr/bin/env bash
# Budget-lottery experiment, stage 1 (CLEAN traced restart): putnam_1972_b6.
# Sequential Opus tickets (25-iteration cap, fresh strategy ledger), TRACING ON.
# All tickets trace to LangSmith project budget_1972_b6. Resets PutnamBench before every ticket.
set -uo pipefail
cd /root/ax-prover-sm
eval "$(grep -E '^export (ANTHROPIC_API_KEY|LANGSMITH_API_KEY|TAVILY_API_KEY|HF_TOKEN)=' ~/.bashrc)"
# THE FIX: enable LangChain/LangSmith tracing for the `prove` path (experiment traces via
# aevaluate; prove does not, so it needs the flag).
export LANGSMITH_TRACING_V2=true
export LANGCHAIN_TRACING_V2=true
export LANGSMITH_PROJECT=budget_1972_b6
export LANGCHAIN_PROJECT=budget_1972_b6

LOG=/root/ax-prover-base/scratchpad/budget_run/stage1.log
STATUS=/root/ax-prover-base/scratchpad/budget_run/stage1.status
MAX_TICKETS=20
PROBLEM="src/putnam_1972_b6.lean:putnam_1972_b6"

echo "stage1 running (traced, max $MAX_TICKETS tickets)" > "$STATUS"
for i in $(seq 1 $MAX_TICKETS); do
  # reset the Lean directory before every ticket
  (cd /root/PutnamBench/lean4 && git checkout -- src/ && rm -f src/tmp_putnam_*.lean)
  echo "=== TICKET $i/$MAX_TICKETS start $(date -u) ===" >> "$LOG"
  timeout 9000 .venv/bin/ax-prover --config budget_stage1.yaml prove "$PROBLEM" \
      --folder /root/PutnamBench/lean4 --skip-build --overwrite >> "$LOG" 2>&1
  rc=$?
  echo "=== TICKET $i end rc=$rc $(date -u) ===" >> "$LOG"
  if grep -q "Successfully proved src.putnam_1972_b6" "$LOG"; then
    echo "stage1 SUCCESS at ticket $i" > "$STATUS"
    echo "=== STAGE1 SUCCESS at ticket $i ===" >> "$LOG"
    exit 0
  fi
done
echo "stage1 EXHAUSTED ($MAX_TICKETS tickets, no proof)" > "$STATUS"
echo "=== STAGE1 EXHAUSTED ===" >> "$LOG"
(cd /root/PutnamBench/lean4 && git checkout -- src/ && rm -f src/tmp_putnam_*.lean)
