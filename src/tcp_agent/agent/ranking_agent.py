from __future__ import annotations

"""
Ranking Agent — "The Expert"

Takes the filtered high-risk tests (T1–T5) from the Filter Agent and
performs deep, research-grounded reasoning to produce the final ranked
output.  T6 tests are appended deterministically without LLM reasoning.

Design
------
1. Uses a LangGraph tool-calling loop (same pattern as the original agent)
   but only requests tool data for the high-risk subset (via test_ids filter).
2. Produces full 2-3 sentence justifications per test.
3. Appends T6 tail sorted by execution cost ascending.
"""

import operator
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from tcp_agent.agent.filter_agent import FilterResult
from tcp_agent.tools.history_tool import get_test_risk_profile
from tcp_agent.tools.complexity_tool import get_test_complexity
from tcp_agent.tools.covered_code_risk_tool import get_covered_code_risk


def _invoke_with_retry(model, messages, max_retries: int = 6):
    """Minimal retry: exponential backoff on transient errors, re-raise length/auth."""
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()
            if name == "LengthFinishReasonError" or "length limit" in msg:
                raise
            if "401" in str(e) or "403" in str(e) or "invalid_api_key" in msg:
                raise
            if attempt == max_retries - 1:
                raise
            wait = 65 if "rate" in msg or "429" in str(e) else min(2 ** attempt, 30)
            print(f"  [ranking-RETRY] {attempt + 1}/{max_retries} {name}: {str(e)[:120]}", flush=True)
            time.sleep(wait)


# ── Structured output schema ─────────────────────────────────────────

class RankedTest(BaseModel):
    test: str = Field(description="Test ID")
    priority: int = Field(description="1 = run first")
    confidence: float = Field(description="0.0–1.0")
    reason: str = Field(description="2-3 sentence explanation citing tier and feature values")


class PrioritizedTests(BaseModel):
    ranked_tests: list[RankedTest] = Field(description="All tests, ordered by priority")


# ── Agent state ──────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# ── System prompt (ranking-focused) ──────────────────────────────────

RANKING_SYSTEM_PROMPT = """\
You are a test case prioritization (TCP) expert for Continuous Integration.
You have been given a PRE-FILTERED list of HIGH-RISK tests (Tier 1-5).
Low-signal (T6) tests have already been separated and will be appended automatically.

Your task: rank ONLY these high-risk tests so failures are detected as early
and cheaply as possible, maximizing APFDc.

## Research Foundation (TCP-CI, Yaraghi et al. 2022)

1. **Feature importance:** REC (history) >> TES (test metrics) >> COV/COD_COV (coverage).
2. **REC features alone achieve near-full-model performance** (CL=0.51 vs Full_M).
3. **Optimal ordering:** Failed tests first, then by execution time ascending.
4. **Class imbalance:** Only ~3% of test executions fail (Mendoza et al., 2022).

## Ranking Rules (strict priority order)

**T1 — Persistent failures:** REC_TotalFailRate ≥ 0.9. Sort by cost ascending.

**T2 — Recent/active failures:** REC_RecentFailRate > 0 AND REC_LastVerdict = 1.
Sort by REC_RecentFailRate desc, then cost ascending. Lower REC_LastFailureAge = worse.

**T3 — Fault-adjacent:** DET_COV_C_Faults > 0 OR DET_COV_IMP_Faults > 0.
Prefer those also covering recently changed code (COV_ChnScoreSum > 0).

**T4 — Historical failures, currently passing:** failure_rate > 0 but
REC_RecentFailRate = 0. Sort by failure_rate desc, factor in TES_PRO_OwnersExperience.

**T5 — High-signal never-failed:** COV_ChnScoreSum > 0, high complexity,
low owner experience, or high covered-code complexity. Sort by coverage desc, cost ascending.

## Tool Call Strategy

You have access to tools that return detailed features for ONLY the high-risk tests.
Call get_test_risk_profile first, then get_test_complexity and get_covered_code_risk
together.  All tools accept test_ids to filter for only the relevant tests.

## Output (CRITICAL)

Your FINAL message must contain ONLY a JSON array (no markdown, no text before/after).
Every high-risk test must appear exactly once.

For each **reason**, write 2-3 sentences that:
1. State which tier (T1-T5), what the tier means, and WHY the test belongs there.
2. Name the specific feature values AND explain what they mean in plain English.
3. Explain tie-breaking logic if relevant.

Good example:
"Tier 1 (Persistent failure): This test fails in 100% of all builds (REC_TotalFailRate=1.0) — it is a guaranteed, persistent failure. At only 31ms average execution time (REC_RecentAvgExeTime), it is the cheapest T1 test, so it catches a fault with almost zero CI cost."

Bad (too vague): "High failure rate, placing first."

[{\"test\":\"id\",\"priority\":1,\"confidence\":0.9,\"reason\":\"...\"}]
"""


# ── LLM call with rate-limit retry ───────────────────────────────────

def _resolve_provider(model_name: str) -> str | None:
    name = model_name.lower()
    if "gemini" in name:
        return "google_genai"
    if name.startswith("claude"):
        return "anthropic"
    return None


def _build_models(model_name: str, tools: list):
    """Initialize a chat model and return (tools-bound, structured-output) pair."""
    provider = _resolve_provider(model_name)
    base = init_chat_model(model_name, model_provider=provider, temperature=0)
    return base.bind_tools(tools), base.with_structured_output(PrioritizedTests)


# ── Main ranking function ────────────────────────────────────────────

def _chunk_list(lst: list, size: int) -> list[list]:
    """Split a list into chunks of the given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


# Max tests per ranking batch — keeps tool output within GPT-4o's 128K context.
# 3 tools × ~30 features × 30 tests ≈ 25K tokens, well within limits.
# Going larger trades a bit of LLM ranking-precision risk for fewer round trips.
_RANKING_BATCH_SIZE = 15

# How many ranking batches to run concurrently across threads.  Each batch is
# an independent LangGraph instance with its own model/tools, so it's safe to
# parallelize.  Limit set to keep token-rate within OpenAI Tier 1+ TPM bands;
# raise to 6-8 on higher tiers, lower to 2 on rate-limited providers.
_RANKING_PARALLELISM = 4


def _rank_batch(
    batch_tests: list[dict],
    batch_ids: list[int],
    dataset_path: str,
    ranking_model: str,
    total_high_risk: int,
    total_tests: int,
    low_signal_count: int,
) -> list[dict]:
    """Rank a single batch of high-risk tests via the LLM tool-calling loop."""

    tools = [get_test_risk_profile, get_test_complexity, get_covered_code_risk]
    tools_by_name = {t.name: t for t in tools}

    model_with_tools, structured_model = _build_models(ranking_model, tools)

    # ── Build context message with filter results ────────────────────
    filter_summary_lines = [
        f"Ranking batch of {len(batch_ids)} high-risk tests (T1-T5) "
        f"(out of {total_high_risk} total high-risk from {total_tests} tests).",
        f"{low_signal_count} T6 tests will be appended automatically.",
        "",
        "High-risk test IDs and preliminary classifications:",
    ]
    for t in batch_tests:
        filter_summary_lines.append(
            f"  Test {t['test_id']}: preliminary T{t['tier']} — {', '.join(t['key_signals'])}"
        )

    context = "\n".join(filter_summary_lines)

    # ── Graph nodes ──────────────────────────────────────────────────

    def call_llm(state: AgentState):
        msgs = [SystemMessage(content=RANKING_SYSTEM_PROMPT)] + state["messages"]
        response = _invoke_with_retry(model_with_tools, msgs)
        return {"messages": [response]}

    def call_tools(state: AgentState):
        tool_results = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            args = dict(tool_call["args"])
            if "test_ids" not in args:
                args["test_ids"] = batch_ids
            result = tool.invoke(args)
            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        return {"messages": tool_results}

    def should_continue(state: AgentState):
        if state["messages"][-1].tool_calls:
            return "call_tools"
        return END

    # ── Build & run graph ────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("call_tools", call_tools)
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", should_continue, ["call_tools", END])
    graph.add_edge("call_tools", "call_llm")
    agent = graph.compile()

    result = agent.invoke({
        "messages": [HumanMessage(
            content=(
                f"Rank these high-risk tests for the next build.\n"
                f"Dataset: {dataset_path}\n"
                f"High-risk test IDs (use these for tool calls): {batch_ids}\n\n"
                f"{context}"
            )
        )]
    })

    # ── Extract structured output ────────────────────────────────────
    final_content = result["messages"][-1].content
    parsed = _invoke_with_retry(
        structured_model,
        [
            SystemMessage(
                content="Extract the test prioritization from this message into the schema. "
                        "Keep all reasons intact."
            ),
            HumanMessage(content=final_content),
        ],
    )

    return [t.model_dump() for t in parsed.ranked_tests]


def run_ranking_agent(
    filter_result: FilterResult,
    dataset_path: str,
    ranking_model: str = "gpt-4o",
) -> list[dict]:
    """Run the Ranking Agent on the filtered high-risk tests.

    Splits high-risk tests into batches to stay within model context limits.
    Each batch is ranked independently, then results are merged by batch
    order to produce the final ordering. Returns the full ranked list:
    high-risk tests with reasoning + T6 tail.
    """
    high_risk_tests = filter_result.high_risk_tests
    high_risk_ids = [t["test_id"] for t in high_risk_tests]

    # ── Short-circuit: if no high-risk tests, return T6 tail only ────
    if not high_risk_ids:
        return _build_t6_tail(filter_result.low_signal_tests, start_priority=1)

    total_tests = filter_result.metadata.get("total_tests", len(high_risk_ids))
    low_signal_count = filter_result.metadata.get("low_signal_count", 0)

    # ── Batch the high-risk tests ────────────────────────────────────
    test_batches = _chunk_list(high_risk_tests, _RANKING_BATCH_SIZE)
    id_batches = _chunk_list(high_risk_ids, _RANKING_BATCH_SIZE)

    print(
        f"  [RANKING] {len(high_risk_ids)} high-risk tests → {len(test_batches)} "
        f"batches of ≤{_RANKING_BATCH_SIZE} (parallelism={_RANKING_PARALLELISM})"
    )

    def _run_one_batch(batch_idx: int, batch_tests, batch_ids):
        """Worker: rank a single batch and tag every result with its batch_idx."""
        print(f"  [RANKING] Batch {batch_idx+1}/{len(test_batches)} ({len(batch_ids)} tests) starting...")
        batch_ranked = _rank_batch(
            batch_tests=batch_tests,
            batch_ids=batch_ids,
            dataset_path=dataset_path,
            ranking_model=ranking_model,
            total_high_risk=len(high_risk_ids),
            total_tests=total_tests,
            low_signal_count=low_signal_count,
        )
        for item in batch_ranked:
            item["_batch_idx"] = batch_idx
        print(f"  [RANKING] Batch {batch_idx+1}/{len(test_batches)} done ({len(batch_ranked)} ranked)")
        return batch_ranked

    all_ranked = []
    if len(test_batches) == 1 or _RANKING_PARALLELISM <= 1:
        # Avoid thread overhead for the trivial case
        for batch_idx, (batch_tests, batch_ids) in enumerate(zip(test_batches, id_batches)):
            all_ranked.extend(_run_one_batch(batch_idx, batch_tests, batch_ids))
    else:
        with ThreadPoolExecutor(max_workers=_RANKING_PARALLELISM) as pool:
            futures = {
                pool.submit(_run_one_batch, idx, bt, bi): idx
                for idx, (bt, bi) in enumerate(zip(test_batches, id_batches))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    all_ranked.extend(fut.result())
                except Exception as e:
                    # One failed batch shouldn't kill the whole dataset — log and
                    # continue.  These tests will be missing from the ranking and
                    # will be appended at max-priority by build_ranked_df.
                    print(f"  [RANKING] Batch {idx+1} FAILED: {type(e).__name__}: {e}")

    # ── Merge: trust the LLM's structured `priority` field within each batch,
    #    and preserve batch order across batches.  Earlier batches contain
    #    higher-failure-rate tests (feature_extractor sorts risk_profiles by
    #    REC_RecentFailRate desc before chunking), so batch order is meaningful.
    def _sort_key(item):
        return (item["_batch_idx"], item.get("priority", 10**6))

    all_ranked.sort(key=_sort_key)
    for item in all_ranked:
        item.pop("_batch_idx", None)

    # ── Recover any high-risk tests the LLM dropped ──────────────────
    # gpt-4o-mini occasionally truncates JSON arrays — append missing tests
    # before validation so the LLM's good rankings aren't discarded just
    # because a few items got dropped from a long structured-output list.
    returned_ids = {str(item["test"]) for item in all_ranked}
    missing = [t for t in high_risk_tests if str(t["test_id"]) not in returned_ids]
    if missing:
        print(
            f"  [RANKING] LLM dropped {len(missing)} high-risk test(s); "
            f"appending with conservative fallback reason."
        )
        for t in missing:
            all_ranked.append({
                "test": str(t["test_id"]),
                "priority": 0,  # re-numbered below
                "confidence": 0.4,
                "reason": (
                    f"Tier {t.get('tier', 5)} (LLM-dropped recovery): The Filter Agent "
                    f"flagged this as high-risk but the Ranking Agent omitted it from its "
                    f"output. Appended at the end of the high-risk section. "
                    f"Filter signals: {', '.join(t.get('key_signals', []))}."
                ),
            })

    # ── Re-number priorities sequentially ────────────────────────────
    for i, item in enumerate(all_ranked):
        item["priority"] = i + 1

    # ── Append T6 tail ───────────────────────────────────────────────
    next_priority = len(all_ranked) + 1
    t6_tail = _build_t6_tail(filter_result.low_signal_tests, start_priority=next_priority)
    all_ranked.extend(t6_tail)

    return all_ranked


def _build_t6_tail(low_signal_tests: list[dict], start_priority: int) -> list[dict]:
    """Build the deterministic T6 tail sorted by execution cost ascending."""
    sorted_t6 = sorted(low_signal_tests, key=lambda t: t.get("avg_exec_time", 0.0))

    tail = []
    for i, t in enumerate(sorted_t6):
        avg_time = t.get("avg_exec_time", 0.0)
        tail.append({
            "test": str(t["test_id"]),
            "priority": start_priority + i,
            "confidence": 0.1,
            "reason": (
                f"Tier 6 (Low-signal): No failure history or risk indicators. "
                f"Sorted by execution cost ({avg_time:.1f}ms). "
                f"Per Mendoza et al., ~97% of tests never fail — "
                f"minimizing CI time is optimal for zero-signal tests."
            ),
        })
    return tail
