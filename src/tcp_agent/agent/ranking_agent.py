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

import time
import operator

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from tcp_agent.agent.filter_agent import FilterResult
from tcp_agent.tools.history_tool import get_test_risk_profile
from tcp_agent.tools.complexity_tool import get_test_complexity
from tcp_agent.tools.covered_code_risk_tool import get_covered_code_risk


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

def _invoke_with_retry(model, messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(65)
            else:
                raise
    raise Exception(f"Still rate-limited after {max_retries} retries")


# ── Main ranking function ────────────────────────────────────────────

def run_ranking_agent(
    filter_result: FilterResult,
    dataset_path: str,
    ranking_model: str = "gpt-4o",
) -> list[dict]:
    """Run the Ranking Agent on the filtered high-risk tests.

    Parameters
    ----------
    filter_result : FilterResult
        Output from the Filter Agent.
    dataset_path : str
        Path to the CSV dataset (passed to tools).
    ranking_model : str
        LLM model name for the ranking agent (default gpt-4o).

    Returns
    -------
    list[dict]
        Full ranked list: high-risk tests with reasoning + T6 tail.
    """
    high_risk_ids = [t["test_id"] for t in filter_result.high_risk_tests]

    # ── Short-circuit: if no high-risk tests, return T6 tail only ────
    if not high_risk_ids:
        return _build_t6_tail(filter_result.low_signal_tests, start_priority=1)

    # ── Init model with tools ────────────────────────────────────────
    model = init_chat_model(ranking_model, temperature=0)

    tools = [get_test_risk_profile, get_test_complexity, get_covered_code_risk]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    # ── Build context message with filter results ────────────────────
    filter_summary_lines = [
        f"Pre-filtered {len(high_risk_ids)} high-risk tests (T1-T5) from "
        f"{filter_result.metadata.get('total_tests', '?')} total tests.",
        f"{filter_result.metadata.get('low_signal_count', '?')} T6 tests will be appended automatically.",
        "",
        "High-risk test IDs and preliminary classifications:",
    ]
    for t in filter_result.high_risk_tests:
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
            # inject test_ids filter if the tool supports it
            args = dict(tool_call["args"])
            if "test_ids" not in args:
                args["test_ids"] = high_risk_ids
            result = tool.invoke(args)
            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        return {"messages": tool_results}

    def should_continue(state: AgentState):
        if state["messages"][-1].tool_calls:
            return "call_tools"
        return END

    # ── Build graph ──────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("call_tools", call_tools)
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", should_continue, ["call_tools", END])
    graph.add_edge("call_tools", "call_llm")
    agent = graph.compile()

    # ── Run the agent ────────────────────────────────────────────────
    result = agent.invoke({
        "messages": [HumanMessage(
            content=(
                f"Rank these high-risk tests for the next build.\n"
                f"Dataset: {dataset_path}\n"
                f"High-risk test IDs (use these for tool calls): {high_risk_ids}\n\n"
                f"{context}"
            )
        )]
    })

    # ── Extract structured output ────────────────────────────────────
    final_content = result["messages"][-1].content
    structured_model = model.with_structured_output(PrioritizedTests)
    parsed = _invoke_with_retry(structured_model, [
        SystemMessage(
            content="Extract the test prioritization from this message into the schema. "
                    "Keep all reasons intact."
        ),
        HumanMessage(content=final_content),
    ])

    ranked = [t.model_dump() for t in parsed.ranked_tests]

    # ── Append T6 tail ───────────────────────────────────────────────
    next_priority = len(ranked) + 1
    t6_tail = _build_t6_tail(filter_result.low_signal_tests, start_priority=next_priority)
    ranked.extend(t6_tail)

    return ranked


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
