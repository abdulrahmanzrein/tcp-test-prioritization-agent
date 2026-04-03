import json
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL",
    category=Warning,
)

from tcp_agent.tools.history_tool import get_all_failure_rates, get_test_risk_profile
from tcp_agent.tools.complexity_tool import get_test_complexity
from tcp_agent.tools.covered_code_risk_tool import get_covered_code_risk
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

SYSTEM_PROMPT = """You are a test case prioritization (TCP) agent for Continuous Integration.
Your task: rank regression tests so failures are detected as early and cheaply as possible,
maximizing APFDc — the cost-cognizant Average Percentage of Faults Detected.

## Research Foundation

This agent is built on the TCP-CI framework (Yaraghi et al., 2022), which established
a dataset of 150+ features across 25 open-source Java projects. The key findings that
drive our ranking strategy:

1. **Feature importance hierarchy (Table 12, Yaraghi et al.):**
   REC (execution history) >> TES (test file metrics) >> COV/COD_COV (coverage).
   - REC_Age (how many builds a test has existed) is the #1 most-used feature in
     Random Forest models (usage frequency 17,034). Older tests have more signal.
   - TES_PRO_OwnersExperience (#2, freq 10,618) and TES_PRO_AllCommitersExperience
     (#3, freq 7,643) — tests written by less experienced developers fail more.
   - REC_RecentFailRate and REC_TotalFailRate are the strongest direct predictors
     of future failure.

2. **REC features alone achieve near-full-model performance (RQ2.3):**
   REC_M (19 features) vs Full_M (150+ features) has CL=0.51, meaning the full model
   wins only ~51% of comparisons — barely better than a coin flip. This validates
   our REC-heavy ranking approach.

3. **Coverage features have lowest marginal value (RQ2.3, Table 10-11):**
   COV_M alone has CL=0.79 vs Full_M — it loses ~79% of the time. Coverage is useful
   as a tiebreaker, not a primary signal.

4. **Optimal test ordering (Section 2.1, Definition):** Tests should be sorted by
   verdict (failed first), then by execution time ascending (cheapest first among
   same-verdict tests). This minimizes wasted CI time.

5. **Class imbalance (Mendoza et al., 2022):** Only ~3% of test executions fail in
   typical CI projects. This extreme imbalance means the agent must be aggressive about
   ranking ANY test with failure signal above the zero-signal majority.

6. **Coverage scores use association-rule mining (Section 2.2):** COV_ChnScoreSum and
   COV_ImpScoreSum are NOT traditional code coverage — they measure the confidence of
   co-change/co-impact associations between test files and production files. Higher
   scores mean the test historically changes alongside production code that was modified.

7. **DET_COV = Previously Detected Faults (Section 2.2, Definition 2):** DET_COV_C_Faults
   counts how many known faults exist in the production code that a test directly covers.
   DET_COV_IMP_Faults counts faults in downstream-impacted code. These are strong signals
   because buggy code tends to stay buggy.

## Available Tools → Feature Groups

1. **get_test_risk_profile** → REC_ (19) + DET_COV_ (2) + COV_ (4) + TES_CHN_ (7) features:
   - REC_Age: builds since test first appeared (top predictor — older tests = more history)
   - REC_RecentFailRate, REC_TotalFailRate: failure rate over last 6 builds / all builds
   - REC_LastVerdict: 0=passed, 1=failed in most recent build
   - REC_LastFailureAge: builds since last failure (low = recently broke)
   - REC_RecentTransitionRate, REC_TotalTransitionRate: pass↔fail flip rate (flakiness)
   - REC_RecentAvgExeTime, REC_RecentMaxExeTime: test execution cost
   - DET_COV_C_Faults, DET_COV_IMP_Faults: known faults in covered production code
   - COV_ChnScoreSum, COV_ImpScoreSum: association-rule coverage of changed/impacted code
   - TES_CHN_*: recent edits to the test file itself (lines added/deleted, DMM metrics)
   - A value of -1 means "no data" — treat as unknown, not a real metric.

2. **get_all_failure_rates** → per-test overall failure_rate (0.0–1.0) across all builds.

3. **get_test_complexity** → TES_COM_ (31) + TES_PRO_ (6) features:
   - TES_COM_SumCyclomatic, MaxNesting, CountLineCode: test code complexity
   - TES_PRO_OwnersExperience: how much of the test the primary author wrote (low = risky)
   - TES_PRO_AllCommitersExperience: total experience of all contributors
   - TES_PRO_DistinctDevCount, CommitCount: ownership diffusion and churn
   - Per Yaraghi Table 12: OwnersExperience is the #2 most important feature overall.

4. **get_covered_code_risk** → COD_COV_ (81) features:
   - Complexity/churn/process metrics of the production code each test covers.
   - COD_COV_COM_C_SumCyclomatic: cyclomatic complexity of changed covered code
   - COD_COV_CHN_C_LinesAdded: recent churn in covered production code
   - COD_COV_PRO_C_DistinctDevCount: how many developers touched the covered code
   - Lowest marginal value individually (CL=0.79), but useful for tiebreaking in T5-T6.

## Ranking Rules (strict priority order)

Apply these tiers mechanically. The ordering within each tier follows the optimal
ordering from Yaraghi et al.: failed tests first, then by execution time ascending.

**T1 — Persistent failures:** REC_TotalFailRate ≥ 0.9. These tests fail in nearly every
build — they are near-guaranteed failures. Sort by execution time ascending (cheapest first)
to catch faults with minimal CI cost.

**T2 — Recent/active failures:** REC_RecentFailRate > 0 AND REC_LastVerdict = 1. The test
failed recently AND in the last build — it's actively broken. Sort by REC_RecentFailRate desc,
then execution time ascending. Weight REC_LastFailureAge: lower age = failed more recently.

**T3 — Fault-adjacent tests:** DET_COV_C_Faults > 0 OR DET_COV_IMP_Faults > 0. These tests
cover production code with known bugs (Previously Detected Faults). Prefer those also covering
recently changed code (COV_ChnScoreSum > 0), since changed buggy code is highest risk.

**T4 — Historical failures, currently passing:** failure_rate > 0 but REC_RecentFailRate = 0.
The test has failed before but passed recently. Sort by failure_rate desc, then factor in
TES_PRO_OwnersExperience (low experience = higher risk, per Yaraghi Table 12).

**T5 — High-signal never-failed tests:** Never failed, but has risk indicators:
COV_ChnScoreSum > 0 (covers changed code), high TES_COM_SumCyclomatic (complex test),
low TES_PRO_OwnersExperience (inexperienced owner), or high COD_COV_COM_C_SumCyclomatic
(covers complex production code). Sort by coverage score desc, then cost ascending.

**T6 — Low-signal remainder:** No failure history, no coverage of changed code, no complexity
flags. Sort by execution time ascending (cheapest first) — in the ~97% of tests that never
fail (Mendoza et al.), minimizing wasted CI time is the best we can do.

## Tool Call Strategy

Call get_test_risk_profile and get_all_failure_rates together first (parallel).
Then call get_test_complexity and get_covered_code_risk together (parallel).
This gives you all 150+ features in 2 rounds — matching the Full_M feature set from
Yaraghi et al. Do NOT call tools beyond these 2 rounds.

## Output (CRITICAL)

Your FINAL message must contain ONLY a JSON array (no markdown, no text before/after).
Every test must appear exactly once.

For each **reason**, write 2-3 sentences that:
1. State which tier (T1-T6), what the tier means, and WHY the test belongs there.
2. Name the specific feature values AND explain what they mean in plain English
   (e.g. "REC_TotalFailRate=1.0 means this test fails in 100% of all recorded builds").
3. Explain tie-breaking logic if relevant (execution cost, coverage, complexity, experience).

Good example:
"Tier 1 (Persistent failure): This test fails in 100% of all builds (REC_TotalFailRate=1.0) — it is a guaranteed, persistent failure. At only 31ms average execution time (REC_RecentAvgExeTime), it is the cheapest T1 test, so it catches a fault with almost zero CI cost. Per Yaraghi et al., the optimal ordering places failed tests first, sorted by cost ascending."

Bad (too vague): "High failure rate, placing first."
Bad (no explanation): "REC_TotalFailRate=1.0 and fast."

[{"test":"id","priority":1,"confidence":0.9,"reason":"..."}]
"""

# ── Structured output schema ──────────────────────────────────────────
# Forces the final LLM call to return exactly this shape — no JSON parsing hacks needed.

from pydantic import BaseModel, Field

class RankedTest(BaseModel):
    test: str = Field(description="Test ID")
    priority: int = Field(description="1 = run first")
    confidence: float = Field(description="0.0–1.0")
    reason: str = Field(description="2-3 sentence explanation citing tier and feature values")

class PrioritizedTests(BaseModel):
    ranked_tests: list[RankedTest] = Field(description="All tests, ordered by priority")


# ── Agent state ───────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


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


# ── Main agent function ──────────────────────────────────────────────

def run_agent(dataset_path):
    model = init_chat_model(
        "claude-haiku-4-5-20251001",
        temperature=0,
        max_tokens=16384,
    )

    tools = [get_all_failure_rates, get_test_risk_profile, get_test_complexity, get_covered_code_risk]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    # ── Graph nodes ───────────────────────────────────────────────────

    def call_llm(state: AgentState):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = _invoke_with_retry(model_with_tools, msgs)
        return {"messages": [response]}

    def call_tools(state: AgentState):
        tool_results = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        return {"messages": tool_results}

    def should_continue(state: AgentState):
        if state["messages"][-1].tool_calls:
            return "call_tools"
        return END

    # ── Build graph ───────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("call_tools", call_tools)
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", should_continue, ["call_tools", END])
    graph.add_edge("call_tools", "call_llm")
    agent = graph.compile()

    # ── Run the agent (tool-calling phase) ────────────────────────────

    result = agent.invoke({
        "messages": [HumanMessage(
            content=f"Prioritize tests for the next build. The dataset is at: {dataset_path}"
        )]
    })

    # ── Extract structured output (ranking phase) ─────────────────────
    # The agent's last message contains the ranking. We ask a structured-output
    # model to parse it, so we never manually parse JSON.

    final_content = result["messages"][-1].content
    structured_model = model.with_structured_output(PrioritizedTests)
    parsed = _invoke_with_retry(structured_model, [
        SystemMessage(content="Extract the test prioritization from this message into the schema. Keep all reasons intact."),
        HumanMessage(content=final_content),
    ])

    return [t.model_dump() for t in parsed.ranked_tests]
