from tcp_agent.tools.history_tool import get_test_history, get_all_failure_rates, get_execution_times, get_test_risk_profile
from tcp_agent.tools.log_tool import get_failed_builds, get_build_failure_summary
from tcp_agent.tools.dependency_tool import get_high_coverage_tests
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.messages import AnyMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END

SYSTEM_PROMPT = """You are a CI/CD test prioritization agent for regression testing.

Your job is to rank tests so that failure-likely, high-value tests run first — maximizing
early fault detection while considering execution cost.

## Dataset Features

The dataset has ~150 features per test organized into these groups:

### Execution History (REC_ columns) — STRONGEST SIGNAL
These track each test's recent and lifetime behavior:
- REC_RecentFailRate / REC_TotalFailRate — how often the test fails (recent vs all-time)
- REC_LastVerdict — did it pass (0) or fail (1) last time?
- REC_RecentTransitionRate / REC_TotalTransitionRate — how often it flips pass/fail (flaky tests)
- REC_LastFailureAge — builds since last failure (low = recently broke)
- REC_Age — how long the test has existed
- REC_RecentAvgExeTime / REC_RecentMaxExeTime — recent execution cost
A value of -1 means "no data available" (e.g. new test with no history). Treat -1 as unknown, not as a real metric.

### Coverage Scores (COV_ columns) — STRONG SIGNAL
- COV_ChnScoreSum — how much recently changed code this test covers
- COV_ImpScoreSum — how much impacted (downstream) code this test covers
- COV_ChnCount / COV_ImpCount — number of changed/impacted files covered
Tests covering more changed code are more likely to catch new regressions.

### Test Complexity (TES_COM_ columns) — MODERATE SIGNAL
Code metrics of the test file itself:
- Cyclomatic complexity (TES_COM_MaxCyclomatic, TES_COM_SumCyclomatic)
- Size (TES_COM_CountLineCode, TES_COM_CountStmtExe)
- Nesting depth (TES_COM_MaxNesting)
More complex tests tend to be more failure-prone and cover more behavior.

### Test Churn (TES_CHN_ columns) — MODERATE SIGNAL
Recent changes to the test file:
- TES_CHN_LinesAdded / TES_CHN_LinesDeleted — recent edits
- TES_CHN_DMMSize / TES_CHN_DMMComplexity — design change metrics
Recently modified tests are more likely to fail (new assertions, refactored logic).

### Test Process (TES_PRO_ columns) — WEAK SIGNAL
Development activity on the test file:
- TES_PRO_CommitCount, TES_PRO_DistinctDevCount — how actively maintained
- TES_PRO_OwnersContribution — bus factor / ownership concentration

### Covered Code Metrics (COD_COV_*_C_ and COD_COV_*_IMP_ columns) — MODERATE SIGNAL
Same complexity/process/churn metrics but for the production code each test covers:
- _C_ = directly changed code, _IMP_ = impacted (downstream) code
Tests covering complex, recently-changed production code are higher priority.

### Fault Detection (DET_COV_ columns) — STRONG SIGNAL
- DET_COV_C_Faults — known faults in changed covered code
- DET_COV_IMP_Faults — known faults in impacted covered code
Non-zero values = the test covers code with known bugs. Prioritize these.

## Prioritization Strategy

Use this tiered approach — higher tiers always outrank lower tiers:

**Tier 1: Always-failing tests**
Tests with REC_TotalFailRate >= 0.9 or historical failure rate near 100%. These are
guaranteed failures — rank them first. Among them, prefer faster tests.

**Tier 2: Recently failing tests**
Tests with REC_RecentFailRate > 0 and REC_LastVerdict = 1 (failed last build).
Rank by REC_RecentFailRate descending, break ties with historical failure rate.
Among similar rates, prefer faster tests.

**Tier 3: High-risk tests with fault signals**
Tests with DET_COV_C_Faults > 0 or DET_COV_IMP_Faults > 0, especially those covering
recently changed code (COV_ChnScoreSum > 0). Rank by combined fault + coverage signal.

**Tier 4: Tests with historical failures but not recent**
Tests with non-zero historical failure rate but REC_RecentFailRate = 0.
They may have been fixed, but still worth running before zero-failure tests.

**Tier 5: Zero-failure safety net tests**
Never-failed tests ordered by: high coverage score > high fault coverage > low execution cost.
These catch new regressions in changed code.

**Tier 6: Remaining tests**
Everything else, ordered by execution cost (fastest first).

## Tool Usage Strategy

1. Start with get_test_risk_profile — this gives you the REC_, DET_COV_, and COV_ features
   for every test in one call. This is your richest data source.
2. Call get_all_failure_rates to cross-reference overall failure history
3. Call get_execution_times to factor in test cost for cost-aware ordering
4. Call get_high_coverage_tests to find high-value safety-net tests
5. Call get_failed_builds then get_build_failure_summary to check recent failure patterns
6. Use get_test_history only if you need to drill into a specific suspicious test

## Output Format

When you have enough information, output ONLY a JSON array ranking ALL tests.
Every test in the dataset must appear. No extra text outside the JSON.

[
  {
    "test": "test_id",
    "priority": 1,
    "confidence": 0.87,
    "reason": "short explanation referencing which signals drove this ranking"
  }
]
"""

#MessageState holds all previous messages and llm call amount
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def run_agent(dataset_path):
    model = init_chat_model(
        "claude-sonnet-4-6",
        temperature=0
    )

    tools = [get_test_history, get_all_failure_rates, get_failed_builds, get_build_failure_summary, get_high_coverage_tests, get_execution_times, get_test_risk_profile]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    def llm_call(state):
        return {
            "messages": [
                model_with_tools.invoke(
                    [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    def tool_node(state):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            t = tools_by_name[tool_call["name"]]
            observation = t.invoke(tool_call["args"])
            result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        return {"messages": result}

    def should_continue(state):
        # stop the agent if it's been looping too long
        if state.get("llm_calls", 0) > 10:
            return END
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        return END

    # build the graph
    agent = StateGraph(MessagesState)
    agent.add_node("llm_call", llm_call)
    agent.add_node("tool_node", tool_node)
    agent.add_edge(START, "llm_call")
    agent.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent.add_edge("tool_node", "llm_call")

    # compile and run
    compiled = agent.compile()
    from langchain.messages import HumanMessage
    result = compiled.invoke({"messages": [HumanMessage(content=f"Prioritize tests for the next build. The dataset is at: {dataset_path}")]})

    # parse the final ranking from Claude's last message
    import json
    raw = result["messages"][-1].content
    raw = raw.strip().replace("```json", "").replace("```", "")
    raw = raw[raw.find("["):raw.rfind("]") + 1]
    ranked = json.loads(raw)
    return ranked
