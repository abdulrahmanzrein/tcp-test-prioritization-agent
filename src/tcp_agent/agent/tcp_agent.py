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

SYSTEM_PROMPT = """You are a test prioritization agent. Rank tests so failures are found first.

## Available Tools → What They Return

1. **get_test_risk_profile** → per-test latest-build snapshot (CALL FIRST):
   - REC_RecentFailRate, REC_TotalFailRate (0-1, strongest signals)
   - REC_LastVerdict (0=pass, 1=fail last build)
   - REC_LastFailureAge (builds since last failure; low = recently broke)
   - REC_RecentTransitionRate, REC_TotalTransitionRate (flakiness)
   - REC_RecentAvgExeTime, REC_RecentMaxExeTime (test cost)
   - DET_COV_C_Faults, DET_COV_IMP_Faults (>0 = covers known buggy code)
   - COV_ChnScoreSum, COV_ImpScoreSum (coverage of changed/impacted code)
   - TES_CHN_LinesAdded, TES_CHN_LinesDeleted (recent test file edits)
   - A value of -1 means "no data" — treat as unknown, not a real metric.

2. **get_all_failure_rates** → per-test overall failure_rate (0.0–1.0) across all builds.

3. **get_test_complexity** → per-test TES_COM_ (31) + TES_PRO_ (6) features:
   - Key: SumCyclomatic, MaxNesting, CountLineCode (complexity)
   - Key: CommitCount, DistinctDevCount, OwnersContribution (ownership)
   - Complex, multi-author tests are more failure-prone.

4. **get_covered_code_risk** → per-test COD_COV_ (81) features:
   - Complexity/churn/process of the production code each test covers.
   - Key: COD_COV_COM_C_SumCyclomatic, COD_COV_CHN_C_LinesAdded
   - Tests covering complex, high-churn production code = higher risk.

## Ranking Rules (strict priority order)

Apply these tiers mechanically. Higher tiers ALWAYS outrank lower tiers.
Within each tier, break ties by: higher failure rate → higher coverage → lower cost.

**T1 — Always-failing:** REC_TotalFailRate ≥ 0.9. Fastest first.
**T2 — Recently failing:** REC_RecentFailRate > 0 AND REC_LastVerdict = 1. Sort by REC_RecentFailRate desc.
**T3 — Fault-signal:** DET_COV_C_Faults > 0 OR DET_COV_IMP_Faults > 0. Prefer those with COV_ChnScoreSum > 0.
**T4 — Historical failures:** failure_rate > 0 but REC_RecentFailRate = 0. Sort by failure_rate desc.
**T5 — High-coverage safety nets:** Never failed, but COV_ChnScoreSum > 0 or high complexity. Sort by coverage desc, cost asc.
**T6 — Remaining:** Everything else. Fastest first.

## Tool Call Strategy

Call get_test_risk_profile and get_all_failure_rates together first (parallel).
Then call get_test_complexity and get_covered_code_risk together (parallel).
That gives you all 150+ features in 2 rounds. Do NOT call tools beyond that
unless you need get_test_history for a specific suspicious test.

## Output (CRITICAL)

Your FINAL message must contain ONLY a JSON array (no markdown, no text before/after the array).
Every test must appear once.

For each object, **reason** must sound like you thinking out loud in first person ("I'm …",
"because …") and must **name at least one real feature** you used (e.g. REC_TotalFailRate,
DET_COV_IMP_Faults, COV_ChnScoreSum, failure_rate, TES_COM_SumCyclomatic, COD_COV_*).
One or two short sentences per test (~40 words max). JSON only outside of string values.

[{"test":"id","priority":1,"confidence":0.9,"reason":"I'm putting this first because REC_TotalFailRate is 1.0 and it fails every build."}]
"""

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def run_agent(dataset_path):
    model = init_chat_model(
        "claude-sonnet-4-6",
        temperature=0,
        max_tokens=16384,
    )

    tools = [get_all_failure_rates, get_test_risk_profile, get_test_complexity, get_covered_code_risk]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    def llm_call(state):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        for attempt in range(5):
            try:
                ai_msg = model_with_tools.invoke(msgs)
                return {"messages": [ai_msg], "llm_calls": state.get("llm_calls", 0) + 1}
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    time.sleep(65)
                else:
                    raise
        raise Exception("Still rate-limited after 5 retries")

    def tool_node(state):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            t = tools_by_name[tool_call["name"]]
            observation = t.invoke(tool_call["args"])
            result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        return {"messages": result}

    def should_continue(state):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        if state.get("llm_calls", 0) > 10:
            return END
        return END

    agent = StateGraph(MessagesState)
    agent.add_node("llm_call", llm_call)
    agent.add_node("tool_node", tool_node)
    agent.add_edge(START, "llm_call")
    agent.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent.add_edge("tool_node", "llm_call")

    compiled = agent.compile()
    result = compiled.invoke({"messages": [HumanMessage(content=f"Prioritize tests for the next build. The dataset is at: {dataset_path}")]})

    content = result["messages"][-1].content
    if not isinstance(content, str):
        raise ValueError(f"Expected string model output, got {type(content).__name__}")
    text = content.strip().replace("```json", "").replace("```", "")
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array in model reply. Start of message:\n{text[:400]!r}")
    return json.loads(text[start : end + 1])
