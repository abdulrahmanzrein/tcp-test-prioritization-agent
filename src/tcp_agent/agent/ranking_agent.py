from __future__ import annotations

"""
Ranking Agent — "The Expert"

Takes the filtered high-risk tests (T1–T5) from the Filter Agent and
performs deep, research-grounded reasoning to produce the final ranked
output.  T6 tests were already filtered by the LLM and are appended after
the high-risk list to save ranking context.

Design
------
1. Uses a LangGraph tool-calling loop (same pattern as the original agent)
   but only requests full feature context for the high-risk subset.
2. Produces one concise justification per test.
3. Optionally uses a Merge Agent to globally reorder locally ranked batches.
4. Appends the Filter Agent's T6 tail after the ranked high-risk tests.
"""

import operator
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from tcp_agent.agent.filter_agent import FilterResult
from tcp_agent.tools.full_context_tool import get_full_test_context


from tcp_agent.utils.llm_utils import (
    resolve_provider,
    invoke_with_retry,
    build_init_chat_model_kwargs,
)


# ── Structured output schema ─────────────────────────────────────────

class RankedTest(BaseModel):
    test: str = Field(description="Test ID")
    priority: int = Field(description="1 = run first")
    confidence: float = Field(description="0.0–1.0")
    reason: str = Field(description="One short sentence citing tier and key feature values")


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
5. **Feature set:** Use every provided legal TCP-CI feature; omitted keys are no-data (-1).

## Ranking Rules (strict priority order)

**T1 — Persistent failures:** REC_TotalFailRate ≥ 0.9. Sort by cost ascending.

**T2 — Recent/active failures:** REC_RecentFailRate > 0 OR REC_LastVerdict = 1
OR low REC_LastFailureAge. Sort by REC_RecentFailRate desc, then cost ascending.
Lower REC_LastFailureAge = worse.

**T3 — Risky covered changed/impacted code:** COV_ChnScoreSum > 0 OR COV_ImpScoreSum > 0,
or high COD_COV_* complexity/churn/process values. Prefer changed-code evidence over impacted-code evidence.

**T4 — Historical failures, currently passing:** failure_rate > 0 but
REC_RecentFailRate = 0. Sort by failure_rate desc, factor in TES_PRO_OwnersExperience.

**T5 — High-signal never-failed:** COV_ChnScoreSum > 0, high complexity,
low owner experience, or high covered-code complexity. Sort by coverage desc, cost ascending.

## Tool Call Strategy

You have access to one tool that returns all legal TCP-CI features for ONLY the high-risk tests.
Call get_full_test_context exactly once with the provided test_ids, then rank from that context.

## Output (CRITICAL)

Your FINAL message must contain ONLY a JSON array (no markdown, no text before/after).
Every high-risk test must appear exactly once.

For each **reason**, write EXACTLY ONE short phrase (max ~10 words) that:
1. States tier (T1-T5) and why the test belongs there.
2. Mentions 1-2 key feature values.
3. Mentions tie-break logic only if it changed ordering.

Good example:
"T2 recent failure; REC_RecentFailRate=0.8."

Bad (too vague): "High failure rate, placing first."

Use this shape, but replace test values with the exact IDs provided in the task:
[{\"test\":\"123\",\"priority\":1,\"confidence\":0.9,\"reason\":\"Tier 2: ...\"}]
"""


MERGE_SYSTEM_PROMPT = """\
You are the global Merge Agent for an LLM-based Test Case Prioritization system.

You receive locally ranked high-risk test batches from the Ranking Agent. Your
job is to create ONE global ranking across all high-risk tests, maximizing APFDc.

Do NOT preserve batch order if a later-batch test has stronger failure evidence.
Use the same research-grounded priority: REC/history signals first, then
test-source TES signals, then coverage/COD_COV signals, with execution cost as
the tie-breaker when risk is similar.

Return every allowed test exactly once. Do not invent IDs. Do not include T6
tests. Your final answer must be ONLY a JSON array with:
test, priority, confidence, reason.

Each reason must be one short phrase, max ~10 words.
"""


# ── LLM call with rate-limit retry ───────────────────────────────────

# _resolve_provider is now imported from llm_utils


def _build_models(model_name: str, tools: list):
    """Initialize a chat model and return (tools-bound, structured-output) pair."""
    provider = resolve_provider(model_name)
    # Reasoning-token models (o1, o3, gpt-5*) only accept the default temperature.
    skip_temp = model_name.startswith(("o1", "o3", "gpt-5"))
    kwargs = build_init_chat_model_kwargs(model_name, skip_temperature=skip_temp)
    base = init_chat_model(model_name, model_provider=provider, **kwargs)
    return base.bind_tools(tools), base.with_structured_output(PrioritizedTests)


def _build_structured_model(model_name: str):
    provider = resolve_provider(model_name)
    skip_temp = model_name.startswith(("o1", "o3", "gpt-5"))
    kwargs = build_init_chat_model_kwargs(model_name, skip_temperature=skip_temp)
    base = init_chat_model(model_name, model_provider=provider, **kwargs)
    return base.with_structured_output(PrioritizedTests)


# ── Main ranking function ────────────────────────────────────────────

def _chunk_list(lst: list, size: int) -> list[list]:
    """Split a list into chunks of the given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _coerce_ranked_tests(items: list) -> list[dict]:
    ranked = []
    for item in items:
        if isinstance(item, BaseModel):
            ranked.append(item.model_dump())
        elif isinstance(item, dict):
            ranked.append(dict(item))
    return ranked


def _extract_json_array(text: str) -> list[dict] | None:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return _coerce_ranked_tests(parsed)


def _batch_validation_errors(ranked: list[dict], batch_ids: list[int]) -> list[str]:
    allowed = set(batch_ids)
    ids = []
    errors = []

    for idx, item in enumerate(ranked):
        if not isinstance(item, dict):
            errors.append(f"Item {idx} is not an object")
            continue
        for key in ("test", "priority", "confidence", "reason"):
            if key not in item:
                errors.append(f"Item {idx} missing key: {key}")
        try:
            tid = int(item.get("test"))
        except (TypeError, ValueError):
            errors.append(f"Invalid test ID: {item.get('test')!r}")
            continue
        ids.append(tid)
        if tid not in allowed:
            errors.append(f"Unknown test ID: {tid}")

    seen = set()
    dupes = []
    for tid in ids:
        if tid in seen:
            dupes.append(tid)
        seen.add(tid)
    if dupes:
        errors.append(f"Duplicate test ID(s): {sorted(set(dupes))}")

    missing = allowed - set(ids)
    if missing:
        errors.append(f"Missing test ID(s): {sorted(missing)}")

    return errors


def _structured_extract(structured_model, final_content: str, ranking_model: str):
    parsed = invoke_with_retry(
        structured_model,
        [
            SystemMessage(
                content=(
                    "Extract the test prioritization from this message into the schema. "
                    "Use only concrete numeric test IDs that appear in the message. "
                    "Never use placeholders."
                )
            ),
            HumanMessage(content=final_content),
        ],
        model_name=ranking_model,
    )
    return _coerce_ranked_tests(parsed.ranked_tests)


def _repair_ranked_tests(
    structured_model,
    ranked: list[dict],
    errors: list[str],
    batch_ids: list[int],
    ranking_model: str,
) -> list[dict]:
    parsed = invoke_with_retry(
        structured_model,
        [
            SystemMessage(
                content=(
                    "Repair this ranking output. Return every allowed test ID exactly once. "
                    "Do not invent IDs. Do not use placeholders. Keep the ranking intent "
                    "from the previous output when possible."
                )
            ),
            HumanMessage(
                content=(
                    f"Allowed test IDs: {batch_ids}\n"
                    f"Validation errors: {errors}\n"
                    f"Previous output: {json.dumps(ranked, separators=(',', ':'))}"
                )
            ),
        ],
        model_name=ranking_model,
    )
    return _coerce_ranked_tests(parsed.ranked_tests)


def _extract_ranked_tests(
    final_content: str,
    structured_model,
    batch_ids: list[int],
    ranking_model: str,
) -> list[dict]:
    """Extract and LLM-repair a batch ranking without deterministic re-ranking."""
    ranked = _extract_json_array(final_content)
    if ranked is None:
        ranked = _structured_extract(structured_model, final_content, ranking_model)

    errors = _batch_validation_errors(ranked, batch_ids)
    if not errors:
        return ranked

    repaired = _repair_ranked_tests(
        structured_model=structured_model,
        ranked=ranked,
        errors=errors,
        batch_ids=batch_ids,
        ranking_model=ranking_model,
    )
    repaired_errors = _batch_validation_errors(repaired, batch_ids)
    if not repaired_errors:
        return repaired

    return ranked


def _merge_validation_errors(ranked: list[dict], allowed_ids: list[int]) -> list[str]:
    return _batch_validation_errors(ranked, allowed_ids)


def _repair_merged_ranking(
    merged: list[dict],
    original_ranked: list[dict],
    allowed_ids: list[int],
) -> list[dict]:
    """Keep valid merge choices, then restore missing IDs from local ranking order."""
    allowed_set = set(allowed_ids)
    by_original_id = {
        int(item["test"]): item
        for item in original_ranked
        if str(item.get("test", "")).isdigit()
    }

    repaired: list[dict] = []
    seen: set[int] = set()
    for item in sorted(merged, key=lambda x: int(x.get("priority", 10**6))):
        try:
            tid = int(item["test"])
        except (KeyError, TypeError, ValueError):
            continue
        if tid not in allowed_set or tid in seen:
            continue
        item["reason"] = item.get("reason") or "Merge order."
        item["confidence"] = float(item.get("confidence", 0.5))
        repaired.append(item)
        seen.add(tid)

    for tid in allowed_ids:
        if tid in seen:
            continue
        fallback = dict(by_original_id.get(tid, {"test": str(tid)}))
        fallback["test"] = str(tid)
        fallback["reason"] = fallback.get("reason") or "Merge repair; local order."
        fallback["confidence"] = min(float(fallback.get("confidence", 0.5)), 0.6)
        repaired.append(fallback)
        seen.add(tid)

    for i, item in enumerate(repaired):
        item["priority"] = i + 1
    return repaired


def _merge_ranked_batches(
    ranked: list[dict],
    high_risk_tests: list[dict],
    ranking_model: str,
) -> tuple[list[dict], str, int]:
    """Use a third LLM agent to globally reorder locally ranked batches."""
    allowed_ids = [int(t["test_id"]) for t in high_risk_tests]
    if len(ranked) <= 1:
        return ranked, "not_needed", 0

    by_id = {int(t["test_id"]): t for t in high_risk_tests}
    merge_items = []
    for item in ranked:
        try:
            tid = int(item["test"])
        except (KeyError, TypeError, ValueError):
            continue
        source = by_id.get(tid, {})
        merge_items.append({
            "test": str(tid),
            "batch": item.get("_batch_idx", 0),
            "local_priority": item.get("priority"),
            "tier": source.get("tier"),
            "key_signals": source.get("key_signals", []),
            "avg_exec_time": source.get("avg_exec_time", 0.0),
            "local_reason": item.get("reason", ""),
            "local_confidence": item.get("confidence", 0.5),
        })

    structured_model = _build_structured_model(ranking_model)
    parsed = invoke_with_retry(
        structured_model,
        [
            SystemMessage(content=MERGE_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Allowed high-risk test IDs: {allowed_ids}\n"
                "Locally ranked batch outputs:\n"
                f"{json.dumps(merge_items, separators=(',', ':'))}"
            )),
        ],
        model_name=ranking_model,
    )
    merged = _coerce_ranked_tests(parsed.ranked_tests)
    errors = _merge_validation_errors(merged, allowed_ids)
    if errors:
        merged_ids = {
            int(item["test"])
            for item in merged
            if str(item.get("test", "")).isdigit()
        }
        missing_count = len(set(allowed_ids) - merged_ids)
        repaired = _repair_merged_ranking(merged, ranked, allowed_ids)
        repaired_errors = _merge_validation_errors(repaired, allowed_ids)
        if repaired_errors:
            raise ValueError(
                "Merge Agent output invalid and repair failed: "
                f"{errors[:2]} -> {repaired_errors[:2]}"
            )
        print(f"  [MERGE] Merge Agent output repaired: {errors[:2]}")
        return repaired, "repaired", missing_count

    merged.sort(key=lambda x: int(x.get("priority", 10**6)))
    for i, item in enumerate(merged):
        item["priority"] = i + 1
    return merged, "clean", 0


# Max tests per ranking batch.
# Keep this conservative because long per-test reasons can hit structured-output
# max_tokens limits (especially on Claude), causing truncated outputs and parse
# failures.
_RANKING_BATCH_SIZE = 8

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

    tools = [get_full_test_context]
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
        response = invoke_with_retry(model_with_tools, msgs, model_name=ranking_model)
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

    return _extract_ranked_tests(
        final_content=result["messages"][-1].content,
        structured_model=structured_model,
        batch_ids=batch_ids,
        ranking_model=ranking_model,
    )


def run_ranking_agent(
    filter_result: FilterResult,
    dataset_path: str,
    ranking_model: str = "gemini-3-flash-preview",
    parallelism: int = 1,
    batch_size: int | None = None,
    merge_agent: bool = False,
) -> list[dict]:
    """Run the Ranking Agent on the filtered high-risk tests.

    Splits high-risk tests into batches to stay within model context limits.
    Each batch is ranked independently, then results are merged by batch
    order to produce the final ordering. Returns the full ranked list:
    high-risk tests with reasoning + T6 tail.

    parallelism: number of concurrent ranking-batch LLM calls. Default 1
    (sequential — safe for rate-limited providers). Raise to 4+ on
    higher-tier API accounts that can handle parallel TPM/RPM.
    batch_size: max high-risk tests per ranking batch. Defaults to
    _RANKING_BATCH_SIZE.
    merge_agent: if True, run a third LLM agent to globally reorder locally
    ranked batches before appending the T6 tail.
    """
    high_risk_tests = filter_result.high_risk_tests
    high_risk_ids = [t["test_id"] for t in high_risk_tests]

    # ── Short-circuit: if no high-risk tests, return T6 tail only ────
    if not high_risk_ids:
        tail = _build_t6_tail(filter_result.low_signal_tests, start_priority=1)
        for item in tail:
            item["_merge_status"] = "not_needed"
            item["_merge_missing_count"] = 0
        return tail

    total_tests = filter_result.metadata.get("total_tests", len(high_risk_ids))
    low_signal_count = filter_result.metadata.get("low_signal_count", 0)
    effective_batch_size = batch_size or _RANKING_BATCH_SIZE

    # ── Batch the high-risk tests ────────────────────────────────────
    test_batches = _chunk_list(high_risk_tests, effective_batch_size)
    id_batches = _chunk_list(high_risk_ids, effective_batch_size)

    print(
        f"  [RANKING] {len(high_risk_ids)} high-risk tests → {len(test_batches)} "
        f"batches of ≤{effective_batch_size} (parallelism={parallelism})"
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
    merge_status = "disabled" if not merge_agent else "not_needed"
    merge_missing_count = 0
    if len(test_batches) == 1 or parallelism <= 1:
        # Avoid thread overhead for the trivial case
        for batch_idx, (batch_tests, batch_ids) in enumerate(zip(test_batches, id_batches)):
            all_ranked.extend(_run_one_batch(batch_idx, batch_tests, batch_ids))
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
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

    # ── Merge local batch rankings. If enabled, a third LLM agent resolves the
    #    global ordering across batches; otherwise preserve batch order.
    def _sort_key(item):
        return (item["_batch_idx"], item.get("priority", 10**6))

    all_ranked.sort(key=_sort_key)

    if merge_agent and len(test_batches) > 1 and all_ranked:
        print(f"  [MERGE] Merging {len(all_ranked)} high-risk tests across {len(test_batches)} batches...")
        all_ranked, merge_status, merge_missing_count = _merge_ranked_batches(
            ranked=all_ranked,
            high_risk_tests=high_risk_tests,
            ranking_model=ranking_model,
        )

    for item in all_ranked:
        item.pop("_batch_idx", None)

    # ── Re-number priorities sequentially ────────────────────────────
    for i, item in enumerate(all_ranked):
        item["priority"] = i + 1

    # ── Append T6 tail ───────────────────────────────────────────────
    next_priority = len(all_ranked) + 1
    t6_tail = _build_t6_tail(filter_result.low_signal_tests, start_priority=next_priority)
    all_ranked.extend(t6_tail)

    for item in all_ranked:
        item["_merge_status"] = merge_status
        item["_merge_missing_count"] = merge_missing_count

    return all_ranked


def _build_t6_tail(low_signal_tests: list[dict], start_priority: int) -> list[dict]:
    """Build the T6 tail from tests the Filter Agent already marked low-signal."""
    sorted_t6 = sorted(low_signal_tests, key=lambda t: t.get("avg_exec_time", 0.0))

    tail = []
    for i, t in enumerate(sorted_t6):
        avg_time = t.get("avg_exec_time", 0.0)
        tail.append({
            "test": str(t["test_id"]),
            "priority": start_priority + i,
            "confidence": 0.1,
            "reason": f"T6 low-signal; cost={avg_time:.1f}ms.",
        })
    return tail
