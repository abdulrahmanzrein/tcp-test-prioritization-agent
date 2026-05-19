"""Rate-limit improvement: ranking parallelism is configurable via parameter
(previously hardcoded at module level as _RANKING_PARALLELISM=4).

Default is 1 (sequential — backward-safe for rate-limited providers).
When set > 1, ranking batches run in a ThreadPoolExecutor with that many workers.
"""
from __future__ import annotations

import concurrent.futures
from unittest.mock import patch

from tcp_agent.agent.filter_agent import FilterResult
from tcp_agent.agent.ranking_agent import (
    PrioritizedTests,
    RankedTest,
    RANKING_SYSTEM_PROMPT,
    _extract_ranked_tests,
    run_ranking_agent,
)


def _make_filter_result(num_high_risk: int) -> FilterResult:
    """Synthesize a FilterResult with N high-risk tests across multiple batches."""
    fr = FilterResult()
    fr.high_risk_tests = [
        {
            "test_id": 1000 + i,
            "tier": 2,
            "key_signals": ["fake"],
            "avg_exec_time": 1.0,
        }
        for i in range(num_high_risk)
    ]
    fr.low_signal_tests = []
    fr.metadata = {
        "total_tests": num_high_risk,
        "low_signal_count": 0,
    }
    return fr


def _fake_rank_batch(batch_tests, batch_ids, **_kwargs):
    """Replace the LLM tool-loop with a deterministic stub."""
    return [
        {"test": str(t["test_id"]), "priority": i + 1, "confidence": 0.5, "reason": "x"}
        for i, t in enumerate(batch_tests)
    ]


def test_ranking_workers_default_is_sequential():
    """Default parallelism=1 means ThreadPoolExecutor is NOT constructed."""
    filter_result = _make_filter_result(num_high_risk=24)  # 3 batches of 8

    with patch(
        "tcp_agent.agent.ranking_agent._rank_batch",
        side_effect=lambda batch_tests, batch_ids, **kw: _fake_rank_batch(batch_tests, batch_ids),
    ), patch(
        "tcp_agent.agent.ranking_agent.ThreadPoolExecutor",
        wraps=concurrent.futures.ThreadPoolExecutor,
    ) as spy_pool:
        result = run_ranking_agent(filter_result, dataset_path="fake.csv", ranking_model="m")

    assert spy_pool.call_count == 0, (
        f"Default (parallelism=1) should not construct ThreadPoolExecutor; "
        f"got {spy_pool.call_count} construction(s)"
    )
    # 24 high-risk tests all ranked
    assert len([r for r in result if int(r["test"]) >= 1000]) == 24


def test_ranking_workers_uses_threadpool_when_gt_one():
    """parallelism > 1 must construct ThreadPoolExecutor with that max_workers."""
    filter_result = _make_filter_result(num_high_risk=24)  # 3 batches

    with patch(
        "tcp_agent.agent.ranking_agent._rank_batch",
        side_effect=lambda batch_tests, batch_ids, **kw: _fake_rank_batch(batch_tests, batch_ids),
    ), patch(
        "tcp_agent.agent.ranking_agent.ThreadPoolExecutor",
        wraps=concurrent.futures.ThreadPoolExecutor,
    ) as spy_pool:
        run_ranking_agent(
            filter_result,
            dataset_path="fake.csv",
            ranking_model="m",
            parallelism=3,
        )

    assert spy_pool.call_count == 1
    kwargs = spy_pool.call_args.kwargs
    args = spy_pool.call_args.args
    max_workers = kwargs.get("max_workers", args[0] if args else None)
    assert max_workers == 3, f"Expected max_workers=3, got {max_workers}"


def test_ranking_keeps_low_signal_tests_out_of_llm_batches_and_appends_tail():
    """T6 was already decided by the Filter LLM, so ranking should save context."""
    filter_result = _make_filter_result(num_high_risk=3)
    filter_result.low_signal_tests = [
        {"test_id": 2000, "tier": 6, "key_signals": ["none"], "avg_exec_time": 1.0},
        {"test_id": 2001, "tier": 6, "key_signals": ["none"], "avg_exec_time": 2.0},
    ]
    filter_result.metadata["total_tests"] = 5
    filter_result.metadata["low_signal_count"] = 2

    def fake_rank_batch(batch_tests, batch_ids, **_kwargs):
        assert batch_ids == [1000, 1001, 1002]
        return [
            {"test": str(tid), "priority": i + 1, "confidence": 0.5, "reason": "llm"}
            for i, tid in enumerate(batch_ids)
        ]

    with patch("tcp_agent.agent.ranking_agent._rank_batch", side_effect=fake_rank_batch):
        result = run_ranking_agent(filter_result, dataset_path="fake.csv", ranking_model="m")

    assert [item["test"] for item in result] == ["1000", "1001", "1002", "2000", "2001"]


def test_ranking_does_not_recover_missing_or_placeholder_ids():
    """Malformed Qwen/Ollama rows should fail validation instead of being repaired."""
    filter_result = _make_filter_result(num_high_risk=3)

    def fake_rank_batch(batch_tests, batch_ids, **_kwargs):
        return [
            {"test": str(batch_ids[0]), "priority": 1, "confidence": 0.8, "reason": "valid"},
            {"test": "{TEST_ID}", "priority": 2, "confidence": 0.5, "reason": "placeholder"},
        ]

    with patch("tcp_agent.agent.ranking_agent._rank_batch", side_effect=fake_rank_batch):
        result = run_ranking_agent(filter_result, dataset_path="fake.csv", ranking_model="m")

    assert [item["test"] for item in result] == ["1000", "{TEST_ID}"]


def test_ranking_prompt_does_not_contain_copyable_placeholder_id():
    assert "{TEST_ID}" not in RANKING_SYSTEM_PROMPT
    assert "{test_id}" not in RANKING_SYSTEM_PROMPT
    assert '"test":"id"' not in RANKING_SYSTEM_PROMPT


def test_extract_ranked_tests_uses_direct_json_without_second_llm():
    class FailingStructuredModel:
        def invoke(self, messages):
            raise AssertionError("structured extraction should not run")

    content = (
        '[{"test":"1000","priority":1,"confidence":0.9,"reason":"Tier 2: recent failure."},'
        '{"test":"1001","priority":2,"confidence":0.8,"reason":"Tier 4: history."}]'
    )

    result = _extract_ranked_tests(
        final_content=content,
        structured_model=FailingStructuredModel(),
        batch_ids=[1000, 1001],
        ranking_model="fake",
    )

    assert [item["test"] for item in result] == ["1000", "1001"]


def test_extract_ranked_tests_repairs_placeholder_ids_with_llm_retry():
    class RepairStructuredModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return PrioritizedTests(
                ranked_tests=[
                    RankedTest(test="1000", priority=1, confidence=0.9, reason="valid"),
                    RankedTest(test="1001", priority=2, confidence=0.8, reason="repaired"),
                ]
            )

    model = RepairStructuredModel()

    result = _extract_ranked_tests(
        final_content='[{"test":"1000"},{"test":"{TEST_ID}"}]',
        structured_model=model,
        batch_ids=[1000, 1001],
        ranking_model="fake",
    )

    assert model.calls == 1
    assert [item["test"] for item in result] == ["1000", "1001"]
