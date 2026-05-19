"""Rate-limit improvement: ranking parallelism is configurable via parameter
(previously hardcoded at module level as _RANKING_PARALLELISM=4).

Default is 1 (sequential — backward-safe for rate-limited providers).
When set > 1, ranking batches run in a ThreadPoolExecutor with that many workers.
"""
from __future__ import annotations

import concurrent.futures
from unittest.mock import patch

from tcp_agent.agent.filter_agent import FilterResult
from tcp_agent.agent.ranking_agent import run_ranking_agent


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
