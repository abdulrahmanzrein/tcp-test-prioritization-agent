"""Rate-limit improvement: --filter-gap sleeps N seconds between filter batches.

Default is 0.0 (no sleep, backward-compatible). When set > 0, run_filter_agent
must call time.sleep between batches (but not after the final batch).
"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from tcp_agent.agent.filter_agent import (
    BatchClassificationResult,
    TestClassification,
    run_filter_agent,
)
from tcp_agent.data_cache import clear_cache


@pytest.fixture(autouse=True)
def _isolate_data_cache():
    clear_cache()
    yield
    clear_cache()


def _write_three_test_csv(path: Path) -> None:
    rows = []
    for tid in (10, 20, 30):
        rows.append({
            "Build": 1, "Test": tid, "Verdict": 0, "Duration": 1.0,
            "REC_TotalFailRate": 0.0, "REC_RecentFailRate": 0.0,
            "REC_LastVerdict": 0,
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _make_t6_for(ids):
    return BatchClassificationResult(
        classifications=[
            TestClassification(test_id=i, tier=6, key_signals=["none"]) for i in ids
        ]
    )


def test_filter_gap_sleeps_between_batches_only(tmp_path: Path):
    """With batch_size=1 and 3 tests → 3 batches → 2 inter-batch sleeps."""
    csv_path = tmp_path / "tiny.csv"
    _write_three_test_csv(csv_path)

    calls_seen = []

    def fake_classify(structured_model, model_name, batch, fr, et, min_chunk=8):
        ids = [int(p["test"]) for p in batch]
        calls_seen.append(ids)
        return _make_t6_for(ids)

    with patch(
        "tcp_agent.agent.filter_agent._classify_batch",
        side_effect=fake_classify,
    ), patch(
        "tcp_agent.agent.filter_agent._build_structured_model",
        return_value=object(),
    ), patch(
        "tcp_agent.agent.filter_agent.time.sleep"
    ) as mock_sleep:
        run_filter_agent(
            str(csv_path),
            batch_size=1,
            filter_model="fake",
            inter_batch_sleep=1.5,
        )

    # Three batches were dispatched
    assert len(calls_seen) == 3
    # Sleep called only between batches (after batch 1 and after batch 2),
    # not after the final batch.
    assert mock_sleep.call_count == 2
    for c in mock_sleep.call_args_list:
        assert c.args[0] == 1.5


def test_filter_gap_default_zero_no_sleep(tmp_path: Path):
    csv_path = tmp_path / "tiny.csv"
    _write_three_test_csv(csv_path)

    def fake_classify(structured_model, model_name, batch, fr, et, min_chunk=8):
        ids = [int(p["test"]) for p in batch]
        return _make_t6_for(ids)

    with patch(
        "tcp_agent.agent.filter_agent._classify_batch",
        side_effect=fake_classify,
    ), patch(
        "tcp_agent.agent.filter_agent._build_structured_model",
        return_value=object(),
    ), patch(
        "tcp_agent.agent.filter_agent.time.sleep"
    ) as mock_sleep:
        # Default — no parameter passed
        run_filter_agent(str(csv_path), batch_size=1, filter_model="fake")

    assert mock_sleep.call_count == 0
