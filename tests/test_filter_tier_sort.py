"""Bug 1 regression test: high_risk_tests must be ordered primarily by tier,
not by REC_RecentFailRate.

Before the fix, the merge sorted by (test_order_position, tier) where
test_order_position came from extract_risk_profiles which sorts by
REC_RecentFailRate descending. This caused T1 tests with low recent failure
rates to land in late ranking batches, ranking them below T2 tests.

After the fix, T1 tests must always appear before T2 tests in high_risk_tests,
regardless of REC_RecentFailRate.
"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pandas as pd
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


def _write_csv(path: Path) -> None:
    """Two builds, two tests:
      Test 100 → T1-eligible: REC_TotalFailRate=0.8 but REC_RecentFailRate=0.0
      Test 200 → T2-eligible: REC_RecentFailRate=0.9, REC_TotalFailRate=0.2
    """
    rows = [
        {
            "Build": 1, "Test": 100, "Verdict": 1, "Duration": 10.0,
            "REC_TotalFailRate": 0.8, "REC_RecentFailRate": 0.0,
            "REC_LastVerdict": 0,
        },
        {
            "Build": 2, "Test": 100, "Verdict": 0, "Duration": 10.0,
            "REC_TotalFailRate": 0.8, "REC_RecentFailRate": 0.0,
            "REC_LastVerdict": 0,
        },
        {
            "Build": 1, "Test": 200, "Verdict": 0, "Duration": 5.0,
            "REC_TotalFailRate": 0.2, "REC_RecentFailRate": 0.9,
            "REC_LastVerdict": 0,
        },
        {
            "Build": 2, "Test": 200, "Verdict": 1, "Duration": 5.0,
            "REC_TotalFailRate": 0.2, "REC_RecentFailRate": 0.9,
            "REC_LastVerdict": 0,
        },
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_t1_outranks_t2_regardless_of_recent_fail_rate(tmp_path: Path):
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)

    # Mock the LLM: classify test 100 as T1 and test 200 as T2.
    # The guardrail will keep test 100 at T1 (no recent failure evidence) and
    # cap test 200 at T2 (rec_recent=0.9 > 0).
    fake_result = BatchClassificationResult(
        classifications=[
            TestClassification(test_id=100, tier=1, key_signals=["REC_TotalFailRate=0.8"]),
            TestClassification(test_id=200, tier=2, key_signals=["REC_RecentFailRate=0.9"]),
        ]
    )

    with patch(
        "tcp_agent.agent.filter_agent._classify_batch",
        return_value=fake_result,
    ), patch(
        "tcp_agent.agent.filter_agent._build_structured_model",
        return_value=object(),  # not used because _classify_batch is mocked
    ):
        result = run_filter_agent(str(csv_path), batch_size=10, filter_model="fake")

    ids_in_order = [e["test_id"] for e in result.high_risk_tests]
    tiers_in_order = [e["tier"] for e in result.high_risk_tests]

    assert ids_in_order == [100, 200], (
        f"T1 test should appear before T2 test regardless of REC_RecentFailRate; "
        f"got order={ids_in_order} tiers={tiers_in_order}"
    )
    assert tiers_in_order == [1, 2]
