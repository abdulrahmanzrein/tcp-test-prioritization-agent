"""Filter output preserves the LLM's tier decisions and ordering."""
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


def test_filter_preserves_llm_order_without_tier_sorting(tmp_path: Path):
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)

    fake_result = BatchClassificationResult(
        classifications=[
            TestClassification(test_id=200, tier=2, key_signals=["REC_RecentFailRate=0.9"]),
            TestClassification(test_id=100, tier=1, key_signals=["REC_TotalFailRate=0.8"]),
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

    assert ids_in_order == [200, 100]
    assert tiers_in_order == [2, 1]
