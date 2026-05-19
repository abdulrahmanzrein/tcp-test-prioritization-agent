from __future__ import annotations

from unittest.mock import patch

import pytest

from tcp_agent.agent.filter_agent import FilterResult
from tcp_agent.agent.tcp_agent import run_multi_agent


def _filter_result() -> FilterResult:
    result = FilterResult()
    result.high_risk_tests = [
        {"test_id": 1, "tier": 2, "key_signals": ["llm"], "avg_exec_time": 1.0}
    ]
    result.low_signal_tests = []
    result.metadata = {"total_tests": 1, "low_signal_count": 0}
    return result


def test_invalid_llm_ranking_raises_instead_of_falling_back():
    invalid_ranked = [
        {"test": "{TEST_ID}", "priority": 1, "confidence": 0.5, "reason": "bad"}
    ]

    with patch(
        "tcp_agent.agent.filter_agent.run_filter_agent",
        return_value=_filter_result(),
    ), patch(
        "tcp_agent.agent.ranking_agent.run_ranking_agent",
        return_value=invalid_ranked,
    ), patch(
        "tcp_agent.tools.feature_extractor.extract_all_test_ids",
        return_value={1},
    ):
        with pytest.raises(ValueError, match="LLM ranking failed validation"):
            run_multi_agent("fake.csv", filter_model="fake", ranking_model="fake")
