from __future__ import annotations

"""
TCP Agent dispatcher.

The production path is the multi-agent pipeline:
    Filter Agent (cheap LLM, classifies T1-T6)
    → Ranking Agent (stronger LLM, deeply ranks T1-T5)
    → Validator (deterministic-fallback safety net)

The legacy single-agent path was removed in favour of the multi-agent
pipeline, which is faster, cheaper, and grounded in the Yaraghi 2022
TCP-CI feature taxonomy.
"""

import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL",
    category=Warning,
)

logger = logging.getLogger(__name__)

from tcp_agent.config import AgentMode, set_mode


def run_multi_agent(
    dataset_path,
    mode: AgentMode = AgentMode.PILOT,
    batch_size: int = 200,
    filter_model: str = "gpt-4o-mini",
    ranking_model: str = "gpt-4o",
):
    """Two-agent pipeline: Filter → Ranking → Validation.

    1. Filter Agent classifies tests into T1-T5 (high-risk) vs T6 (low-signal)
    2. Ranking Agent performs deep reasoning on only the high-risk subset
    3. Validator checks output; deterministic fallback on failure
    """
    from tcp_agent.agent.filter_agent import run_filter_agent
    from tcp_agent.agent.ranking_agent import run_ranking_agent
    from tcp_agent.agent.validator import (
        validate_ranking, deterministic_fallback, log_validation_errors,
    )
    from tcp_agent.tools.feature_extractor import extract_all_test_ids

    set_mode(mode, dataset_path=dataset_path)

    logger.info(
        "Starting Filter Agent (batch_size=%d, model=%s)",
        batch_size, filter_model,
    )
    filter_result = run_filter_agent(
        dataset_path,
        batch_size=batch_size,
        filter_model=filter_model,
    )
    logger.info(filter_result.summary())

    logger.info(
        "Starting Ranking Agent (model=%s, high_risk=%d tests)",
        ranking_model, len(filter_result.high_risk_tests),
    )
    ranked = run_ranking_agent(
        filter_result,
        dataset_path,
        ranking_model=ranking_model,
    )

    expected_ids = extract_all_test_ids(dataset_path)
    validation = validate_ranking(ranked, expected_ids, filter_result)
    logger.info(str(validation))

    if validation.is_valid:
        return ranked
    log_validation_errors(validation)
    logger.warning("Falling back to deterministic ranker")
    return deterministic_fallback(dataset_path)


def run_agent(
    dataset_path,
    mode: AgentMode = AgentMode.PILOT,
    batch_size: int = 200,
    filter_model: str = "gpt-4o-mini",
    ranking_model: str = "gpt-4o",
):
    """Run the TCP multi-agent pipeline."""
    return run_multi_agent(
        dataset_path,
        mode=mode,
        batch_size=batch_size,
        filter_model=filter_model,
        ranking_model=ranking_model,
    )
