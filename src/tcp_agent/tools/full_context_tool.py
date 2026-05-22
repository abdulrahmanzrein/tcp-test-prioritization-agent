from __future__ import annotations

"""
Combined ranking-context tool.

The Ranking Agent is still responsible for deciding the final order, but this
tool avoids a multi-turn "ask three tools" loop by returning the full legal
TCP-CI feature vector for the requested high-risk tests in one call.
"""

from langchain_core.tools import tool

from tcp_agent.tools.feature_extractor import extract_risk_profiles


@tool
def get_full_test_context(dataset_path: str, test_ids=None) -> list[dict]:
    """Return the full legal TCP-CI feature context for selected tests.

    Includes all CSV feature columns except Build, Test, Verdict, and Duration.
    DET_COV_*_Faults are retained as historical previously-detected-fault
    features from the TCP-CI feature model. Values of -1 are omitted because
    TCP-CI uses -1 as a no-data sentinel; real zero values are kept.
    Optional: pass test_ids (list of ints) to restrict context to a ranking batch.
    """
    return extract_risk_profiles(dataset_path, sparse=True, test_ids=test_ids)
