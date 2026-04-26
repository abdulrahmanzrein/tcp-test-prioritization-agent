from __future__ import annotations

"""
Filter Agent — "The Scanner"

Processes tests in batches to classify them as high-risk (T1–T5) or
low-signal (T6).  Uses pre-extracted features (no tool-calling loop)
and a single LLM call per batch to keep costs low.

Design
------
1. All feature data is extracted once via pure Python (feature_extractor).
2. Tests are split into configurable batches (default 200).
3. Each batch gets one LLM call with a concise classification prompt.
4. Results are merged into a FilterResult for the Ranking Agent.
"""

import json
import math
import time
import warnings

warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings")

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from tcp_agent.tools.feature_extractor import (
    extract_risk_profiles,
    extract_failure_rates,
    extract_exec_times,
)


# ── Structured output schemas ────────────────────────────────────────

class TestClassification(BaseModel):
    test_id: int = Field(description="Test ID")
    tier: int = Field(description="Estimated tier 1-6")
    key_signals: list[str] = Field(
        description="1-2 key feature signals, e.g. 'REC_TotalFailRate=0.92'"
    )


class BatchClassificationResult(BaseModel):
    classifications: list[TestClassification] = Field(
        description="Classification for every test in the batch"
    )


# ── Filter result data class ─────────────────────────────────────────

class FilterResult:
    """Container for filter agent output, consumed by the ranking agent."""

    def __init__(self):
        self.high_risk_tests: list[dict] = []   # T1-T5 with key_signals
        self.low_signal_tests: list[dict] = []  # T6 with avg_exec_time
        self.metadata: dict = {}

    def summary(self) -> str:
        return (
            f"Filter Result: {len(self.high_risk_tests)} high-risk (T1-T5), "
            f"{len(self.low_signal_tests)} low-signal (T6)"
        )


# ── System prompt ────────────────────────────────────────────────────

FILTER_SYSTEM_PROMPT = """\
You are a test-case triage agent.  Your ONLY job is to classify each test as
"high-risk" (Tier 1-5) or "low-signal" (Tier 6).

## Tier Criteria  (from TCP-CI research, Yaraghi et al. 2022)

T1 — Persistent failure:   REC_TotalFailRate >= 0.9
T2 — Recent/active failure: REC_RecentFailRate > 0  AND  REC_LastVerdict = 1
T3 — Fault-adjacent:        DET_COV_C_Faults > 0  OR  DET_COV_IMP_Faults > 0
T4 — Historical failure:    failure_rate > 0  but  REC_RecentFailRate = 0
T5 — High-signal, never failed:
     COV_ChnScoreSum > 0  OR  high TES_COM_SumCyclomatic  OR
     low TES_PRO_OwnersExperience  OR  high COD_COV_COM_C_SumCyclomatic
T6 — Low-signal remainder:  none of the above

## Rules
- If a test meets ANY of T1-T5, classify it at its HIGHEST tier (T1 > T2 > …).
- A missing feature (absent key) means "no data" — do NOT treat it as zero.
- For each test output: test_id, tier (int 1-6), and 1-2 key_signals
  (feature=value strings that justify the tier).
- You MUST classify EVERY test in the batch — do not skip any.
"""


# ── Helper: LLM call with retry ──────────────────────────────────────

def _invoke_with_retry(model, messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            if "429" in str(e) or "rate_limit" in err:
                time.sleep(65)
            elif "connection" in err or "ssl" in err or "read" in err or "timeout" in err:
                time.sleep(10)
            else:
                raise
    raise Exception(f"Still rate-limited after {max_retries} retries")


# ── Core filter logic ────────────────────────────────────────────────

def _build_batch_prompt(
    batch_profiles: list[dict],
    failure_rates: dict,
    exec_times: dict,
) -> str:
    """Build the human message for one batch of tests."""
    lines = [f"Classify these {len(batch_profiles)} tests:\n"]

    for profile in batch_profiles:
        tid = profile["test"]
        fr = failure_rates.get(tid, 0.0)
        et = exec_times.get(tid, 0.0)

        # compact representation: only non-default features
        feats = {k: v for k, v in profile.items() if k != "test"}
        feats["failure_rate"] = round(fr, 4)
        feats["avg_exec_time"] = round(et, 2)

        lines.append(f"Test {tid}: {json.dumps(feats, separators=(',', ':'))}")

    return "\n".join(lines)


def _chunk(lst: list, size: int) -> list[list]:
    """Split a list into chunks of the given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def run_filter_agent(
    dataset_path: str,
    batch_size: int = 200,
    filter_model: str = "gpt-4o-mini",
) -> FilterResult:
    """Run the Filter Agent over the dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the CSV dataset.
    batch_size : int
        Number of tests per LLM batch call (default 200).
    filter_model : str
        LLM model name for the filter agent (default gpt-4o-mini).

    Returns
    -------
    FilterResult
        Container with high_risk_tests, low_signal_tests, and metadata.
    """
    # ── 1. Pre-extract all features (pure Python, no LLM) ────────────
    risk_profiles = extract_risk_profiles(dataset_path, sparse=True)
    failure_rates = extract_failure_rates(dataset_path)
    exec_times = extract_exec_times(dataset_path)

    total_tests = len(risk_profiles)
    batches = _chunk(risk_profiles, batch_size)

    # ── 2. Init LLM with structured output ───────────────────────────
    provider = "google_genai" if "gemini" in filter_model else None
    model = init_chat_model(filter_model, model_provider=provider, temperature=0)
    structured_model = model.with_structured_output(BatchClassificationResult)

    # ── 3. Process each batch ────────────────────────────────────────
    result = FilterResult()

    for batch_idx, batch in enumerate(batches):
        prompt = _build_batch_prompt(batch, failure_rates, exec_times)

        parsed = _invoke_with_retry(structured_model, [
            SystemMessage(content=FILTER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        # merge classifications into result
        batch_test_ids = {p["test"] for p in batch}
        classified_ids = set()

        for cls in parsed.classifications:
            classified_ids.add(cls.test_id)
            entry = {
                "test_id": cls.test_id,
                "tier": cls.tier,
                "key_signals": cls.key_signals,
                "avg_exec_time": exec_times.get(cls.test_id, 0.0),
            }
            if cls.tier <= 5:
                result.high_risk_tests.append(entry)
            else:
                result.low_signal_tests.append(entry)

        # safety: any test the LLM missed gets classified as T6
        missed = batch_test_ids - classified_ids
        for tid in missed:
            result.low_signal_tests.append({
                "test_id": tid,
                "tier": 6,
                "key_signals": ["missed_by_filter"],
                "avg_exec_time": exec_times.get(tid, 0.0),
            })

    # ── 4. Metadata ──────────────────────────────────────────────────
    result.metadata = {
        "total_tests": total_tests,
        "num_batches": len(batches),
        "batch_size": batch_size,
        "high_risk_count": len(result.high_risk_tests),
        "low_signal_count": len(result.low_signal_tests),
        "filter_model": filter_model,
    }

    return result
