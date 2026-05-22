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
import logging
import time
import warnings

warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings")

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from tcp_agent.tools.feature_extractor import (
    extract_risk_profiles,
    extract_exec_times,
)


from tcp_agent.utils.llm_utils import (
    resolve_provider,
    invoke_with_retry,
    build_init_chat_model_kwargs,
)

logger = logging.getLogger(__name__)


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
You are a test-case triage agent for a Continuous Integration system.
Your ONLY job is to classify each test as "high-risk" (Tier 1-5) or
"low-signal" (Tier 6). The Ranking Agent will deeply rank T1-T5; T6 is
sorted by execution time and appended last. Misclassifying a real failure
as T6 is the costliest mistake you can make.

## Class imbalance — be aggressive

Per Mendoza et al. 2022, only ~3% of regression test executions fail.
The training distribution is heavily biased toward passes. To compensate,
err on the side of classifying borderline tests as T1-T5 rather than T6.
False positives are cheap (the Ranking Agent re-sorts them); false
negatives bury real failures and tank APFDc.

## You receive the FULL Yaraghi 2022 feature set per test

Each test has up to 150 features across nine groups. Use them all when
they're present, but weight them by their empirical predictive power.

### Feature importance hierarchy (Yaraghi 2022 Table 12, RQ2.4)

The published top-15 features by usage frequency in the trained Random
Forest TCP model are dominated by three groups:

  Tier-A (most predictive — these dominate the RF model's splits):
    • REC_Age                       (#1, freq 17034 — sharp failure drop
                                     after the first 10-20% of a test's
                                     lifetime; low REC_Age = real risk)
    • TES_PRO_OwnersExperience      (#2, freq 10618)
    • TES_PRO_AllCommitersExperience (#3, freq 7643)
    • REC_TotalMaxExeTime, REC_LastExeTime, REC_RecentMaxExeTime
    • TES_PRO_OwnersContribution, TES_PRO_CommitCount
    • REC_TotalAvgExeTime, REC_RecentAvgExeTime
    • TES_COM_RatioCommentToCode, TES_COM_CountStmtDecl,
      TES_COM_CountLineCodeDecl, TES_COM_CountLineBlank,
      TES_COM_CountStmtExe

  Tier-B (still informative but lower weight):
    • Other REC_* (fail/transition rates, assert/exc rates, last verdict)
    • Other TES_COM_* (cyclomatic, nesting, class/method counts)
    • TES_CHN_* (test source churn metrics for THIS build)
    • Other TES_PRO_* (DistinctDevCount, MinorContributorCount)

  Tier-C (weakest group — coverage; CL=0.79 vs Full_M, Yaraghi RQ2.3):
    • COV_*           (file coverage of changed/impacted code)
    • COD_COV_*       (complexity/process/churn of covered prod code,
                       split into _C_ for changed and _IMP_ for impacted)

When two tests are otherwise tied, prefer evidence from Tier-A over
Tier-B over Tier-C. Do not let the volume of COD_COV_* features (81 of
them) drown out a single strong REC_ or TES_PRO signal.

The strongest single-feature heuristic is REC_TotalFailRate descending.
Tests that failed in the past tend to fail again.

## Tier criteria — ordered, take the HIGHEST that applies

T1 — Persistent failure
       REC_TotalFailRate >= 0.5
       (test has been broken in a large fraction of its history)

T2 — Recent / active failure
       REC_RecentFailRate > 0
       OR REC_LastVerdict != 0
       OR REC_LastFailureAge in {0, 1, 2}
       (don't require both — either is sufficient. A test that flipped
       pass→fail→pass in recent builds still belongs here.)

T3 — Risky covered code (coverage signals — weakest group, but still
     a real signal when other evidence is absent)
       COV_ChnScoreSum > 0  OR  COV_ImpScoreSum > 0
       OR high COD_COV_COM_C_SumCyclomatic / COD_COV_CHN_C_LinesAdded
       OR high COD_COV_PRO_C_DistinctDevCount
       (test covers production code that is changed, complex, churning,
       or owned by many developers)

T4 — Historical failure or instability
       REC_TotalFailRate > 0
       OR REC_TotalAssertRate > 0
       OR REC_TotalExcRate > 0
       OR REC_TotalTransitionRate > 0
       OR REC_RecentTransitionRate > 0
       (any past failure or verdict transition — even one — is a signal)

T5 — Predictive non-history signals (no past failure, but the model
     would still flag it). Promote to T5 if ANY of these hold:
       • TES_CHN_LinesAdded > 0  OR  TES_CHN_LinesDeleted > 0
         (the test source itself was edited recently)
       • REC_Age <= 5
         (newly created test; failure prob. peaks in first 10-20% of life)
       • TES_PRO_OwnersExperience low (<= 0.5)
         (test owned by a relatively new contributor)
       • TES_PRO_CommitCount >= 3
         (test under active development — many recent edits)
       • TES_PRO_MinorContributorCount >= 2
         (multiple unfamiliar contributors → higher fault risk)
       • TES_COM_SumCyclomatic >= 20  OR  TES_COM_CountStmtExe >= 50
         OR TES_COM_MaxNesting >= 4  OR  TES_COM_CountLineCode >= 200
         (large / complex test → exercises more code, more fault paths)
       • REC_TotalMaxExeTime > 60s  OR  REC_LastExeTime > 60s
         (long-running tests cover more code; treat as high-signal)

T6 — Low-signal remainder
     ONLY if every check above is false. To land in T6 a test should
     look like: stable history, no recent edits, no coverage of changed
     code, mature age, experienced owner, low complexity, short execution.
     When unsure between T5 and T6, choose T5 (see "class imbalance"
     above — false negatives are costly).

## Rules

- Pick the SINGLE highest tier that applies (T1 > T2 > T3 > T4 > T5 > T6).
- A missing feature key = "no data" — do NOT treat it as zero.
  Real zeros (e.g. REC_LastVerdict=0) ARE present and ARE meaningful.
- A value of -1 means "no data" / "never observed" — IGNORE it; do NOT
  use it as evidence in either direction.
- Numeric thresholds are guidelines, not hard cutoffs. If a test is
  borderline-but-trending-risky, choose the higher tier.
- DET_COV_C_Faults and DET_COV_IMP_Faults are in your feature set as
  historical previously-detected-fault features for changed/impacted covered
  source files. Treat them as coverage/fault-history signals, not as target
  build Verdict labels.
- Output: test_id (int), tier (int 1-6), and 1 (max 2) key_signals.
- key_signals MUST be short "feature=value" strings, ≤30 chars each.
  Example: ["REC_TotalFailRate=0.92"], ["REC_Age=2"], ["TES_CHN_LinesAdded=14"].
  No prose, no explanations.
- You MUST classify EVERY test in the batch — do not skip any.
"""


# ── Helper: LLM call with retry ──────────────────────────────────────

def _build_structured_model(model_name: str):
    """Initialize a chat model and bind the BatchClassificationResult schema."""
    provider = resolve_provider(model_name)
    # Reasoning-token models (o1, o3, gpt-5*) only accept the default temperature.
    skip_temp = model_name.startswith(("o1", "o3", "gpt-5"))
    kwargs = build_init_chat_model_kwargs(model_name, skip_temperature=skip_temp)
    base = init_chat_model(model_name, model_provider=provider, **kwargs)
    return base.with_structured_output(BatchClassificationResult)


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

        # compact representation: only TCP-CI feature columns
        feats = {k: v for k, v in profile.items() if k != "test"}

        lines.append(f"Test {tid}: {json.dumps(feats, separators=(',', ':'))}")

    return "\n".join(lines)


def _chunk(lst: list, size: int) -> list[list]:
    """Split a list into chunks of the given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _is_length_error(e: Exception) -> bool:
    """Detect OpenAI's LengthFinishReasonError without importing the class
    (it's only raised through structured-output calls). Fallback to string match."""
    name = type(e).__name__
    if name == "LengthFinishReasonError":
        return True
    msg = str(e).lower()
    return "length limit was reached" in msg or "length_finish_reason" in msg


def _classify_batch(
    structured_model,
    model_name: str,
    batch: list[dict],
    failure_rates: dict,
    exec_times: dict,
    min_chunk: int = 8,
):
    """Classify a single batch. Auto-splits on output-length errors."""
    prompt = _build_batch_prompt(batch, failure_rates, exec_times)
    try:
        return invoke_with_retry(
            structured_model,
            [
                SystemMessage(content=FILTER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            model_name=model_name
        )
    except Exception as e:
        if not _is_length_error(e) or len(batch) <= min_chunk:
            raise
        mid = len(batch) // 2
        print(
            f"   ⚠️  Filter batch hit output-length limit "
            f"({len(batch)} tests) — splitting into {mid} + {len(batch) - mid}"
        )
        left = _classify_batch(structured_model, model_name, batch[:mid], failure_rates, exec_times, min_chunk)
        right = _classify_batch(structured_model, model_name, batch[mid:], failure_rates, exec_times, min_chunk)
        return BatchClassificationResult(
            classifications=list(left.classifications) + list(right.classifications)
        )


def _classify_batch_complete(
    structured_model,
    model_name: str,
    batch: list[dict],
    failure_rates: dict,
    exec_times: dict,
    min_chunk: int = 8,
):
    """Classify a batch and retry omitted test IDs in smaller chunks.

    Some local OpenAI-compatible models return valid structured output but skip
    rows when a batch is too large. Treat that like a soft length failure:
    keep the valid rows, then recursively classify only the omitted profiles.
    """
    parsed = _classify_batch(
        structured_model,
        model_name,
        batch,
        failure_rates,
        exec_times,
        min_chunk=min_chunk,
    )

    expected_ids = {p["test"] for p in batch}
    seen_ids = {cls.test_id for cls in parsed.classifications}
    missed = expected_ids - seen_ids
    if not missed:
        return parsed

    if len(batch) <= 1:
        raise ValueError(
            "Filter LLM omitted test ID after singleton retry: "
            f"{sorted(int(tid) for tid in missed)}"
        )

    missed_profiles = [p for p in batch if p["test"] in missed]
    logger.warning(
        "Filter LLM omitted %d/%d test(s); retrying omitted IDs in smaller batch(es): %s",
        len(missed),
        len(batch),
        sorted(int(tid) for tid in missed)[:10],
    )

    if len(missed_profiles) <= min_chunk:
        retry_chunks = [[p] for p in missed_profiles]
    else:
        mid = max(1, len(missed_profiles) // 2)
        retry_chunks = [missed_profiles[:mid], missed_profiles[mid:]]

    repaired = list(parsed.classifications)
    for retry_batch in retry_chunks:
        retry_result = _classify_batch_complete(
            structured_model,
            model_name,
            retry_batch,
            failure_rates,
            exec_times,
            min_chunk=min_chunk,
        )
        repaired.extend(retry_result.classifications)

    return BatchClassificationResult(classifications=repaired)


def run_filter_agent(
    dataset_path: str,
    batch_size: int = 40,
    filter_model: str = "gpt-5-nano",
    inter_batch_sleep: float = 0.0,
) -> FilterResult:
    """Run the Filter Agent over the dataset.

    Returns a FilterResult containing high_risk_tests (T1-T5),
    low_signal_tests (T6), and metadata.

    inter_batch_sleep: seconds to sleep between consecutive filter LLM calls
    (default 0.0 — backward-compatible). Set > 0 to throttle on rate-limited
    providers.
    """
    risk_profiles = extract_risk_profiles(dataset_path, sparse=True)
    failure_rates = {}
    exec_times = extract_exec_times(dataset_path)

    total_tests = len(risk_profiles)
    batches = _chunk(risk_profiles, batch_size)

    structured_model = _build_structured_model(filter_model)
    result = FilterResult()

    for batch_idx, batch in enumerate(batches):
        parsed = _classify_batch_complete(structured_model, filter_model, batch, failure_rates, exec_times)
        if inter_batch_sleep > 0 and batch_idx < len(batches) - 1:
            time.sleep(inter_batch_sleep)

        # merge classifications into result
        batch_test_ids = {p["test"] for p in batch}
        classified_ids = set()
        unknown_ids = set()

        for cls in parsed.classifications:
            if cls.test_id not in batch_test_ids:
                unknown_ids.add(cls.test_id)
                continue
            classified_ids.add(cls.test_id)
            entry = {
                "test_id": cls.test_id,
                "tier": int(cls.tier),
                "key_signals": cls.key_signals,
                "avg_exec_time": exec_times.get(cls.test_id, 0.0),
            }
            if entry["tier"] <= 5:
                result.high_risk_tests.append(entry)
            else:
                result.low_signal_tests.append(entry)

        if unknown_ids:
            logger.warning(
                "Filter LLM returned %d unknown test ID(s); ignoring: %s",
                len(unknown_ids),
                sorted(int(tid) for tid in unknown_ids)[:10],
            )

        missed = batch_test_ids - classified_ids
        if missed:
            raise ValueError(
                "Filter LLM omitted "
                f"{len(missed)} test(s) after retry: "
                f"{sorted(int(tid) for tid in missed)[:10]}"
            )

    all_expected_ids = {p["test"] for p in risk_profiles}
    all_classified_ids = {
        entry["test_id"] for entry in result.high_risk_tests + result.low_signal_tests
    }
    missing_after_batches = all_expected_ids - all_classified_ids
    if missing_after_batches:
        raise ValueError(
            "Filter Agent missing "
            f"{len(missing_after_batches)} test(s) after all retries: "
            f"{sorted(int(tid) for tid in missing_after_batches)[:10]}"
        )

    duplicate_count = (
        len(result.high_risk_tests) + len(result.low_signal_tests) - len(all_classified_ids)
    )
    if duplicate_count:
        logger.warning(
            "Filter LLM returned %d duplicate classification row(s); keeping first occurrence",
            duplicate_count,
        )

    # Keep the LLM's classification choices. If structured output duplicates a
    # test ID, preserve the first LLM row.
    deduped: dict[int, dict] = {}
    for entry in result.high_risk_tests + result.low_signal_tests:
        tid = entry["test_id"]
        if tid not in deduped:
            deduped[tid] = entry

    merged = list(deduped.values())
    result.high_risk_tests = [e for e in merged if e["tier"] <= 5]
    result.low_signal_tests = [e for e in merged if e["tier"] > 5]

    post_dedupe_ids = {
        entry["test_id"] for entry in result.high_risk_tests + result.low_signal_tests
    }
    unknown_after_dedupe = post_dedupe_ids - all_expected_ids
    if unknown_after_dedupe:
        raise ValueError(
            "Filter Agent produced unknown test ID(s) after duplicate cleanup: "
            f"{sorted(int(tid) for tid in unknown_after_dedupe)[:10]}"
        )

    missing_after_dedupe = all_expected_ids - post_dedupe_ids
    if missing_after_dedupe:
        raise ValueError(
            "Filter Agent missing "
            f"{len(missing_after_dedupe)} test(s) after duplicate cleanup: "
            f"{sorted(int(tid) for tid in missing_after_dedupe)[:10]}"
        )

    result.metadata = {
        "total_tests": total_tests,
        "num_batches": len(batches),
        "batch_size": batch_size,
        "high_risk_count": len(result.high_risk_tests),
        "low_signal_count": len(result.low_signal_tests),
        "filter_model": filter_model,
    }

    return result
