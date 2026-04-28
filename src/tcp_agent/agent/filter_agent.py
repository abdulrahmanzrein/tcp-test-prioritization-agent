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


def _invoke_with_retry(model, messages, max_retries: int = 6):
    """Minimal retry: exponential backoff on transient errors, re-raise length/auth."""
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()
            if name == "LengthFinishReasonError" or "length limit" in msg:
                raise
            if "401" in str(e) or "403" in str(e) or "invalid_api_key" in msg:
                raise
            if attempt == max_retries - 1:
                raise
            wait = 65 if "rate" in msg or "429" in str(e) else min(2 ** attempt, 30)
            print(f"  [filter-RETRY] {attempt + 1}/{max_retries} {name}: {str(e)[:120]}", flush=True)
            time.sleep(wait)


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

## What the research says (Yaraghi et al. 2022, TCP-CI, IEEE TSE)

The Random Forest TCP model trained on the full feature set achieves the
best APFDc. Its top-15 most predictive features (Table 12) are dominated
by THREE feature groups:
  • REC      — execution history    (REC_Age is feature #1 paper-wide)
  • TES_PRO  — test ownership        (OwnersExperience is #2,
                                      AllCommitersExperience is #3)
  • TES_COM  — test source code      (size, complexity, comment ratio)
Coverage features (COV / DET_COV / COD_COV) are the WEAKEST group.
On some subjects (e.g. thinkaurelius/titan) TES_M (TES_PRO + TES_COM +
TES_CHN) is the single best feature group, beating REC_M. So you MUST
treat TES_PRO and TES_COM signals as first-class evidence.

The best single-feature heuristic the paper found is REC_TotalFailRate
(descending). Tests that failed in the past tend to fail again; this
remains the strongest individual signal.

REC_Age key insight (Figure 5 of the paper): failures drop sharply after
the first 10–20% of a test's lifetime. Newly added tests are MUCH more
likely to fail than mature tests. Treat low REC_Age as a real risk signal.

## Tier criteria — ordered, take the HIGHEST that applies

T1 — Persistent failure
       REC_TotalFailRate >= 0.5
       (test has been broken in a large fraction of its history)

T2 — Recent / active failure
       REC_RecentFailRate > 0
       OR REC_LastVerdict != 0
       OR REC_LastFailureAge in {0, 1, 2}
       (don't require BOTH RecentFailRate>0 AND LastVerdict=1 — either is
       sufficient. A test that flipped pass→fail→pass in recent builds
       still belongs here.)

T3 — Fault-adjacent
       DET_COV_C_Faults > 0  OR  DET_COV_IMP_Faults > 0
       (covers code that has a history of bugs — buggy code stays buggy)

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
       • COV_ChnScoreSum > 0  OR  COV_ImpScoreSum > 0
         (test covers files that changed in this build)
       • REC_Age <= 5
         (newly created test; failure prob. is highest in the first 10–20%
         of a test's life — paper Fig. 5)
       • TES_PRO_OwnersExperience low (<= 0.5)
         (test owned by a relatively new contributor)
       • TES_PRO_CommitCount >= 3
         (test under active development — many recent edits)
       • TES_PRO_MinorContributorCount >= 2
         (multiple unfamiliar contributors → higher fault risk)
       • TES_COM_SumCyclomatic >= 20  OR  TES_COM_CountStmtExe >= 50
         (large / complex test → exercises more code, more fault paths)
       • REC_TotalMaxExeTime > 60s  OR  REC_LastExeTime > 60s
         (long-running tests cover more code; treat as high-signal)

T6 — Low-signal remainder
     ONLY if every check above is false. To land in T6 a test should look
     like: stable history, no recent edits, no coverage of changed code,
     mature age, experienced owner, low complexity, short execution.
     When unsure between T5 and T6, choose T5 — false positives are
     cheap (the Ranking Agent re-sorts them) but false negatives are
     expensive (a real failure gets buried).

## Rules

- Pick the SINGLE highest tier that applies (T1 > T2 > T3 > T4 > T5 > T6).
- A missing feature (absent key) = "no data" — do NOT treat it as zero.
  Real zeros (e.g. REC_LastVerdict=0) ARE present and ARE meaningful.
- A value of -1 means "no data" / "never observed" — IGNORE it; do NOT
  use it as evidence in either direction.
- Numeric thresholds above are guidelines, not hard cutoffs. Use them to
  build intuition; if a test is borderline-but-trending-risky, choose
  the higher tier.
- Output: test_id (int), tier (int 1-6), and 1 (max 2) key_signals.
- key_signals MUST be short "feature=value" strings, ≤30 chars each.
  Example: ["REC_TotalFailRate=0.92"], ["REC_Age=2"], ["TES_CHN_LinesAdded=14"].
  No prose, no explanations.
- You MUST classify EVERY test in the batch — do not skip any.
"""


# ── Helper: LLM call with retry ──────────────────────────────────────

def _resolve_provider(model_name: str) -> str | None:
    name = model_name.lower()
    if "gemini" in name:
        return "google_genai"
    if name.startswith("claude"):
        return "anthropic"
    return None


def _build_structured_model(model_name: str):
    """Initialize a chat model and bind the BatchClassificationResult schema."""
    provider = _resolve_provider(model_name)
    base = init_chat_model(model_name, model_provider=provider, temperature=0)
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
    batch: list[dict],
    failure_rates: dict,
    exec_times: dict,
    min_chunk: int = 8,
):
    """Classify a single batch. Auto-splits on output-length errors."""
    prompt = _build_batch_prompt(batch, failure_rates, exec_times)
    try:
        return _invoke_with_retry(
            structured_model,
            [
                SystemMessage(content=FILTER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
        )
    except Exception as e:
        if not _is_length_error(e) or len(batch) <= min_chunk:
            raise
        mid = len(batch) // 2
        print(
            f"   ⚠️  Filter batch hit output-length limit "
            f"({len(batch)} tests) — splitting into {mid} + {len(batch) - mid}"
        )
        left = _classify_batch(structured_model, batch[:mid], failure_rates, exec_times, min_chunk)
        right = _classify_batch(structured_model, batch[mid:], failure_rates, exec_times, min_chunk)
        return BatchClassificationResult(
            classifications=list(left.classifications) + list(right.classifications)
        )


def run_filter_agent(
    dataset_path: str,
    batch_size: int = 100,
    filter_model: str = "gpt-4o-mini",
) -> FilterResult:
    """Run the Filter Agent over the dataset.

    Returns a FilterResult containing high_risk_tests (T1-T5),
    low_signal_tests (T6), and metadata.
    """
    risk_profiles = extract_risk_profiles(dataset_path, sparse=True)
    failure_rates = extract_failure_rates(dataset_path)
    exec_times = extract_exec_times(dataset_path)

    total_tests = len(risk_profiles)
    batches = _chunk(risk_profiles, batch_size)

    structured_model = _build_structured_model(filter_model)
    result = FilterResult()

    for batch_idx, batch in enumerate(batches):
        parsed = _classify_batch(structured_model, batch, failure_rates, exec_times)

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

    result.metadata = {
        "total_tests": total_tests,
        "num_batches": len(batches),
        "batch_size": batch_size,
        "high_risk_count": len(result.high_risk_tests),
        "low_signal_count": len(result.low_signal_tests),
        "filter_model": filter_model,
    }

    return result
