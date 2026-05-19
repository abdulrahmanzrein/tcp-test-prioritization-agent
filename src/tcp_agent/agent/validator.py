from __future__ import annotations

"""
Output Validation Layer

Sits between the LLM ranking and final result acceptance.  Catches
malformed, incomplete, or rule-violating output before it silently
degrades TCP quality.
"""

import logging
from dataclasses import dataclass, field

from tcp_agent.agent.filter_agent import FilterResult

logger = logging.getLogger(__name__)


# ── Validation result ────────────────────────────────────────────────

@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self):
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"Validation: {status}"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN:  {w}")
        return "\n".join(parts)


# ── Core validation ──────────────────────────────────────────────────

def validate_ranking(
    ranked: list[dict],
    expected_test_ids: set,
    filter_result: FilterResult | None = None,
) -> ValidationResult:
    """Validate a ranked list of tests against completeness, uniqueness,
    priority, and tier-rule constraints.

    Parameters
    ----------
    ranked : list[dict]
        The ranked output — each dict must have test, priority, confidence, reason.
    expected_test_ids : set[int]
        All test IDs that should be present in the ranking.
    filter_result : FilterResult | None
        If provided, used to check tier-rule constraints (T6 not ahead of T1-T5).

    Returns
    -------
    ValidationResult
    """
    vr = ValidationResult()

    # ── Schema check ─────────────────────────────────────────────────
    required_keys = {"test", "priority", "confidence", "reason"}
    for i, item in enumerate(ranked):
        if not isinstance(item, dict):
            vr.errors.append(f"Item {i} is not a dict: {type(item)}")
            vr.is_valid = False
            continue
        missing = required_keys - set(item.keys())
        if missing:
            vr.errors.append(f"Item {i} (test={item.get('test', '?')}) missing keys: {missing}")
            vr.is_valid = False

    if not vr.is_valid:
        return vr  # can't do further checks with broken schema

    # ── Extract test IDs from ranked list ────────────────────────────
    ranked_ids = []
    for item in ranked:
        try:
            ranked_ids.append(int(item["test"]))
        except (ValueError, TypeError):
            vr.errors.append(f"Invalid test ID: {item['test']!r}")
            vr.is_valid = False

    if not vr.is_valid:
        return vr

    ranked_set = set(ranked_ids)

    # ── Completeness: all expected tests present ─────────────────────
    missing = expected_test_ids - ranked_set
    if missing:
        vr.errors.append(f"Missing {len(missing)} test(s): {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        vr.is_valid = False

    # ── No unknowns: no extra test IDs ───────────────────────────────
    unknown = ranked_set - expected_test_ids
    if unknown:
        vr.errors.append(f"Unknown test ID(s) not in dataset: {sorted(unknown)[:10]}")
        vr.is_valid = False

    # ── No duplicates ────────────────────────────────────────────────
    if len(ranked_ids) != len(ranked_set):
        seen = set()
        dupes = set()
        for tid in ranked_ids:
            if tid in seen:
                dupes.add(tid)
            seen.add(tid)
        vr.errors.append(f"Duplicate test ID(s): {sorted(dupes)}")
        vr.is_valid = False

    # ── Priority validity: sequential from 1 ────────────────────────
    priorities = [item["priority"] for item in ranked]
    expected_priorities = list(range(1, len(ranked) + 1))
    if sorted(priorities) != expected_priorities:
        vr.warnings.append(
            f"Priorities are not sequential 1..{len(ranked)}. "
            f"Got range [{min(priorities)}..{max(priorities)}], "
            f"unique count={len(set(priorities))}"
        )
        # this is a warning, not an error — the ranker module handles re-numbering

    # ── Tier rule enforcement ────────────────────────────────────────
    if filter_result is not None:
        high_risk_ids = {t["test_id"] for t in filter_result.high_risk_tests}
        low_signal_ids = {t["test_id"] for t in filter_result.low_signal_tests}

        # find the worst (highest) priority assigned to any high-risk test
        hr_priorities = [
            item["priority"] for item in ranked
            if int(item["test"]) in high_risk_ids
        ]
        # find the best (lowest) priority assigned to any low-signal test
        ls_priorities = [
            item["priority"] for item in ranked
            if int(item["test"]) in low_signal_ids
        ]

        if hr_priorities and ls_priorities:
            worst_hr = max(hr_priorities)
            best_ls = min(ls_priorities)
            if best_ls < worst_hr:
                vr.warnings.append(
                    f"Tier rule violation: T6 test appears at priority {best_ls} "
                    f"but a T1-T5 test is at priority {worst_hr}. "
                    f"T6 tests should always be in the tail."
                )

    # ── Confidence range ─────────────────────────────────────────────
    for item in ranked:
        conf = item.get("confidence", 0)
        if not (0.0 <= conf <= 1.0):
            vr.warnings.append(
                f"Test {item['test']} has confidence={conf} outside [0,1]"
            )

    return vr


def log_validation_errors(validation: ValidationResult):
    """Log all validation errors and warnings."""
    for e in validation.errors:
        logger.error(f"Validation error: {e}")
    for w in validation.warnings:
        logger.warning(f"Validation warning: {w}")
