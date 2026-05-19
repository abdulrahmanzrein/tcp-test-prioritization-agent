"""Bug 2 regression test: the results CSV column must be named recall_at_10
because the value stored is failure_recall_at_k (not precision_at_k).

Before the fix, the column was named p_at_10 but the value stored was recall.
After the fix, the column name reflects what's actually computed.
"""
from __future__ import annotations

import csv
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_llm_agent import _append_result  # noqa: E402


def test_append_result_uses_recall_at_10_column(tmp_path: Path):
    csv_path = tmp_path / "out.csv"
    row = {
        "dataset": "demo.csv",
        "apfd": "0.7", "apfdc": "0.8", "recall_at_10": "0.5",
        "filter_model": "m1", "ranking_model": "m2",
        "eval_window": "5", "failed_builds_only": "1",
        "wall_seconds": "1.0", "timestamp": "now",
        "status": "ok", "error": "",
    }

    _append_result(csv_path, threading.Lock(), row)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        first_row = next(reader)

    assert "recall_at_10" in header, f"Expected recall_at_10 in header, got {header}"
    assert "p_at_10" not in header, (
        f"Expected p_at_10 NOT in header (renamed), got {header}"
    )
    assert first_row["recall_at_10"] == "0.5"
