"""
In-memory CSV cache for the TCP agent.

Datasets are immutable during a run, but tools and feature extractors each
parse them independently — costing ~700ms-3s per call on the larger
TCP-CI datasets. With 16+ reads per build × 5 builds × 25 datasets, that
adds up to 30-40 minutes of redundant pandas work per full evaluation.

This module provides a single thread-safe entry point that parses each CSV
once and hands back the same `pandas.DataFrame` on every subsequent call.

Safety
------
The cached DataFrame is shared, never copied. All call sites in this
codebase use only non-mutating operations (`sort_values`, `groupby`,
`assign`, boolean indexing, `to_dict`), all of which return new
objects and leave the cached frame untouched. **Do not mutate the
returned DataFrame in place** (no `df["col"] = ...`, no
`df.sort_values(inplace=True)`, no `df.drop(..., inplace=True)`); make a
`.copy()` first if you need to.

Thread safety
-------------
The first caller for a given path acquires the lock and parses the CSV;
concurrent callers wait microseconds and receive the cached frame.
This is important because ranking batches and `--workers` evaluation
both run multiple threads that may load the same dataset simultaneously.
"""

from __future__ import annotations

import threading
from typing import Dict

import pandas as pd

_cache: Dict[str, pd.DataFrame] = {}
_lock = threading.Lock()


def load_dataset(path: str) -> pd.DataFrame:
    """Return the parsed DataFrame for `path`, reading from disk only on the
    first call. Subsequent calls (from any thread) reuse the cached frame.
    """
    with _lock:
        df = _cache.get(path)
        if df is None:
            df = pd.read_csv(path)
            _cache[path] = df
        return df


def clear_cache() -> None:
    """Drop all cached frames. Useful for tests or long-running processes
    that need to free memory after a dataset is fully evaluated.
    """
    with _lock:
        _cache.clear()
