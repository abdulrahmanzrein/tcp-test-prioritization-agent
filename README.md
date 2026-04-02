# TCP Test Prioritization Agent

Undergrad research project — an LLM-based agent that ranks regression tests so the ones most likely to fail run first. No model training, no manual feature engineering. Claude reads CI/CD data through tools and decides the order on its own.

We compare it against a traditional ML baseline (Random Forest) that trains on the same ~150 features.

---

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="your-key-here"   # from console.anthropic.com

# single dataset
PYTHONPATH=src python3 scripts/run_llm_agent.py --data datasets/Angel-ML@angel.csv

# all 25 repos (batch)
PYTHONPATH=src python3 scripts/run_llm_agent.py --data-dir datasets --quiet

# ML baseline (no API key needed)
PYTHONPATH=src python3 scripts/run_agent.py --data datasets/Angel-ML@angel.csv
```

---

## How It Works

The agent runs in a **LangGraph loop** — Claude (Sonnet 4.6) picks which tools to call, reads the results, and repeats until it has enough info to rank every test. There's a **10-call safety limit** to prevent runaway loops.

```
"prioritize tests for the next build"
     │
     ▼
┌─────────────────────────────────┐
│       LangGraph Agent Loop      │
│                                 │
│  Claude picks tools → reads     │
│  results → picks more → done    │
│                                 │
│  4 tools:                       │
│   • get_test_risk_profile       │
│   • get_all_failure_rates       │
│   • get_test_complexity         │
│   • get_covered_code_risk       │
└─────────────────────────────────┘
     │
     ▼
ranked JSON with a reason per test
```

Evaluation uses **per-build splitting**: the most recent build is held out as the target, and the agent only sees historical builds. This simulates a real CI scenario where you're predicting which tests will fail in the *next* build.

---

## What the Tools Expose (~150 features)

The dataset has pre-extracted features from the [TCP-CI](https://github.com/icse2020/tcp-ci) benchmark. The agent accesses them through four tools:

| Tool | What it returns | Key features |
|------|----------------|--------------|
| `get_test_risk_profile` | Latest-build snapshot per test (strongest signals) | `REC_RecentFailRate`, `REC_TotalFailRate`, `REC_LastVerdict`, `DET_COV_C_Faults`, `COV_ChnScoreSum` |
| `get_all_failure_rates` | Historical failure rate (0–1) per test | `failure_rate` |
| `get_test_complexity` | Test file complexity + ownership (37 features) | `TES_COM_SumCyclomatic`, `TES_COM_MaxNesting`, `TES_PRO_CommitCount` |
| `get_covered_code_risk` | Risk of production code each test covers (81 features) | `COD_COV_COM_C_SumCyclomatic`, `COD_COV_CHN_C_LinesAdded` |

The system prompt tells Claude to call the first two in parallel, then the last two in parallel — getting all features in **2 tool rounds**.

---

## Tiered Ranking Strategy

The system prompt defines a strict priority order. Claude applies it based on what the tools return:

| Tier | Rule | Example signal |
|------|------|----------------|
| T1 | Always-failing tests | `REC_TotalFailRate ≥ 0.9` |
| T2 | Recently failing | `REC_RecentFailRate > 0` and `REC_LastVerdict = 1` |
| T3 | Covers code with known faults | `DET_COV_C_Faults > 0` or `DET_COV_IMP_Faults > 0` |
| T4 | Historical failures but not recent | `failure_rate > 0`, `REC_RecentFailRate = 0` |
| T5 | High-coverage safety nets | Never failed, high `COV_ChnScoreSum` or complexity |
| T6 | Everything else | Fastest first |

Within tiers, ties are broken by: higher failure rate → higher coverage → lower execution cost.

---

## Example Output

```
  #1 test 5140 — I'm ranking this first because REC_TotalFailRate=1.0 and
     REC_RecentFailRate=1.0 — it fails every single build.
  #2 test 5141 — Same 100% failure rate, slightly faster execution time.
  #3 test 2161 — REC_RecentFailRate=1.0, LastVerdict=1, fastest recently-
     failing test at ~20ms.
  ...
  #46 test 2757 — Zero failures, no fault signal. Ranked last by cost.

APFD=0.8600  APFDc=0.9327  P@10=0.7000
```

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **APFD** | How early failures appear in the ranking (0–1, higher = better, random ≈ 0.5) |
| **APFDc** | Same idea but weighted by test execution cost — finding failures in fast tests is worth more |
| **Precision@10** | Of the top 10 ranked tests, how many actually failed? |

---

## Running All 25 Repos

The `datasets/` folder contains CSVs from 25 open-source Java projects (TCP-CI benchmark). To evaluate across all of them:

```bash
PYTHONPATH=src python3 scripts/run_llm_agent.py --data-dir datasets --quiet
```

Each repo gets one line:
```
[1/25] Angel-ML@angel.csv       APFD=0.8600  APFDc=0.9327  P@10=0.7000
[2/25] CompEvol@beast2.csv       APFD=...     APFDc=...     P@10=...
...
```

Options:
- `--quiet` — suppress per-test rankings, just print metrics
- `--gap 65` — seconds between runs to stay under API rate limits (default 65)
- `--gap 0` — no waiting (faster, but may hit 429 errors)

---

## Project Structure

```
src/tcp_agent/
├── agent/
│   ├── tcp_agent.py          # LangGraph agent — system prompt, tool loop, JSON parsing
│   └── ranker.py             # merges Claude's ranking with actual Verdict + Duration
├── tools/
│   ├── history_tool.py       # get_all_failure_rates, get_test_risk_profile
│   ├── complexity_tool.py    # get_test_complexity (TES_COM_ + TES_PRO_)
│   └── covered_code_risk_tool.py  # get_covered_code_risk (COD_COV_*)
├── evaluation.py             # APFD, APFDc, Precision@k
├── data_loader.py            # loads dataset (ML baseline)
├── features.py               # SMOTE + feature cleaning (ML baseline)
├── model.py                  # Random Forest (ML baseline)
└── ranking.py                # probability-based ranking (ML baseline)

scripts/
├── run_llm_agent.py          # runs the LLM agent (single or batch)
└── run_agent.py              # runs the ML baseline
```

---

## How It All Connects

```
run_llm_agent.py
  │  reads CSV → splits into history builds + target build
  │  writes history to /tmp/history.csv
  ▼
tcp_agent.py (run_agent)
  │  builds LangGraph: llm_call ↔ tool_node (max 10 rounds)
  │  Claude calls tools → reads results → repeats
  │  final message = JSON array of ranked tests
  ▼
ranker.py (build_ranked_df)
  │  normalizes Claude's output (dedupes, validates types)
  │  merges with target build's Verdict + Duration
  ▼
evaluation.py
  │  scores: APFD, APFDc, Precision@10
  ▼
prints metrics
```

---

## Notes

- Datasets come from the [TCP-CI](https://github.com/icse2020/tcp-ci) benchmark (RTP-Torrent). Each CSV has pre-extracted features.
- The ML baseline is kept for comparison. GitHub Actions runs it on every push.
- Claude isn't fully deterministic even at temperature=0, so APFDc can vary slightly between runs.
- Rate limits: free-tier Anthropic API has a 30k tokens/minute cap. The `--gap` flag spaces requests out so batch runs don't get throttled. The agent also retries automatically on 429 errors.
- The `tools/backup/` folder has older tools from earlier iterations — not wired into the current agent.
