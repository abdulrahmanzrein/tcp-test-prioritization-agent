# TCP Test Prioritization Agent

Undergrad research project exploring whether an LLM can replace trained ML models for test case prioritization (TCP) in Continuous Integration. The agent (Claude Haiku 4.5 via LangGraph) reads ~150 pre-extracted features through tools and ranks regression tests by failure likelihood -- no model training, no manual feature engineering.

Built on the **TCP-CI** framework and dataset from Yaraghi et al. (2022), validated against 25 open-source Java projects. Compared against a traditional Random Forest baseline trained on the same features.

---

## Research Foundation

This project is grounded in two papers:

1. **Yaraghi et al. (2022)** -- *"Scalable and Accurate Test Case Prioritization in Continuous Integration Contexts"*
   - Established the TCP-CI dataset: 25 Java projects, 150+ features across 9 groups
   - Key finding: execution history features (REC) dominate -- REC alone achieves near-full-model APFDc (CL=0.51 vs Full_M)
   - Feature importance (Table 12): REC_Age (#1, freq 17,034), TES_PRO_OwnersExperience (#2, freq 10,618), TES_PRO_AllCommitersExperience (#3, freq 7,643)
   - Coverage features (COV/COD_COV) have lowest marginal value (CL=0.79 vs Full_M)
   - Random Forest pairwise ranking achieves APFDc ~0.82

2. **Mendoza et al. (2022)** -- *"On the Effectiveness of Data Balancing Techniques in ML-Based TCP"*
   - Only ~3% of test executions fail in typical CI projects -- extreme class imbalance
   - SMOTE/Random Over-sampling improve APFDc by avg 0.06 on the same TCP-CI dataset

Our agent uses these findings to define a research-grounded tiered ranking strategy that prioritizes REC features, uses TES_PRO experience metrics for tiebreaking, and treats coverage as a low-weight signal -- matching the empirical feature importance hierarchy from the papers.

---

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="your-key-here"   # from console.anthropic.com

# single dataset
python scripts/run_llm_agent.py --data datasets/Angel-ML@angel.csv

# all 25 datasets (batch)
python scripts/run_llm_agent.py --data-dir datasets/ --quiet

# ML baseline (no API key needed)
python scripts/run_agent.py --data datasets/Angel-ML@angel.csv
```

---

## How It Works

The agent runs in a **LangGraph loop** -- Claude (Haiku 4.5) picks which tools to call, reads the results, and repeats until it has enough data to rank every test. The ranking is then extracted via structured output (Pydantic schema) to guarantee valid JSON.

```
"prioritize tests for the next build"
     |
     v
+-----------------------------------+
|       LangGraph Agent Loop        |
|                                   |
|  Claude picks tools -> reads      |
|  results -> picks more -> done    |
|                                   |
|  4 tools:                         |
|   - get_test_risk_profile         |
|   - get_all_failure_rates         |
|   - get_test_complexity           |
|   - get_covered_code_risk         |
+-----------------------------------+
     |
     v
Structured output (Pydantic schema)
     |
     v
Ranked JSON with reasons per test
```

The system prompt instructs Claude to call `get_test_risk_profile` and `get_all_failure_rates` in parallel first, then `get_test_complexity` and `get_covered_code_risk` in parallel -- getting all ~150 features in 2 tool rounds.

Evaluation uses **per-build splitting**: the most recent build is held out as the target, and the agent only sees historical builds. This simulates a real CI scenario where you're predicting which tests will fail in the *next* build.

---

## Feature Groups (~150 features)

The TCP-CI dataset (Yaraghi et al., Table 3) defines 9 feature groups. Our tools expose all of them:

| Tool | Feature Groups | Count | Key Features |
|------|---------------|-------|-------------|
| `get_test_risk_profile` | REC (execution history) + DET_COV (fault detection) + COV (coverage) + TES_CHN (test churn) | 32 | `REC_Age`, `REC_RecentFailRate`, `REC_TotalFailRate`, `REC_LastVerdict`, `DET_COV_C_Faults`, `COV_ChnScoreSum` |
| `get_all_failure_rates` | Derived from Verdict history | 1 | `failure_rate` |
| `get_test_complexity` | TES_COM (test complexity) + TES_PRO (test process) | 37 | `TES_COM_SumCyclomatic`, `TES_COM_MaxNesting`, `TES_PRO_CommitCount`, `TES_PRO_OwnersExperience` |
| `get_covered_code_risk` | COD_COV_COM + COD_COV_PRO + COD_COV_CHN (covered code) | 81 | `COD_COV_COM_C_SumCyclomatic`, `COD_COV_CHN_C_LinesAdded`, `COD_COV_PRO_C_DistinctDevCount` |

**Why REC-heavy?** Yaraghi et al. RQ2.3 shows REC_M alone vs Full_M has CL=0.51 -- the full model barely outperforms REC-only. Coverage features (COV_M) have CL=0.79 vs Full_M -- they lose ~79% of comparisons. Our tier system reflects this: REC-based tiers (T1, T2, T4) outrank coverage-based tiers (T3, T5).

### Feature Definitions from the Papers

- **REC_Age** (Section 2.2): Number of builds since the test first appeared. The #1 most-used feature in RF models. Older tests have more historical signal.
- **COV_ChnScoreSum** (Section 2.2, Definition 1): Not traditional code coverage -- uses association-rule mining to measure co-change confidence between test and production files.
- **DET_COV_C_Faults** (Section 2.2, Definition 2): Previously Detected Faults -- count of known bugs in production code the test covers. Buggy code tends to stay buggy.
- **TES_PRO_OwnersExperience** (Table 12): Primary author's contribution proportion. Low values indicate inexperienced ownership -- the #2 most important feature overall.
- **TES_CHN_DMM\*** (Section 2.2): Delta Maintainability Model metrics measuring recent changes to test file size, complexity, and interface.

---

## Tiered Ranking Strategy

The system prompt defines a strict priority order grounded in the papers' findings. The optimal ordering from Yaraghi et al. Section 2.1: *"test cases are first sorted by their verdict with failed test cases at the beginning, second sorted by execution time ascending."*

| Tier | Rule | Research Basis |
|------|------|---------------|
| **T1 -- Persistent failures** | `REC_TotalFailRate >= 0.9`. Cheapest first. | Near-guaranteed failures. Fastest detection at minimal CI cost. |
| **T2 -- Recent/active failures** | `REC_RecentFailRate > 0` AND `REC_LastVerdict = 1`. Sort by fail rate desc, then cost asc. | Actively broken tests. "Recent" = last 6 builds (Yaraghi Section 2.2). |
| **T3 -- Fault-adjacent** | `DET_COV_C_Faults > 0` OR `DET_COV_IMP_Faults > 0`. Prefer with `COV_ChnScoreSum > 0`. | Previously Detected Faults (Definition 2). Changed buggy code = highest risk. |
| **T4 -- Historical failures** | `failure_rate > 0` but `REC_RecentFailRate = 0`. Sort by rate desc, factor in `owners_experience`. | Past failures + low ownership experience (#2 feature) = regression risk. |
| **T5 -- High-signal, never failed** | `COV_ChnScoreSum > 0`, high complexity, low experience, or high `covered_code_risk_score`. | Risk indicators without failure history. Coverage as tiebreaker (CL=0.79). |
| **T6 -- Low-signal remainder** | No failure history, no risk signal. Cheapest first. | ~97% of tests never fail (Mendoza et al.). Minimize wasted CI time. |

---

## Evaluation Metrics

| Metric | Definition | Source |
|--------|-----------|--------|
| **APFD** | How early failures appear in the ranking (0-1, random ~ 0.5) | Rothermel et al. (1999) |
| **APFDc** | Cost-cognizant APFD -- weighted by test execution time. Finding failures in fast tests scores higher. | Yaraghi et al. Section 4.3 |
| **Precision@10** | Of the top 10 ranked tests, how many actually failed? | Standard IR metric |

APFDc is the primary metric, matching Yaraghi et al. and Mendoza et al. evaluation methodology.

---

## ML Baseline (Random Forest)

The RF baseline (`scripts/run_agent.py`) trains a Random Forest classifier on the same ~150 features for comparison:

- **SMOTE** for class balancing (~3% failure rate, justified by Mendoza et al.)
- 80/20 train/test split with `random_state=7`
- Ranks tests by predicted failure probability

Note: this uses a random split, not the walk-forward protocol from the papers. The paper's RF uses pairwise ranking trained per-build. Our simplified RF is included for reproducible comparison, not as an exact replication of their model.

```bash
python scripts/run_agent.py --data datasets/Angel-ML@angel.csv
```

---

## Running All 25 Datasets

The `datasets/` folder contains CSVs from 25 open-source Java projects (TCP-CI benchmark):

```bash
# LLM agent (requires API key)
python scripts/run_llm_agent.py --data-dir datasets/ --quiet

# ML baseline (local, no API key)
python scripts/run_agent.py --data datasets/Angel-ML@angel.csv
```

Output:
```
[1/25] Angel-ML@angel.csv       APFD=0.8600  APFDc=0.9327  P@10=0.7000
[2/25] CompEvol@beast2.csv       APFD=...     APFDc=...     P@10=...
...
```

Options:
- `--quiet` -- suppress per-test rankings, just print metrics
- `--gap 65` -- seconds between runs to stay under API rate limits (default 65)

---

## Project Structure

```
src/tcp_agent/
  agent/
    tcp_agent.py              # LangGraph agent + system prompt + structured output
    ranker.py                 # merges ranking with actual Verdict/Duration (outer merge)
  tools/
    history_tool.py           # get_all_failure_rates, get_test_risk_profile
    complexity_tool.py        # get_test_complexity (37 raw TES_COM + TES_PRO features)
    covered_code_risk_tool.py # get_covered_code_risk (81 raw COD_COV features)
  evaluation.py               # APFD, APFDc, Precision@k
  data_loader.py              # CSV loader (ML baseline)
  features.py                 # SMOTE + feature cleaning (ML baseline)
  model.py                    # Random Forest classifier (ML baseline)
  ranking.py                  # probability-based ranking (ML baseline)

scripts/
  run_llm_agent.py            # runs the LLM agent (single or batch)
  run_agent.py                # runs the ML baseline

datasets/                     # 25 CSVs from TCP-CI benchmark
```

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **REC-heavy tier system** | Yaraghi Table 12: REC_Age (#1), OwnersExperience (#2), AllCommitersExperience (#3). REC_M alone has CL=0.51 vs Full_M. |
| **Coverage as tiebreaker only** | COV_M has CL=0.79 vs Full_M -- loses ~79% of comparisons (Yaraghi Table 10). |
| **Structured output (Pydantic)** | Eliminates JSON parsing failures. Agent reasons freely, then output is extracted into a schema. |
| **Outer merge in ranker** | Tests the LLM misses get worst rank instead of silently vanishing -- prevents inflated APFDc. |
| **All raw features returned** | Tools return full feature vectors (37 for complexity, 81 for covered code risk) -- no summarization. |
| **Optimal ordering within tiers** | Yaraghi Section 2.1: failed first, then by execution time ascending. Applied as within-tier tiebreaker. |

---

## Differences from the Papers

| Aspect | Papers | Our Agent |
|--------|--------|-----------|
| **Model** | Random Forest pairwise ranking | LLM (Claude Haiku 4.5) with heuristic tiers |
| **Training** | Trained per-build, walk-forward | Zero training -- prompt-based reasoning |
| **Feature access** | All 150+ features as numeric vectors | Same features via tool calls (full raw values) |
| **Evaluation** | Last 50 failed builds, mean APFDc +/- stddev | Per-build splitting (single target build) |
| **Class balancing** | SMOTE / Random Over-sampling | Not applicable (no training) |
| **FF test handling** | Three-sigma outlier removal | Ranked in T1 (always-failing) |

---

## Notes

- **Determinism**: Claude isn't fully deterministic even at `temperature=0`, so APFDc can vary slightly between runs.
- **Rate limits**: The `--gap` flag spaces batch requests (default 65s). The agent retries automatically on 429 errors (up to 5 retries with 65s waits).
- **Datasets**: From the [TCP-CI](https://github.com/icse2020/tcp-ci) benchmark (RTP-Torrent). Each CSV has pre-extracted features.
