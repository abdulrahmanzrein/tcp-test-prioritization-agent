# TCP Test Prioritization Agent

LLM-driven **test case prioritization (TCP)** for **Continuous Integration**, evaluated on the **TCP-CI** CSV benchmarks (Yaraghi et al., IEEE TSE 2022). The pipeline uses **two LLM stages** plus validation—no training step for the LLM path (a separate **Random Forest** script exists for baseline comparison).

---

## What this repo does

1. **Filter Agent** — Classifies every test into **T1–T5** (high-risk) or **T6** (low-signal) using structured output and batched prompts. Features come from **`feature_extractor`** (pure Python on the CSV), not tool calls.
2. **Ranking Agent** — For **T1–T5 only**, runs a **LangGraph** tool loop (`get_test_risk_profile`, `get_test_complexity`, `get_covered_code_risk`), then structured extraction of the final ordering. **T6** is appended deterministically (by average execution time).
3. **Validator** — Checks completeness / IDs / duplicates; on failure, **`deterministic_fallback`** produces a valid ranking from latest CSV rows.

Research context: **Yaraghi et al. (TCP-CI)** — comprehensive features and REC-heavy importance; **Mendoza et al. (PROMISE ’22)** — imbalance and APFDc-focused ML-TCP (your LLM path does not apply SMOTE; the RF baseline does).

---

## Requirements

- Python 3.10+ (recommended)
- API keys in **`.env`** at the project root (loaded by `run_llm_agent.py`):

| Models you use | Set |
|----------------|-----|
| OpenAI (`gpt-4o`, `gpt-4o-mini`, …) | `OPENAI_API_KEY` |
| Anthropic (`claude-…`) | `ANTHROPIC_API_KEY` |
| Google Gemini (`…gemini…`) | `GOOGLE_API_KEY` |

`run_llm_agent.py` inspects **`--filter-model`** and **`--ranking-model`** and requires every provider key implied by those names.

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Main entry point: `scripts/run_llm_agent.py`

Rolling **evaluation**: for each of the last **`--eval-window`** builds, the agent sees only **history** (`Build < target`), writes it to a temp CSV, runs **`run_agent`** on that file, then scores the predicted order against the **target** build’s `Verdict` / `Duration` (**APFD**, **APFDc**, **P@10**).

### CLI reference

| Argument | Default | Role |
|----------|---------|------|
| `--data` *path* | (one required) | Single dataset CSV |
| `--data-dir` *path* | mutually exclusive with `--data` | All `*.csv` in folder |
| `--mode` | `pilot` | `pilot` = CSV pilot; `production` hooks (see `config.py`) |
| `--batch-size` | `100` | Tests per **Filter** LLM batch (auto-splits on length errors) |
| `--filter-model` | `gpt-4o-mini` | Filter LLM id |
| `--ranking-model` | `gpt-4o` | Ranking LLM id |
| `--eval-window` | `5` | Number of most recent builds to average metrics over |
| `--gap` | `65` | Seconds to sleep between full agent runs per build (`0` disables) |
| `--workers` | `1` | Parallel **datasets** when using `--data-dir` (raises API load) |
| `--results-csv` | `results/evaluation_summary.csv` | Append metrics; **resume** skips datasets already listed |
| `--quiet` | off | Less console output for single `--data` runs |

### Commands

```bash
# One dataset (prints mean APFD / APFDc / P@10)
python3 scripts/run_llm_agent.py --data datasets/apache@rocketmq.csv --eval-window 5 --ranking-model gpt-4o

# All CSVs in datasets/ (sequential unless --workers > 1)
python3 scripts/run_llm_agent.py --data-dir datasets/ --eval-window 5 --ranking-model gpt-4o --quiet

# Long run with log
python3 scripts/run_llm_agent.py --data-dir datasets/ --eval-window 5 2>&1 | tee run_output.log
```

### Results file (`--results-csv`)

Append-only **CSV** columns: `dataset`, `apfd`, `apfdc`, `p_at_10`, `filter_model`, `ranking_model`, `eval_window`, `wall_seconds`, `timestamp`, `status`, `error`. Each completed dataset is one row (`fsync` after write). Rerun the same command to **resume** unfinished batches.

---

## ML baseline: `scripts/run_agent.py`

Trains a **Random Forest** on a random **80/20** split (see `model.py` / `features.py`), ranks the holdout by predicted failure probability, prints **APFD**, **APFDc**, **Precision@10**. **No API key.** Not the same walk-forward protocol as the TCP-CI papers; useful as a quick local baseline.

```bash
python3 scripts/run_agent.py --data datasets/Angel-ML@angel.csv
```

(`sys.path` is adjusted so this works without `PYTHONPATH`.)

---

## Metrics (`src/tcp_agent/evaluation.py`)

| Symbol | Meaning |
|--------|--------|
| **APFD** | Average Percentage of Faults Detected — earlier failures in the ordered list score higher (~0.5 random). |
| **APFDc** | Cost-cognizant variant using **`Duration`** (see implementation). |
| **P@10** here | `(# failing tests in the first 10 positions) / 10` — **not** recall; with very few failures, the value is capped (e.g. one failure → max `0.1`). |

Merging LLM output with the target build uses **`build_ranked_df`** (`ranker.py`): **outer** merge so missing tests keep worst priority instead of disappearing.

---

## Ranking internals (summary)

- High-risk tests are processed in batches of **`_RANKING_BATCH_SIZE`** (15). Up to **`_RANKING_PARALLELISM`** batches (4) may run in **threads**—increase only if your API tier supports the extra TPM/RPM.
- **Models** are created with LangChain **`init_chat_model`**; names starting with `claude` use Anthropic, names containing `gemini` use Google, otherwise OpenAI.

---

## Repository layout

```
tcp-test-prioritization-agent/
├── scripts/
│   ├── run_llm_agent.py      # Evaluation + batch runs + CLI
│   └── run_agent.py          # RF baseline (local, --data required)
├── datasets/                 # TCP-CI-style CSVs (25 subjects in this project)
├── results/                  # Default output for --results-csv
├── requirements.txt
├── .env                      # API keys (not committed)
└── src/tcp_agent/
    ├── agent/
    │   ├── tcp_agent.py      # run_agent → Filter → Ranking → validate / fallback
    │   ├── filter_agent.py   # T1–T6 classification
    │   ├── ranking_agent.py  # LangGraph ranking for T1–T5
    │   ├── validator.py      # Checks + deterministic_fallback
    │   └── ranker.py         # normalize_ranked_items, build_ranked_df
    ├── tools/
    │   ├── feature_extractor.py   # CSV features for Filter (no LLM tools)
    │   ├── history_tool.py        # get_test_risk_profile (Ranking)
    │   ├── complexity_tool.py     # get_test_complexity
    │   └── covered_code_risk_tool.py
    ├── data_cache.py         # Thread-safe in-memory CSV parse cache
    ├── data_loader.py        # RF baseline loading
    ├── evaluation.py         # APFD, APFDc, precision_at_k
    ├── config.py             # AgentMode PILOT / PRODUCTION
    └── features.py, model.py, ranking.py   # RF baseline pipeline
```

Tier **definitions** for the Filter live in **`FILTER_SYSTEM_PROMPT`** inside `filter_agent.py`; ranking rules and tool instructions live in **`RANKING_SYSTEM_PROMPT`** in `ranking_agent.py`.

---

## Practical notes

- **Rate limits:** Use **`--gap`** (and keep **`--workers 1`**) when you see `[filter-RETRY]` / `[ranking-RETRY]` or HTTP 429s. Ranking parallelism multiplies concurrent LLM traffic.
- **Variance:** LLM outputs can vary run-to-run even at `temperature=0`.
- **Data source:** CSVs follow the TCP-CI feature schema described in Yaraghi et al.; column presence can vary slightly by subject—tools and extractor guard missing columns where needed.

---

## References

- A. S. Yaraghi *et al.*, “Scalable and Accurate Test Case Prioritization in Continuous Integration Contexts,” *IEEE TSE*, 2022. (arXiv:2109.13168.)
- J. Mendoza *et al.*, “On the Effectiveness of Data Balancing Techniques in the Context of ML-Based Test Case Prioritization,” PROMISE ’22. (ACM: 3558489.3559073.)
