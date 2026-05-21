# TCP Test Prioritization Agent

LLM-driven test case prioritization for CI datasets in the TCP-CI/Yaraghi-style CSV format. The main path is a two-agent LLM pipeline evaluated with rolling historical builds; a separate Random Forest baseline is included for quick local comparison.

## Pipeline

1. **Filter Agent** (`src/tcp_agent/agent/filter_agent.py`)
   - Reads the latest historical feature snapshot per test from CSV.
   - Sends batched structured-output prompts to classify tests as high-risk tiers **T1-T5** or low-signal **T6**.
   - Uses all legal CSV feature columns except identifiers, labels, outcomes, and leakage columns.

2. **Ranking Agent** (`src/tcp_agent/agent/ranking_agent.py`)
   - Ranks only T1-T5 tests with a LangGraph tool-calling loop.
   - Calls one combined context tool that exposes all legal TCP-CI features for the ranking batch.
   - T6 tests are appended after ranked high-risk tests, ordered by average execution time.

3. **Validator** (`src/tcp_agent/agent/validator.py`)
   - Checks schema, missing IDs, duplicate IDs, unknown IDs, and priority consistency.
   - Invalid LLM output raises unless `--no-validation` is passed.

The CLI enables a third **Merge Agent** by default. It globally reorders locally ranked high-risk batches before T6 tests are appended, which reduces batch-order artifacts.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the repo root with the provider keys you use:

| Model/provider | Required setting |
| --- | --- |
| OpenAI-compatible models | `OPENAI_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| Anthropic Claude | `ANTHROPIC_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Local/OpenAI-compatible Qwen | `OPENAI_BASE_URL`, usually ending in `/v1`; `OPENAI_API_KEY` is optional |

Qwen model names containing `qwen` route through the OpenAI-compatible stack. For Ollama or vLLM, set:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
```

or pass `--openai-base-url`.

## Main Evaluation Command

```bash
python3 scripts/run_llm_agent.py --data datasets/apache@rocketmq.csv
```

The evaluator walks over the most recent target builds. For each target build, the agent only sees rows with `Build < target`, then its predicted ordering is scored against the target build.

Useful examples:

```bash
# Single dataset, one target build for quick tuning
python3 scripts/run_llm_agent.py --data datasets/apache@rocketmq.csv --eval-window 1

# All datasets, resumable results CSV
python3 scripts/run_llm_agent.py --data-dir datasets --quiet

# Local Qwen through an OpenAI-compatible server
python3 scripts/run_llm_agent.py --data datasets/apache@rocketmq.csv \
  --openai-base-url http://127.0.0.1:11434/v1 \
  --filter-model qwen2.5:32b \
  --ranking-model qwen2.5:32b
```

## CLI Reference

| Argument | Default | Meaning |
| --- | --- | --- |
| `--data` | required unless `--data-dir` | Evaluate one CSV |
| `--data-dir` | required unless `--data` | Evaluate every `*.csv` in a directory |
| `--mode` | `pilot` | `pilot` reads CSV features; `production` hooks are placeholders |
| `--batch-size` | `40` | Filter Agent tests per LLM batch; auto-splits on length errors |
| `--filter-model` | `gpt-5-nano` | Model used for T1-T6 filtering |
| `--ranking-model` | `gemini-3-flash-preview` | Model used for high-risk ranking |
| `--eval-window` | `5` | Number of most recent eligible target builds |
| `--failed-builds-only` | on | Evaluate only target builds with at least one failure |
| `--all-builds` | off | Include zero-failure target builds |
| `--gap` | `0` | Sleep between full agent runs for target builds |
| `--filter-gap` | `0` | Sleep between Filter Agent batches |
| `--ranking-workers` | `1` | Concurrent Ranking Agent batches |
| `--ranking-batch-size` | `8` | High-risk tests per Ranking Agent batch |
| `--no-merge-agent` | off | Disable global Merge Agent |
| `--workers` | `1` | Concurrent datasets for `--data-dir` |
| `--results-csv` | `results/evaluation_summary.csv` | Resumable append-only results file |
| `--no-validation` | off | Return invalid LLM rankings instead of raising |
| `--diagnose-failures` | off | Print where failing tests landed per build |
| `--openai-base-url` | unset | OpenAI-compatible base URL for local/proxy models |

Results CSV columns include metrics, model names, batch settings, wall time, timestamp, status, and error. Completed rows are skipped only when the dataset and experiment settings match.

## Current Internals

- Filter batches default to **40** tests.
- Ranking batches default to **8** tests and can be changed with `--ranking-batch-size`.
- The Merge Agent is enabled by default in the CLI and can be disabled with `--no-merge-agent`.
- Ranking parallelism is controlled by `--ranking-workers`; the module constant `_RANKING_PARALLELISM` is not used by the CLI path.
- CSV parsing is cached by `src/tcp_agent/data_cache.py`, but each target build still writes a temporary history CSV.
- Provider routing lives in `src/tcp_agent/utils/llm_utils.py`.
- Token usage is logged to `logs/token_usage.log`.

## Metrics

- **APFD:** rewards finding failing tests earlier.
- **APFDc:** APFD variant weighted by `Duration`.
- **Recall@10:** fraction of all failing tests found in the top 10.

`build_ranked_df` uses an outer merge so tests missed by the LLM remain in the evaluation and receive worst priority rather than disappearing.

## Random Forest Baseline

```bash
python3 scripts/run_agent.py --data datasets/Angel-ML@angel.csv
```

This baseline trains a Random Forest with an 80/20 random split, applies SMOTE to the training fold, ranks the holdout by predicted failure probability, and prints APFD, APFDc, and Precision@10. It is not the same rolling evaluation protocol as the LLM path.

## Repository Map

```text
scripts/run_llm_agent.py          Rolling LLM evaluation CLI
scripts/run_agent.py              Random Forest baseline CLI
src/tcp_agent/agent/filter_agent.py
src/tcp_agent/agent/ranking_agent.py
src/tcp_agent/agent/validator.py
src/tcp_agent/agent/ranker.py
src/tcp_agent/tools/              LangChain tools and CSV feature extraction
src/tcp_agent/evaluation.py       APFD, APFDc, Precision@K, Recall@K
src/tcp_agent/data_cache.py       Thread-safe pandas CSV cache
tests/                            Regression tests for batching, validation, Qwen routing, results CSV
```

## Notes

- LLM output can vary even with `temperature=0`.
- `DET_COV_C_Faults` and `DET_COV_IMP_Faults` are excluded from the LLM feature set because they leak post-run fault information.
- Use `--gap`, `--filter-gap`, lower `--workers`, or lower `--ranking-workers` when providers return 429/rate-limit errors.

## References

- A. S. Yaraghi et al., “Scalable and Accurate Test Case Prioritization in Continuous Integration Contexts,” IEEE TSE, 2022.
- J. Mendoza et al., “On the Effectiveness of Data Balancing Techniques in the Context of ML-Based Test Case Prioritization,” PROMISE 2022.
