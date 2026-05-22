# TCP Test Prioritization Agent

This repository evaluates an LLM-based Test Case Prioritization (TCP) pipeline
on TCP-CI/Yaraghi-style datasets.

The research idea is:

> Keep the TCP-CI feature and evaluation setting, but replace the paper's
> Random Forest/RankLib ranker with a multi-agent LLM ranker.

The practical problem is speed. A full paper-style target build can contain
hundreds or thousands of tests. Sending every test, with roughly 150 features
per test, through multiple LLM agents is expensive. The repo therefore supports
two modes:

- **Uncapped:** send every target-build test to the LLM. This is more faithful
  to a full-suite paper-style ranking setup, but can be very slow.
- **Candidate capped:** deterministically select the top-K risky tests for the
  LLM, then append every remaining test as a deterministic tail. This is much
  faster, but must be reported as a capped LLM experiment.

## What The LLM Sees

For each target build, the runner creates a temporary CSV containing target-build
feature rows with outcome columns removed.

The LLM input includes:

- `Test` IDs
- legal TCP-CI feature columns, normally 150 features
- historical `DET_COV_*_Faults` features from the paper's feature model

The LLM input does **not** include:

- target-build `Verdict`
- target-build `Duration`

`Verdict` and `Duration` are used only after ranking, when computing APFD,
APFDc, and recall metrics.

Implementation detail: the temporary sanitized CSV keeps the full legal feature
columns. The Filter Agent's compact prompt may omit feature values equal to `-1`
because TCP-CI uses `-1` as a "no data" sentinel. Real zero values are kept.

## Relationship To The Paper

| Aspect | TCP-CI/Yaraghi paper | This repo |
| --- | --- | --- |
| Feature setting | 150 TCP-CI features | 150 TCP-CI features when present in the CSV |
| Target verdict exposed to ranker | No | No |
| Target duration exposed to ranker | No | No |
| Evaluation target builds | Latest failed builds | Configurable; default is failed builds only |
| Main metric | APFDc | APFDc, plus APFD and Recall@10 |
| Ranking method | Random Forest/RankLib | Multi-agent LLM pipeline |
| Candidate cap | No | Optional; `--candidate-cap 0` disables it |

So this is not an exact reproduction of the paper's model. It is an LLM
approach using the paper's feature/evaluation framing.

## Pipeline

The main CLI is `scripts/run_llm_agent.py`.

1. **Runner** (`scripts/run_llm_agent.py`)
   - Selects target builds.
   - Removes `Verdict` and `Duration` from target-build rows before the LLM.
   - Optionally applies deterministic candidate capping.
   - Calls the LLM agent pipeline.
   - Scores the final ranking using the original target build.

2. **Filter Agent** (`src/tcp_agent/agent/filter_agent.py`)
   - Reads sanitized target-build feature rows.
   - Classifies each test into risk tiers `T1` through `T6`.
   - `T1` to `T5` are high-risk candidates for deeper ranking.
   - `T6` is low-signal and is appended later.

3. **Ranking Agent** (`src/tcp_agent/agent/ranking_agent.py`)
   - Ranks the `T1` to `T5` tests.
   - Works in batches so large builds do not require one enormous prompt.
   - Can use multiple workers for ranking batches.

4. **Merge Agent** (`src/tcp_agent/agent/ranking_agent.py`)
   - Enabled by default.
   - Globally reorders locally ranked high-risk batches.
   - Disable with `--no-merge-agent`.

5. **Validator** (`src/tcp_agent/agent/validator.py`)
   - Checks for invalid, missing, duplicated, or hallucinated test IDs.
   - The pipeline repairs or rejects unsafe LLM output instead of silently
     trusting it.

6. **Evaluator** (`src/tcp_agent/agent/ranker.py`,
   `src/tcp_agent/evaluation.py`)
   - Merges the LLM ranking back with the true target build.
   - Computes APFD, APFDc, and Recall@10.

## Setup

Create and install the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use the virtualenv Python when running scripts:

```bash
.venv/bin/python scripts/run_llm_agent.py --help
```

If you use local Ollama, make sure Ollama is running and the model exists:

```bash
ollama list
ollama pull qwen2.5:32b
```

If `ollama serve` says `address already in use`, that usually means Ollama is
already running.

For local Ollama/Qwen, pass:

```text
--openai-base-url http://127.0.0.1:11434/v1
```

For hosted models, create a `.env` file in the repo root with the key you need:

| Provider/model type | Environment variable |
| --- | --- |
| OpenAI-compatible | `OPENAI_API_KEY` |
| Gemini | `GOOGLE_API_KEY` |
| Anthropic Claude | `ANTHROPIC_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Ollama/vLLM/local OpenAI-compatible | `OPENAI_BASE_URL`; API key is optional |

## Recommended Professor Demo

Use this when you want a small, readable evidence run showing what the LLM
pipeline does.

```bash
.venv/bin/python scripts/run_llm_agent.py \
  --data datasets/apache@rocketmq.csv \
  --results-csv results/professor_uncapped_summary.csv \
  --build-results-csv results/professor_uncapped_builds.csv \
  --filter-model qwen2.5:32b \
  --ranking-model qwen2.5:32b \
  --openai-base-url http://127.0.0.1:11434/v1 \
  --eval-window 1 \
  --history-failed-builds-only \
  --candidate-cap 0 \
  --batch-size 10 \
  --ranking-batch-size 12 \
  --ranking-workers 2 \
  --quiet
```

This produces:

```text
results/professor_uncapped_summary.csv
results/professor_uncapped_builds.csv
```

The most useful high-level file is:

```text
results/professor_uncapped_builds.csv
```

It shows, per target build:

| Column | Meaning |
| --- | --- |
| `dataset` | Dataset CSV name |
| `target_build` | Build being ranked |
| `num_tests` | Total tests in the target build |
| `num_failures` | Failing tests in the target build, used after ranking |
| `candidate_cap` | Cap size; blank means uncapped |
| `candidate_selected_count` | Tests sent to the LLM |
| `candidate_tail_count` | Tests appended after the LLM as deterministic tail |
| `candidate_cap_recall` | Fraction of true failing tests included by the cap |
| `wall_seconds` | Runtime for that target build |
| `apfd`, `apfdc`, `recall_at_10` | Evaluation metrics |

## Candidate Cap Demonstration

To show why deterministic candidate capping is useful, run the same one-build
setup twice: once uncapped and once capped.

Uncapped:

```bash
.venv/bin/python scripts/run_llm_agent.py \
  --data datasets/apache@rocketmq.csv \
  --results-csv results/professor_uncapped_summary.csv \
  --build-results-csv results/professor_uncapped_builds.csv \
  --filter-model qwen2.5:32b \
  --ranking-model qwen2.5:32b \
  --openai-base-url http://127.0.0.1:11434/v1 \
  --eval-window 1 \
  --history-failed-builds-only \
  --candidate-cap 0 \
  --batch-size 10 \
  --ranking-batch-size 12 \
  --ranking-workers 2 \
  --quiet
```

Capped at 150:

```bash
.venv/bin/python scripts/run_llm_agent.py \
  --data datasets/apache@rocketmq.csv \
  --results-csv results/professor_cap150_summary.csv \
  --build-results-csv results/professor_cap150_builds.csv \
  --filter-model qwen2.5:32b \
  --ranking-model qwen2.5:32b \
  --openai-base-url http://127.0.0.1:11434/v1 \
  --eval-window 1 \
  --history-failed-builds-only \
  --candidate-cap 150 \
  --batch-size 10 \
  --ranking-batch-size 12 \
  --ranking-workers 2 \
  --quiet
```

Then compare:

```text
results/professor_uncapped_builds.csv
results/professor_cap150_builds.csv
```

The high-level argument is:

```text
Uncapped:
tests sent to LLM = num_tests

Capped:
tests sent to LLM = candidate_selected_count
remaining tests = candidate_tail_count
```

For a large build, the difference is dramatic:

```text
4000 target tests * 150 features = 600,000 feature values
150 capped tests * 150 features = 22,500 feature values
```

With `--batch-size 10`, 4000 tests means roughly 400 Filter Agent batches before
ranking even starts. A cap of 150 means roughly 15 Filter Agent batches.

## Optional Per-Test Trace

If you need proof of exactly what went into and came out of the LLM, add:

```text
--llm-trace-csv results/professor_one_build_trace.csv
```

The trace CSV has one row per test sent to the LLM. Important columns:

| Column | Meaning |
| --- | --- |
| `input_has_verdict` | Should be `0` |
| `input_has_duration` | Should be `0` |
| `input_feature_count` | Usually `150` |
| `llm_input_features_json` | The feature values shown to the LLM |
| `llm_priority` | Priority returned by the LLM pipeline |
| `llm_confidence` | LLM confidence |
| `llm_reason` | LLM explanation |
| `actual_verdict` | True target outcome, used after ranking |
| `actual_duration` | True target duration, used after ranking for APFDc |
| `build_wall_seconds` | Runtime for the build |

This file is useful as backup evidence. For a professor meeting, the
per-build CSV is usually easier to read first.

## Larger Dataset Note

`apache@rocketmq.csv` is a small demo dataset. In the current local CSV:

- latest failed build has about 204 tests
- maximum build size is about 215 tests

For the "thousands of tests" argument, use a larger dataset such as
`JMRI@JMRI.csv`. That dataset is much larger and is better for showing why
uncapped LLM ranking becomes impractical.

## Main CLI Reference

| Argument | Default | Meaning |
| --- | --- | --- |
| `--data` | required unless `--data-dir` | Evaluate one dataset CSV |
| `--data-dir` | required unless `--data` | Evaluate every CSV in a directory |
| `--mode` | `pilot` | `pilot` reads CSV features; `production` is placeholder work |
| `--eval-window` | `5` | Number of most recent eligible target builds |
| `--failed-builds-only` | on | Evaluate only builds with at least one failure |
| `--all-builds` | off | Include zero-failure target builds |
| `--history-failed-builds-only` | off | Restrict prior context checks to failed historical builds |
| `--candidate-cap` | `0` | Top-K deterministic preselection; `0` disables |
| `--batch-size` | `40` | Filter Agent tests per batch |
| `--ranking-batch-size` | `8` | High-risk tests per Ranking Agent batch |
| `--ranking-workers` | `1` | Concurrent ranking batches |
| `--no-merge-agent` | off | Disable the Merge Agent |
| `--workers` | `1` | Parallel datasets for `--data-dir` |
| `--results-csv` | `results/evaluation_summary.csv` | Dataset-level result CSV |
| `--build-results-csv` | derived from `--results-csv` | Per-build result CSV |
| `--llm-trace-csv` | unset | Optional per-test LLM input/output trace |
| `--diagnose-failures` | off | Print where failing tests were ranked |
| `--no-validation` | off | Bypass validation; not recommended |
| `--openai-base-url` | unset | OpenAI-compatible API base URL |

## Output Files

Each run can produce up to three CSVs:

1. **Dataset summary CSV**
   - Controlled by `--results-csv`.
   - One row per dataset/run configuration.
   - Good for final aggregate results.

2. **Per-build CSV**
   - Controlled by `--build-results-csv`.
   - One row per evaluated build.
   - Best file for explaining runtime, number of tests, and candidate cap load.

3. **Per-test trace CSV**
   - Controlled by `--llm-trace-csv`.
   - One row per test sent to the LLM.
   - Best file for proving the LLM did not see `Verdict` or `Duration`.

## Metrics

- **APFD:** rewards ranking failing tests earlier.
- **APFDc:** APFD variant weighted by test execution duration.
- **Recall@10:** fraction of failing tests that appear in the top 10.

The evaluator keeps missing tests in the scored ranking. If the LLM omits a
valid test ID, validation/repair prevents that test from disappearing from the
evaluation.

## Random Forest Baseline

There is also an older Random Forest baseline:

```bash
.venv/bin/python scripts/run_agent.py --data datasets/Angel-ML@angel.csv
```

This is a quick local comparison path, not the same rolling LLM protocol. The
paper uses RankLib-style ranking models; this baseline should not be treated as
the main reproduction path.

## Repository Map

```text
scripts/run_llm_agent.py              Rolling LLM evaluation CLI
scripts/run_agent.py                  Random Forest baseline CLI
src/tcp_agent/agent/tcp_agent.py      Filter -> Rank -> Validate orchestration
src/tcp_agent/agent/filter_agent.py   LLM tier classification
src/tcp_agent/agent/ranking_agent.py  LLM ranking and merge logic
src/tcp_agent/agent/validator.py      Ranking validation and safety checks
src/tcp_agent/agent/ranker.py         Merge ranked IDs with target labels
src/tcp_agent/tools/feature_extractor.py
                                      Feature selection and candidate score
src/tcp_agent/tools/full_context_tool.py
                                      Context shown to Ranking Agent
src/tcp_agent/evaluation.py           APFD, APFDc, Precision@K, Recall@K
src/tcp_agent/data_cache.py           Thread-safe CSV cache
reports/                              Professor-facing summaries
results/                              Experiment outputs
tests/                                Regression tests
```

## Troubleshooting

Use `.venv/bin/python`, not plain `python3`, if your system Python is missing
dependencies such as `python-dotenv`.

If Ollama says:

```text
bind: address already in use
```

then the server is probably already running.

If provider calls fail with rate limits:

- lower `--ranking-workers`
- lower `--workers`
- add `--filter-gap`
- add `--gap`
- reduce `--batch-size`
- reduce `--ranking-batch-size`

If a no-cap run takes too long, that is expected on large builds. Use the
per-build CSV to show the number of tests sent to the LLM and compare with a
capped run.

## References

- A. S. Yaraghi et al., "Scalable and Accurate Test Case Prioritization in
  Continuous Integration Contexts," IEEE TSE, 2022.
- J. Mendoza et al., "On the Effectiveness of Data Balancing Techniques in the
  Context of ML-Based Test Case Prioritization," PROMISE 2022.
