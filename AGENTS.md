# Codex Instructions

This repo implements an LLM-based Test Case Prioritization (TCP) pipeline for
TCP-CI datasets. Before making code changes, read the relevant files first and
preserve the experiment protocol.

## Key Files

- `scripts/run_llm_agent.py`: experiment runner, resume logic, candidate cap,
  build selection, CSV output.
- `src/tcp_agent/agent/tcp_agent.py`: main filter/rank/evaluate orchestration.
- `src/tcp_agent/agent/filter_agent.py`: LLM filter stage.
- `src/tcp_agent/agent/ranking_agent.py`: LLM ranking and merge stage.
- `src/tcp_agent/agent/validator.py`: ranking validation.
- `src/tcp_agent/tools/feature_extractor.py`: feature filtering and candidate
  risk scoring.
- `src/tcp_agent/tools/full_context_tool.py`: per-test context given to agents.

## Experiment Legality

- Never expose the target build's `Verdict` to the LLM.
- Never use target-build failure labels for candidate selection or ranking.
- Target `Duration` may be used for APFDc scoring after ranking, not for LLM
  prioritization decisions.
- The LLM should rank target-build feature rows with `Verdict` and `Duration`
  hidden, matching the paper's prediction setup.
- Candidate selection must use only history/features available before target
  test execution.
- If a target build uses a candidate cap, tests outside the cap must still be
  appended as a deterministic tail for evaluation.

## Current Experiment Setup

- Local OpenAI-compatible endpoint, usually Ollama:
  `http://127.0.0.1:11434/v1`
- Main local model: `qwen2.5:32b`
- Quick smoke setup:
  - `--eval-window 1`
  - `--candidate-cap 150`
  - `--batch-size 10`
  - `--ranking-batch-size 12`
  - `--ranking-workers 2`
- Main practical capped setup:
  - `--eval-window 51`
  - `--history-failed-builds-only`
  - `--candidate-cap 150`
  - Must be described as a capped LLM experiment, not exact paper replication.

## Paper Protocol Notes

- The TCP-CI papers use the latest failed builds for evaluation.
- The Yaraghi TCP-CI paper evaluates the latest 50 failed builds.
- The Mendoza data-balancing paper discusses the latest 51 failed builds, where
  rolling train/test uses prior failed builds.
- Their method is RF/RankLib, not LLM-based. This project should be described as
  using the TCP-CI feature setting and APFD/APFDc evaluation while replacing the
  ranking method with a multi-agent LLM pipeline.
- `DET_COV_*_Faults` are historical previously-detected-fault features in the
  paper and are included by the current feature extraction tooling. Describe
  the feature set as the 150-feature TCP-CI setting.

## Robustness Rules

- Treat LLM output as untrusted structured text.
- LLMs must never be allowed to invent usable test IDs.
- Drop or repair unknown IDs deterministically.
- Append missing valid IDs deterministically before validation.
- If a filter or ranking batch fails, prefer a deterministic fallback ranking
  over losing tests or failing a long experiment.
- Keep resume and output metadata explicit when adding new experiment flags.

## Important TODOs

- Add deterministic fallback classifications for omitted Filter Agent IDs.
- Add deterministic ranking fallback for failed Ranking Agent batches.
- Make Merge Agent repair tolerant of malformed priorities/confidence values.
- Add tests for hallucinated IDs, omitted IDs, duplicate IDs, and malformed
  ranking JSON.

## Common Run Command

```bash
python3 scripts/run_llm_agent.py \
  --data-dir datasets \
  --results-csv results/qwen32b_eval51_cap150.csv \
  --filter-model qwen2.5:32b \
  --ranking-model qwen2.5:32b \
  --openai-base-url http://127.0.0.1:11434/v1 \
  --eval-window 51 \
  --history-failed-builds-only \
  --batch-size 10 \
  --ranking-batch-size 12 \
  --ranking-workers 2 \
  --candidate-cap 150 \
  --quiet
```
