# Codebase Stale-Code Audit

Date: 2026-05-22

This audit classifies files and functions by whether they are used by the
current LLM TCP pipeline.

Legend:

- **Active:** used by the current `scripts/run_llm_agent.py` path.
- **Legacy:** works only for the older Random Forest baseline or old design.
- **Stale:** not referenced by the current pipeline, tests, or docs except as
  dead helper code.
- **Keep:** not part of the main path, but useful enough to keep for now.

## Current Main Path

The current experiment entrypoint is:

```text
scripts/run_llm_agent.py
  -> tcp_agent.agent.tcp_agent.run_agent
  -> run_multi_agent
  -> run_filter_agent
  -> run_ranking_agent
  -> validate_ranking
  -> build_ranked_df
  -> APFD/APFDc/Recall@10
```

The LLM receives sanitized target-build feature rows. `Verdict` and `Duration`
are removed before the LLM call and are used only afterward for evaluation.

## File-Level Summary

| File | Classification | Notes |
| --- | --- | --- |
| `scripts/run_llm_agent.py` | Active | Main rolling LLM evaluator. Large, but active. |
| `src/tcp_agent/agent/tcp_agent.py` | Active | Orchestrates Filter -> Rank -> Validate. |
| `src/tcp_agent/agent/filter_agent.py` | Active | Filter Agent. |
| `src/tcp_agent/agent/ranking_agent.py` | Active | Ranking Agent, Merge Agent, repair logic. |
| `src/tcp_agent/agent/validator.py` | Active | Output validation. |
| `src/tcp_agent/agent/ranker.py` | Active | Merges ranked IDs with true target labels for scoring. |
| `src/tcp_agent/tools/feature_extractor.py` | Active with stale helpers | Current feature extraction and candidate risk score live here. |
| `src/tcp_agent/tools/full_context_tool.py` | Active | Current single context tool for Ranking Agent. |
| `src/tcp_agent/data_cache.py` | Active | Shared CSV cache. |
| `src/tcp_agent/evaluation.py` | Active with legacy helper | APFD/APFDc/Recall@10 active; Precision@10 is legacy baseline-only. |
| `src/tcp_agent/utils/llm_utils.py` | Active | Provider routing, local OpenAI-compatible config, retry wrapper. |
| `src/tcp_agent/utils/token_logger.py` | Active with stale helper | Token logger active; metrics getter not currently used. |
| `src/tcp_agent/config.py` | Partially stale | `AgentMode` still used; production config is placeholder. |
| `scripts/run_agent.py` | Legacy | Old Random Forest baseline CLI, not current LLM experiment. |
| `src/tcp_agent/data_loader.py` | Legacy | Used only by old RF baseline. |
| `src/tcp_agent/features.py` | Legacy | Used only by old RF baseline. |
| `src/tcp_agent/model.py` | Legacy | Used only by old RF baseline. |
| `src/tcp_agent/ranking.py` | Legacy | Used only by old RF baseline. |
| `src/tcp_agent/tools/history_tool.py` | Stale | Old separate LangChain tool; replaced by `full_context_tool.py`. |
| `src/tcp_agent/tools/complexity_tool.py` | Stale | Old separate LangChain tool; not imported by current Ranking Agent. |
| `src/tcp_agent/tools/covered_code_risk_tool.py` | Stale | Old separate LangChain tool; not imported by current Ranking Agent. |
| `scratch/analyze_usage.py` | Stale utility | Manual token-cost script with hardcoded assumptions. |
| `scratch/test_token_usage.py` | Stale utility | Manual API experiment. |
| `tests/*.py` | Keep | Tests describe important behavior, but pytest is not installed in current venv. |

## Function-Level Audit

### `scripts/run_llm_agent.py`

All functions are active in the main CLI path unless noted.

| Function | Status | Notes |
| --- | --- | --- |
| `_print_diagnosis` | Active | Used by `--diagnose-failures`. |
| `_extract_merge_status` | Active | Used for result metadata and candidate tail rows. |
| `_extract_merge_missing_count` | Active | Used for result metadata and candidate tail rows. |
| `_build_agent_feature_frame` | Active | Removes `Verdict` and `Duration` before the LLM. |
| `_feature_profile_score_map` | Active | Builds deterministic candidate-cap scores. |
| `_apply_candidate_cap` | Active | Implements `--candidate-cap`. |
| `_filter_history_to_failed_builds` | Active | Used by `--history-failed-builds-only`. |
| `_append_candidate_tail` | Active | Keeps capped-out tests in evaluation. |
| `evaluate` | Active | Per-dataset rolling build evaluator. |
| `main` | Active | CLI entrypoint. |
| `_normalize_failed_builds_flag` | Active | Resume-key parsing. |
| `_normalize_bool_key` | Active | Resume-key normalization. |
| `_completed_key_from_values` | Active | Resume-key construction. |
| `_load_completed` | Active | Reads completed result rows. |
| `_append_csv_row` | Active | Atomic one-row CSV append. |
| `_append_csv_rows` | Active | Atomic multi-row trace append. |
| `_json_safe` | Active | Trace JSON serialization helper. |
| `_feature_json` | Active | Trace feature JSON helper. |
| `_append_llm_trace` | Active | Optional `--llm-trace-csv`. |
| `_append_result` | Active | Dataset-level result output. |
| `_append_build_result` | Active | Per-build result output. |
| `_evaluate_one` | Active | Worker path for `--data-dir`. |
| `_run_data_dir` | Active | Multi-dataset runner. |

Cleanup already applied:

- removed stale `precision_at_k` import from `run_llm_agent.py`.

### `src/tcp_agent/agent/filter_agent.py`

| Function/class | Status | Notes |
| --- | --- | --- |
| `TestClassification` | Active | Structured output schema. |
| `BatchClassificationResult` | Active | Structured output schema. |
| `FilterResult` | Active | Carries high-risk and low-signal tests. |
| `FilterResult.__init__` | Active | Constructs result container. |
| `FilterResult.summary` | Active | Logged by orchestrator. |
| `_build_structured_model` | Active | Creates model with structured output. |
| `_build_batch_prompt` | Active | Builds Filter Agent prompt. |
| `_chunk` | Active | Filter batching. |
| `_is_length_error` | Active | Batch-splitting fallback. |
| `_classify_batch` | Active | Performs one Filter Agent LLM call. |
| `_classify_batch_complete` | Active | Repairs omitted filter IDs by retrying. |
| `run_filter_agent` | Active | Public Filter Agent entrypoint. |

No fully stale functions found here.

### `src/tcp_agent/agent/ranking_agent.py`

| Function/class | Status | Notes |
| --- | --- | --- |
| `RankedTest` | Active | Structured output schema. |
| `PrioritizedTests` | Active | Structured output schema. |
| `AgentState` | Active | LangGraph state type. |
| `_build_models` | Active | Creates tool-bound and structured models. |
| `_build_structured_model` | Active | Used by merge/repair. |
| `_chunk_list` | Active | Ranking batching. |
| `_coerce_ranked_tests` | Active | Normalizes structured/direct JSON output. |
| `_extract_json_array` | Active | Parses direct JSON from local models. |
| `_batch_validation_errors` | Active | Validates one ranking batch. |
| `_structured_extract` | Active | Structured fallback extraction. |
| `_repair_ranked_tests` | Active | LLM repair fallback. |
| `_deterministic_repair_ranked_tests` | Active | Deterministic last resort. |
| `_extract_ranked_tests` | Active | Main extraction/repair wrapper. |
| `_merge_validation_errors` | Active | Merge output validation. |
| `_repair_merged_ranking` | Active | Merge repair fallback. |
| `_merge_ranked_batches` | Active | Merge Agent. |
| `_rank_batch` | Active | Per-batch Ranking Agent graph. |
| `run_ranking_agent` | Active | Public Ranking Agent entrypoint. |
| `_build_t6_tail` | Active | Appends T6 tests after high-risk ranked tests. |

No fully stale functions found here.

### `src/tcp_agent/agent/tcp_agent.py`

| Function | Status | Notes |
| --- | --- | --- |
| `run_multi_agent` | Active | Main orchestrator. |
| `run_agent` | Active | Wrapper used by `scripts/run_llm_agent.py`. |

No fully stale functions found here.

### `src/tcp_agent/agent/validator.py`

| Function/class | Status | Notes |
| --- | --- | --- |
| `ValidationResult` | Active | Validation result object. |
| `ValidationResult.__str__` | Active | Logged/raised by orchestrator. |
| `validate_ranking` | Active | Main validation function. |
| `log_validation_errors` | Active | Used on validation failure. |

No fully stale functions found here.

### `src/tcp_agent/agent/ranker.py`

| Function | Status | Notes |
| --- | --- | --- |
| `normalize_ranked_items` | Active | Used by scoring and trace. |
| `build_ranked_df` | Active | Used by `evaluate`. |

No fully stale functions found here.

### `src/tcp_agent/tools/feature_extractor.py`

| Function | Status | Notes |
| --- | --- | --- |
| `_legal_feature_cols` | Active | Used by `extract_risk_profiles`. |
| `extract_risk_profiles` | Active | Used by Filter Agent and full context tool. |
| `extract_failure_rates` | Stale | No longer called after removing prompt-only failure-rate aliases. |
| `extract_exec_times` | Active | Used by Filter Agent for T6 tail timing. |
| `extract_all_test_ids` | Active | Used by validator path. |
| `candidate_risk_score` | Active | Used by candidate cap in `run_llm_agent.py`. |
| `select_candidate_test_ids` | Stale | Old helper; current runner scores target rows directly. |

Cleanup already applied:

- removed stale `pandas as pd` import.

### `src/tcp_agent/tools/full_context_tool.py`

| Function | Status | Notes |
| --- | --- | --- |
| `get_full_test_context` | Active | The only Ranking Agent context tool currently used. |

No fully stale functions found here.

### `src/tcp_agent/tools/history_tool.py`

| Function | Status | Notes |
| --- | --- | --- |
| `get_test_risk_profile` | Stale | Old separate tool. Current ranking uses `get_full_test_context`. |

This file can likely be deleted if the old multi-tool Ranking Agent design is
not being kept.

### `src/tcp_agent/tools/complexity_tool.py`

| Function | Status | Notes |
| --- | --- | --- |
| `_pilot_get_test_complexity` | Stale | Only used by stale `get_test_complexity`. |
| `_production_get_test_complexity` | Stale placeholder | Raises `NotImplementedError`. |
| `get_test_complexity` | Stale | Old separate tool. Current ranking uses `get_full_test_context`. |

Cleanup already applied:

- removed stale `get_config` import.

This file can likely be deleted if the old multi-tool Ranking Agent design is
not being kept.

### `src/tcp_agent/tools/covered_code_risk_tool.py`

| Function | Status | Notes |
| --- | --- | --- |
| `_pilot_get_covered_code_risk` | Stale | Only used by stale `get_covered_code_risk`. |
| `_production_get_covered_code_risk` | Stale placeholder | Raises `NotImplementedError`. |
| `get_covered_code_risk` | Stale | Old separate tool. Current ranking uses `get_full_test_context`. |

Cleanup already applied:

- removed stale `get_config` import.

This file can likely be deleted if the old multi-tool Ranking Agent design is
not being kept.

### `src/tcp_agent/config.py`

| Function/class | Status | Notes |
| --- | --- | --- |
| `AgentMode` | Active | Used by CLI and orchestrator. |
| `set_mode` | Partially stale | Called by orchestrator, but only old tools read the mode. |
| `get_mode` | Stale if old tools are deleted | Currently used only by stale old tools. |
| `get_config` | Stale | No active caller. |

The `PRODUCTION` mode is currently a placeholder. The main LLM path is
effectively CSV/pilot mode.

### `src/tcp_agent/data_cache.py`

| Function | Status | Notes |
| --- | --- | --- |
| `load_dataset` | Active | Central CSV loader. |
| `clear_cache` | Keep | Used by tests and useful for long-running processes. |

No fully stale functions found here.

### `src/tcp_agent/evaluation.py`

| Function | Status | Notes |
| --- | --- | --- |
| `precision_at_k` | Legacy | Used by old RF baseline only, not LLM path. |
| `failure_recall_at_k` | Active | Main LLM metric. |
| `apfd` | Active | Main metric. |
| `apfdc` | Active | Main metric. |

### Old Random Forest Baseline

These are coherent as a group, but they are not part of the current LLM
experiment path.

| File/function | Status | Notes |
| --- | --- | --- |
| `scripts/run_agent.py::main` | Legacy | Old RF CLI. |
| `data_loader.load_data` | Legacy | Used only by RF CLI. |
| `data_loader.get_features_and_labels` | Legacy | Used only by RF CLI. |
| `data_loader.get_metadata` | Legacy | Used only by RF CLI. |
| `features.clean_features` | Legacy | Used only by RF CLI. |
| `features.apply_smote` | Legacy | Used only by RF model. |
| `model.train_model` | Legacy | Used only by RF CLI. |
| `ranking.rank_tests` | Legacy | Used only by RF CLI. |

Decision point:

- If you want a clean LLM-only repo, move/delete this whole baseline group.
- If you want a simple comparison baseline, keep it but label it clearly as
  `legacy_rf_baseline`.

### `src/tcp_agent/utils/llm_utils.py`

| Function | Status | Notes |
| --- | --- | --- |
| `resolve_provider` | Active | Used by Filter/Ranking model builders. |
| `openai_compat_base_from_env` | Active | Used by CLI key check and kwargs builder. |
| `uses_openai_sdk_stack` | Active | Used by kwargs builder and tests. |
| `build_init_chat_model_kwargs` | Active | Used by Filter/Ranking model builders. |
| `invoke_with_retry` | Active | Used by Filter/Ranking LLM calls. |

No fully stale functions found here.

### `src/tcp_agent/utils/token_logger.py`

| Function/class | Status | Notes |
| --- | --- | --- |
| `TokenUsageLogger` | Active | Used by `invoke_with_retry`. |
| `TokenUsageLogger.__new__` | Active | Singleton behavior. |
| `TokenUsageLogger.__init__` | Active | Creates log file. |
| `TokenUsageLogger.log_request` | Active | Called for LLM requests/errors. |
| `TokenUsageLogger.get_current_metrics` | Stale/Keep | Not used, but useful for future live dashboards. |

Cleanup already applied:

- removed stale `logging` and `os` imports.

### Scratch Scripts

| File/function | Status | Notes |
| --- | --- | --- |
| `scratch/analyze_usage.py::analyze_logs` | Stale utility | Hardcoded `avg_* / 5` and old o3-mini pricing. |
| `scratch/test_token_usage.py::test_usage` | Stale utility | Manual API experiment; not a pytest test. |

These can be deleted or moved to `docs/dev-notes/` if you want less clutter.

## Safest Cleanup Order

1. Remove unused imports. Done in this audit pass:
   - `precision_at_k` from `scripts/run_llm_agent.py`
   - `pandas as pd` from `src/tcp_agent/tools/feature_extractor.py`
   - `pandas as pd` from `src/tcp_agent/ranking.py`
   - `get_config` from stale tool files
   - `logging` and `os` from `src/tcp_agent/utils/token_logger.py`

2. Delete or archive scratch scripts:
   - `scratch/analyze_usage.py`
   - `scratch/test_token_usage.py`

3. Decide what to do with the RF baseline:
   - Keep and rename/docs-label it as legacy, or
   - Move/delete `scripts/run_agent.py`, `data_loader.py`, `features.py`,
     `model.py`, `ranking.py`, and legacy `precision_at_k`.

4. Delete old separate context tools if no longer needed:
   - `history_tool.py`
   - `complexity_tool.py`
   - `covered_code_risk_tool.py`

5. Simplify `config.py` after old tools are gone:
   - keep `AgentMode` if the CLI still accepts `--mode`
   - remove `get_mode`, `get_config`, and unused production config if the
     production path is not being implemented soon

6. Split `scripts/run_llm_agent.py` later:
   - It is active, not stale, but it is over 1,100 lines.
   - Good extraction targets: candidate cap helpers, CSV append/resume helpers,
     trace helpers, and CLI parser.

## Recommended First PR

Do not delete behavior in the first cleanup. Start with a low-risk hygiene PR:

- remove unused imports
- add/update this audit report
- update README wording
- leave legacy files in place

Then do a second PR for either:

- `legacy_rf_baseline/` move, or
- full deletion of the old RF baseline and old context tools.
