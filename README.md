# TCP Test Prioritization Agent

Undergrad research project — an LLM agent that figures out which regression tests are most likely to fail so they run first. No model training, no feature engineering, just Claude reasoning over CI/CD data.

---

## How to Run

```bash
# setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run the LLM agent (needs an Anthropic API key)
export ANTHROPIC_API_KEY="your-key-here"
PYTHONPATH=src python3 scripts/run_llm_agent.py --data dataset.csv

# run the ML baseline (no API key needed)
PYTHONPATH=src python3 scripts/run_agent.py --data dataset.csv
```

Get an API key at console.anthropic.com → API Keys → Create Key.

---

## What It Does

The agent runs a **LangGraph loop** — Claude (Sonnet 4.6) picks which tools to call, looks at the results, and keeps going until it has enough info to rank all the tests. The agent has a **10-call safety limit** to prevent runaway loops.

Evaluation uses **per-build splitting**: the most recent build is held out as the target, and the agent only sees historical builds. This simulates a real CI scenario where you're predicting failures for the next build.

```
"prioritize tests for the next build"
     │
     ▼
┌────────────────────────────────┐
│      LangGraph Agent Loop      │
│                                │
│  Claude thinks → picks a       │
│  tool → sees result →          │
│  thinks again → repeats        │
│                                │
│  available tools (7):          │
│   • get_test_risk_profile      │
│   • get_all_failure_rates      │
│   • get_execution_times        │
│   • get_high_coverage_tests    │
│   • get_failed_builds          │
│   • get_build_failure_summary  │
│   • get_test_history           │
└────────────────────────────────┘
     │
     ▼
ranked list with reasons
```

Example output:
```
#1 Test 5141 — 100% failure rate across 31 runs, fails every single build
#2 Test 5140 — 100% failure rate across 31 runs, same deal
#3 Test 4691 — part of a 7-test cluster that fails together in recent builds
#4 Test 509  — same cluster, 21.1% historical failure rate
...
```

The cool part — Claude actually spotted that 7 tests always fail together as a group. That wasn't hardcoded anywhere, it figured it out from the data.

---

## How the Agent Reasons (real run)

This is what actually happens behind the scenes when the agent runs. Claude decides each step on its own — the system prompt suggests a tool usage strategy but the agent chooses what to call and in what order.

```
Step 1: Claude calls get_test_risk_profile(dataset_path)
        → gets REC_, DET_COV_, and COV_ features for every test
        → sees which tests have high recent failure rates and fault signals

Step 2: Claude calls get_all_failure_rates(dataset_path)
        → cross-references overall failure history
        → confirms tests 5141 and 5140 have failure_rate: 1.0

Step 3: Claude calls get_execution_times(dataset_path)
        → factors in test duration for cost-aware ordering
        → fast-failing tests should run before slow ones

Step 4: Claude calls get_failed_builds(dataset_path, n=5)
        → finds recent builds that had test failures

Step 5: Claude calls get_build_failure_summary on each failed build
        → discovers the same 7 tests (509, 510, 511, 512, 513, 2161, 4691)
           show up in every single failed build — a cluster

Step 6: Claude calls get_high_coverage_tests(dataset_path, n=46)
        → sees tests 5659 (62.3), 5656 (60.0), 5660 (58.2) have the
           highest coverage but zero failures — good safety nets

Step 7: Claude combines everything using a tiered strategy:
        → Tier 1: always-failing tests first (5141, 5140)
        → Tier 2: recently failing tests + the 7-test failure cluster
        → Tier 3: tests covering code with known faults
        → Tier 4: historical failures not recent
        → Tier 5: zero-failure safety nets ranked by coverage
        → Tier 6: remaining tests by execution cost
        → outputs ranked JSON with a reason for every test
```

No hardcoded rules. Claude looked at the data, spotted patterns, and made decisions. Different runs might call tools in a different order — that's the agentic part.

---

## Results

|                | LangGraph Agent | Single-Shot Agent | ML Baseline (RF) | Random |
|----------------|-----------------|-------------------|-------------------|--------|
| APFD           | 0.80            | 0.80              | 0.94              | ~0.50  |
| APFDc          | 0.77–0.82       | 0.77              | 0.95              | ~0.50  |
| Precision@10   | 1.00            | 1.00              | 1.00              | ~0.12  |

The ML baseline scores higher but needs 150+ engineered features and retraining. The LLM agent gets competitive results with full explainability and zero training.

APFDc varies slightly between runs since LLMs aren't fully deterministic even with temperature=0.

---

## Project Structure

```
src/tcp_agent/
├── agent/
│   ├── tcp_agent.py        # the main agent — LangGraph loop with tool calling
│   └── ranker.py           # merges Claude's ranking with real Verdict data
├── tools/
│   ├── history_tool.py     # failure rates, test history, execution times, risk profiles
│   ├── log_tool.py         # recent failed builds and which tests broke
│   ├── dependency_tool.py  # coverage scores and changed-file matching
│   └── git_tool.py         # placeholder for live GitHub integration later
├── data_loader.py          # loads dataset (ML baseline)
├── features.py             # SMOTE and feature cleaning (ML baseline)
├── model.py                # Random Forest training (ML baseline)
├── ranking.py              # failure probability ranking (ML baseline)
└── evaluation.py           # APFD, APFDc, Precision@k

scripts/
├── run_llm_agent.py        # runs the LLM agent (per-build evaluation)
└── run_agent.py            # runs the ML baseline (80/20 split evaluation)
```

---

## How Everything Connects

```
┌──────────────────────┐
│   run_llm_agent.py   │  splits dataset → history builds + target build
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│    tcp_agent.py      │  builds the LangGraph agent loop (max 10 calls)
│                      │
│  ┌────────────────┐  │
│  │  Claude thinks │──┼──► picks a tool from:
│  └───▲────────────┘  │     • get_test_risk_profile (REC/DET/COV features)
│      │               │     • get_all_failure_rates (failure history)
│      │  loop until   │     • get_execution_times (test durations)
│      │  done         │     • get_high_coverage_tests (coverage scores)
│      │               │     • get_failed_builds (recent broken builds)
│      │               │     • get_build_failure_summary (which tests broke)
│  ┌───┴────────────┐  │     • get_test_history (single test lookup)
│  │  tool returns  │◄─┼──── tool reads history CSV (target build excluded)
│  │  result        │  │
│  └────────────────┘  │
│                      │
│  outputs ranked JSON │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│     ranker.py        │  merges ranking with target build's Verdict + Duration
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   evaluation.py      │  scores it — APFD, APFDc, Precision@k
└──────────────────────┘
```

---

## Notes

- Currently runs on a pre-extracted dataset. `git_tool.py` is ready for when we hook it up to a live GitHub repo.
- The ML baseline (Phase 1) is kept for comparison — not the main focus.
- GitHub Actions runs the ML baseline on every push to main.
- The earlier single-shot agent (Phase 2a) sent all data in one prompt. The LangGraph version (Phase 2b) lets Claude choose what to look at.
- The agent uses a detailed tiered prioritization strategy in its system prompt — see `tcp_agent.py` for the full prompt.
