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

The agent runs a **LangGraph loop** — Claude picks which tools to call, looks at the results, and keeps going until it has enough info to rank all the tests.

```
"prioritize tests for the next build"
     │
     ▼
┌────────────────────────────┐
│    LangGraph Agent Loop    │
│                            │
│  Claude thinks → picks a   │
│  tool → sees result →      │
│  thinks again → repeats    │
│                            │
│  available tools:          │
│   • get_all_failure_rates    │
│   • get_test_history       │
│   • get_failed_builds      │
│   • get_build_failure_summary │
│   • get_high_coverage_tests│
└────────────────────────────┘
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

This is what actually happens behind the scenes when the agent runs. Claude decides each step on its own.

```
Step 1: Claude calls get_all_failure_rates("dataset.csv")
        → gets all 46 tests with failure rates
        → sees test 5141 and 5140 have failure_rate: 1.0 (meaning they fail every run)

Step 2: Claude calls get_failed_builds("dataset.csv", n=5)
        → finds 3 recent builds that had test failures

Step 3: Claude calls get_build_failure_summary on each failed build
        → discovers the same 7 tests (509, 510, 511, 512, 513, 2161, 4691)
           show up in every single failed build — a cluster

Step 4: Claude calls get_high_coverage_tests("dataset.csv", n=46)
        → sees tests 5659 (62.3), 5656 (60.0), 5660 (58.2) have the
           highest coverage but zero failures — good safety nets

Step 5: Claude combines everything:
        → 100% failure rate tests go first (5141, 5140)
        → the 7-test failure cluster goes next
        → high failure rate + high coverage tests after that
        → zero-failure tests ranked by coverage score at the bottom
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
│   ├── history_tool.py     # failure rates and test history
│   ├── log_tool.py         # recent failed builds and which tests broke
│   ├── dependency_tool.py  # coverage scores per test
│   └── git_tool.py         # placeholder for live GitHub integration later
├── data_loader.py          # loads dataset (Phase 1)
├── features.py             # SMOTE and feature cleaning (Phase 1)
├── model.py                # Random Forest training (Phase 1)
├── ranking.py              # failure probability ranking (Phase 1)
└── evaluation.py           # APFD, APFDc, Precision@k

scripts/
├── run_llm_agent.py        # runs the LLM agent
└── run_agent.py            # runs the ML baseline
```

---

## How Everything Connects

```
┌──────────────────────┐
│   run_llm_agent.py   │  entry point — kicks everything off
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│    tcp_agent.py      │  builds the LangGraph agent loop
│                      │
│  ┌────────────────┐  │
│  │  Claude thinks │──┼──► picks a tool from:
│  └───▲────────────┘  │     • get_all_failure_rates (failure rates)
│      │               │     • get_test_history (single test lookup)
│      │  loop until   │     • get_failed_builds (recent broken builds)
│      │  done         │     • get_build_failure_summary (which tests broke)
│      │               │     • get_high_coverage_tests (coverage scores)
│  ┌───┴────────────┐  │
│  │  tool returns  │◄─┼──── tool reads dataset.csv
│  │  result        │  │
│  └────────────────┘  │
│                      │
│  outputs ranked JSON │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│     ranker.py        │  merges ranking with real Verdict + Duration
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   evaluation.py      │  scores it — APFD, APFDc, Precision@k
└──────────────────────┘
```

---

## Notes

- Currently runs on a pre-extracted dataset. `git_tool.py` is ready for when we hook it up to a live GitHub repo.
- Phase 1 (ML baseline) is kept for comparison — not the main focus.
- GitHub Actions runs the ML baseline on every push.
- The earlier single-shot agent (Phase 2a) sent all data in one prompt. The LangGraph version (Phase 2b) lets Claude choose what to look at.
