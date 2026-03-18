# TCP Test Prioritization Agent

Undergraduate research project — building an LLM-based agent that prioritizes regression tests in CI environments.

Instead of training a model, the agent reasons over test history, build failures, coverage, and execution times to decide which tests should run first. No retraining needed.

---

## How to Run

```bash
# set up
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run the LLM agent (needs an Anthropic API key)
export ANTHROPIC_API_KEY="your-key-here"
PYTHONPATH=src python3 scripts/run_llm_agent.py --data dataset.csv

# run the ML baseline (no API key needed)
PYTHONPATH=src python3 scripts/run_agent.py --data dataset.csv
```

You can get an Anthropic API key at console.anthropic.com → API Keys → Create Key.

---

## What the Agent Does

It reads the dataset and pulls together 5 different signals:
- How often each test has failed historically
- Which tests failed in the most recent builds
- Which specific tests failed in the latest build
- How much code each test covers
- How long each test takes to run

All of that gets sent to Claude, which reasons over it and outputs a ranked list with explanations.

Example output:
```
#1 Test 5141 — This test has failed in every single build historically, making it the highest risk
#2 Test 2953 — High failure rate of 65% with consistent failures across recent builds
#3 Test 2954 — 45% failure rate combined with high code coverage
...
```

---

## Results

|                | LLM Agent | ML Baseline (RF) | Random |
|----------------|-----------|-------------------|--------|
| APFD           | 0.8043    | 0.9361            | ~0.50  |
| APFDc          | 0.7682    | 0.9498            | ~0.50  |
| Precision@10   | 1.0000    | 1.0000            | ~0.12  |

The LLM agent gets solid results with zero training and full explainability.

---

## Project Structure

```
src/tcp_agent/
├── agent/
│   ├── tcp_agent.py        # the main agent — gathers context and asks Claude to rank tests
│   └── ranker.py           # merges Claude's output with real Verdict data
├── tools/
│   ├── history_tool.py     # pulls historical failure rates from the dataset
│   ├── log_tool.py         # pulls recent failed builds from the dataset
│   ├── dependency_tool.py  # ranks tests by coverage score
│   └── git_tool.py         # for future use — will fetch live commit diffs from GitHub
├── data_loader.py          # loads dataset.csv
├── features.py             # cleans features, applies SMOTE
├── model.py                # trains Random Forest model
├── ranking.py              # sorts tests by predicted failure probability
└── evaluation.py           # scores the ranking with APFDc, APFD, Precision@k

scripts/
├── run_llm_agent.py        # runs the LLM agent
└── run_agent.py            # runs the ML baseline
```

---

## How It All Connects

```
dataset.csv
     │
     ├── history_tool    → failure rates per test
     ├── log_tool        → recent failed builds + latest build details
     ├── dependency_tool → coverage scores per test
     └── execution times → how long each test takes
                │
                ▼
          tcp_agent.py sends everything to Claude
                │
                ▼
          Claude ranks the tests and explains why
                │
                ▼
          ranker.py attaches real Verdict data
                │
                ▼
          evaluation.py scores the ranking
```

---

## Notes

- The agent currently works on a pre-extracted dataset. `git_tool.py` is set up for when we connect it to a real GitHub repo later.
- The ML baseline (Phase 1) is there for comparison, not the main research focus.
- GitHub Actions runs the ML baseline automatically on every push.
- Research question: can an LLM agent match traditional ML approaches for test prioritization without needing to retrain, while being easier to maintain and fully explainable?
