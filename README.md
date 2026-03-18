# TCP Test Prioritization Agent

An undergraduate research project building an LLM-based agent for Test Case Prioritization (TCP) in CI environments.

Based on:
> Yaraghi et al. (2022). *Scalable and Accurate TCP in Continuous Integration Contexts.* IEEE TSE.
> Mendoza et al. (2022). *On the Effectiveness of Data Balancing Techniques in the Context of ML-Based TCP.* PROMISE '22.

---

## The Problem

When a developer pushes code, CI runs all regression tests. For large projects this can take hours. If we can predict which tests are most likely to fail, we can run them first and catch bugs faster.

---

## How It Works

There are two phases:

**Phase 1 — ML Baseline**
Loads a pre-extracted feature dataset (~150 features per test), trains a Random Forest to predict failure probability, and ranks tests highest to lowest risk. Beats the paper's benchmark — APFDc 0.9498 vs 0.82.

**Phase 2 — LLM Agent**
Claude reasons over multiple signals — historical failure rates, recent build failures, and coverage scores — and outputs a ranked test list with explanations for each decision. No retraining needed.

---

## Project Structure

```
tcp-test-prioritization-agent/
├── src/
│   └── tcp_agent/
│       ├── agent/
│       │   ├── tcp_agent.py      # Claude agent — gathers context and ranks tests
│       │   └── ranker.py         # merges Claude's output with real Verdict data
│       ├── tools/
│       │   ├── history_tool.py   # queries historical failure rates from dataset
│       │   ├── log_tool.py       # queries recent failed builds from dataset
│       │   ├── dependency_tool.py # coverage-based test ranking
│       │   └── git_tool.py       # (future) fetches live commit diffs from GitHub
│       ├── data_loader.py        # loads dataset.csv
│       ├── features.py           # cleans features, applies SMOTE
│       ├── model.py              # trains Random Forest model
│       ├── ranking.py            # sorts tests by predicted failure probability
│       └── evaluation.py        # computes APFDc, APFD, Precision@k
├── scripts/
│   ├── run_agent.py              # runs Phase 1 (ML baseline)
│   └── run_llm_agent.py          # runs Phase 2 (LLM agent)
├── dataset.csv
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3a. Run the ML baseline (no API key needed)**
```bash
PYTHONPATH=src python3 scripts/run_agent.py --data dataset.csv
```

Expected output:
```
Loaded 3683 rows, 154 columns from dataset.csv
failure rate: 12.0%  |  features: 148
before SMOTE - failures: 351, passes: 2595
after SMOTE  — failures: 2595, passes: 2595
model trained on 5190 samples
APFD:          0.9361
APFDc:         0.9498
Precision@10:  1.0000
```

**3b. Run the LLM agent (requires Anthropic API key)**
```bash
export ANTHROPIC_API_KEY="your-key-here"
PYTHONPATH=src python3 scripts/run_llm_agent.py --data dataset.csv
```

---

## Notes

- Phase 1 uses Random Forest with `predict_proba` instead of RankLib — simpler, interpretable, and scored higher than the paper's benchmark. The 12% failure rate in this dataset (vs the paper's 3%) gives the model more signal to learn from.
- SMOTE was the key insight from Mendoza et al. Without balancing, the model just predicts pass every time since 88% of tests pass.
- Phase 2 agent works on pre-extracted features for now. `git_tool.py` is stubbed for future live GitHub integration.
- The research question: can an LLM agent match ML-based TCP without retraining, while being more maintainable and explainable?

---

## References

- Yaraghi et al. (2022). [arXiv:2109.13168](https://arxiv.org/abs/2109.13168)
- Mendoza et al. (2022). PROMISE '22.
- Spieker et al. (2017). *RETECS: Reinforcement Learning for TCP.*
