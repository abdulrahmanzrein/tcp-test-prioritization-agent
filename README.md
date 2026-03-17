# TCP Test Prioritization Agent

An undergraduate research project building an ML agent for Test Case Prioritization (TCP) in CI environments.

Based on:
> Yaraghi et al. (2022). *Scalable and Accurate TCP in Continuous Integration Contexts.* IEEE TSE.
> Mendoza et al. (2022). *On the Effectiveness of Data Balancing Techniques in the Context of ML-Based TCP.* PROMISE '22.

---

## The Problem

When a developer pushes code, CI runs all regression tests. For large projects this can take hours. If we can predict which tests are most likely to fail, we can run them first and catch bugs faster.

---

## How It Works

The dataset from Yaraghi et al. contains ~150 pre-extracted features per test case from 25 open-source Java projects. Each row is one test case in a CI build. The features cover three things: execution history (how often it failed before, how old it is), test source code metrics (complexity, lines of code), and coverage (which files it exercises).

The agent loads this dataset, trains a model to predict failure probability, and ranks tests from highest to lowest risk. The main evaluation metric is **APFDc** — how early in the ranked list failing tests appear, weighted by execution time.

One thing I learned from reading the papers: the most predictive single feature is `Age` — how long the test has existed. Newer tests fail much more often. Also, the dataset is heavily imbalanced (most tests pass), so SMOTE is applied before training based on findings from the Mendoza et al. paper.

---

## Project Structure

```
tcp-test-prioritization-agent/
├── src/
│   └── tcp_agent/
│       ├── data_loader.py    # loads dataset.csv
│       ├── features.py       # cleans features, applies SMOTE
│       ├── model.py          # trains Random Forest model
│       ├── ranking.py        # sorts tests by predicted failure probability
│       └── evaluation.py     # computes APFDc, APFD, Precision@k
├── scripts/
│   └── run_agent.py
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

**3. Run the agent**
```bash
PYTHONPATH=src python3 scripts/run_agent.py --data dataset.csv
```

**Expected output:**
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

---

## Notes

- I'm consuming the pre-extracted feature dataset from Yaraghi et al. rather than building the extraction pipeline — that alone would be a separate project.
- I used Random Forest with `predict_proba` instead of RankLib because it's simpler, interpretable, and scored higher than the paper's reported benchmark (APFDc 0.94 vs 0.82). I think this is partly because this dataset has a 12% failure rate vs the paper's 3% — more signal for the model to learn from.
- SMOTE was the key insight from Mendoza et al. Without balancing, the model just predicts pass every time since 88% of tests pass.
- Future directions: GitHub MCP integration, LLM signals from commit diffs, learning-to-rank models.

---

## References

- Yaraghi et al. (2022). [arXiv:2109.13168](https://arxiv.org/abs/2109.13168)
- Mendoza et al. (2022). PROMISE '22.
- Spieker et al. (2017). *RETECS: Reinforcement Learning for TCP.*
