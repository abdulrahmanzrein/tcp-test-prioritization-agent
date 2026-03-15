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

```bash
pip install -r requirements.txt
python scripts/run_agent.py --data path/to/dataset.csv
```

---

## Notes

- Not rebuilding the feature extraction pipeline — only consuming pre-extracted features.
- Baseline model is a Random Forest classifier. The original paper uses a pairwise RF ranking model through RankLib but a classifier with probability scores is a reasonable starting point.
- Future directions: learning-to-rank models, reinforcement learning, LLM signals from commit messages.

---

## References

- Yaraghi et al. (2022). [arXiv:2109.13168](https://arxiv.org/abs/2109.13168)
- Mendoza et al. (2022). PROMISE '22.
- Spieker et al. (2017). *RETECS: Reinforcement Learning for TCP.*
