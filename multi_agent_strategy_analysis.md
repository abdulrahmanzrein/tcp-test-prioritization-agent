# Agent Architecture Analysis: TCP Test Prioritization

This document analyzes the current one-agent workflow for test case prioritization (TCP) and proposes a multi-agent transition to handle large datasets effectively.

## 1. Current Workflow: One-Agent Limitations

The current system uses a single LLM agent (Claude/GPT) in a LangGraph loop. While effective for small pilot datasets, it faces several critical bottlenecks when scaled to large industrial test suites:

### Context Window Overflow
*   **The Issue:** Tools like `get_test_risk_profile` and `get_covered_code_risk` return raw feature vectors for **every test case**.
*   **The Difficulty:** In datasets with thousands of tests, the tool output exceeds the LLM's context window. The agent literally "forgets" the start of the list or crashes when receiving the tool response.
*   **Impact:** Failure to process projects with more than a few hundred tests.

### Reasoning Overload (The "Middle" Problem)
*   **The Issue:** The agent is asked to rank and provide a 2-3 sentence justification for every single test.
*   **The Difficulty:** LLMs struggle to maintain consistent reasoning across high-volume outputs. For 1,000 tests, the agent would need to generate ~2,000-3,000 sentences of reasoning in one response, which is computationally expensive and prone to truncation.
*   **Impact:** Inconsistent rankings and incomplete results.

### Token Inefficiency
*   **The Issue:** 97% of tests typically never fail (Mendoza et al., 2022). 
*   **The Difficulty:** The current workflow sends full feature data for these "low-signal" (T6) tests to the LLM, wasting tokens on tests that are eventually just sorted by cost.
*   **Impact:** High API costs and slow execution.

---

## 2. Proposed Multi-Agent Approach

To solve the scalability issue, we propose a **Two-Agent Workflow** that separates data filtering from fine-grained prioritization.

### Agent 1: The Filter Agent (The "Scanner")
*   **Role:** Iterates through the large dataset in manageable batches (e.g., 200 tests at a time).
*   **Task:** Identifies "High-Risk" tests that meet Tier 1–5 criteria (any history of failure or high coverage/complexity risk). It filters out the "Zero-Signal" (T6) tests.
*   **Output:** A condensed list of "Interest Tests" (T1-T5) + a summarized metadata object for the remainder.

### Agent 2: The Ranking Agent (The "Expert")
*   **Role:** Takes the filtered list from the Filter Agent.
*   **Task:** Performs the deep reasoning, applies the research-grounded tier rules, and generates the final JSON ranking for the High-Risk tests.
*   **Handling the Tail:** Automatically appends the T6 tests (provided by Agent 1) to the end of the ranking based on a simple "Cost Ascending" rule.

### How this improves performance and handles large data:

1.  **Context Management:** By splitting the task, the "Expert" agent only ever sees a subset of tests (the ~3-5% that actually matter for fault detection). This ensures it always operates within context limits.
2.  **Scalability:** The Filter Agent can process a dataset of 100,000 tests by iterating through batches without ever hitting a hard limit.
3.  **Cost Efficiency:** We stop paying for "Premium Reasoning" (high-end LLM calls) on the 97% of tests that have no failure signal.
4.  **Faster Response:** The final prioritized list is generated much faster because the model isn't trying to write a justification for 1,000 uninteresting tests.

> [!NOTE]
> This approach uses exactly **two agents**: one to solve the "Big Data" filtering problem and one to solve the "Precision Ranking" problem. This keeps the architecture simple while removing the primary bottleneck.
