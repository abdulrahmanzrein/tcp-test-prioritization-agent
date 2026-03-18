import os
import json
import anthropic
from tcp_agent.tools.history_tool import get_test_history, get_recent_failures
from tcp_agent.tools.log_tool import get_failed_builds, get_build_failure_summary
from tcp_agent.tools.dependency_tool import get_high_coverage_tests

# the system prompt tells the LLM what it is and what to output
SYSTEM_PROMPT = """
You are a CI/CD test prioritization agent.

Given information about recent build failures, test failure rates, and coverage scores,
your job is to rank tests by how likely they are to fail in the next build.

You will receive:
- recent_failures: recent builds that failed and how many tests failed in each
- latest_build_failures: which specific tests failed in the most recent failed build
- test_failure_rates: the top tests ranked by historical failure rate
- high_coverage_tests: tests that cover the most code
- execution_times: average duration per test (prioritize fast-failing tests)

Output ONLY a JSON array ranking ALL tests from most likely to fail first. No extra text. Write reasons as natural sentences explaining why this test should be prioritized.
Format:
[
  {
    "test": "test_id",
    "priority": 1,
    "confidence": 0.87,
    "reason": "short explanation"
  }
]
"""


def run_agent(dataset_path):
    # call functions to store data
    recent_failures = get_failed_builds(dataset_path)
    failure_rates = get_recent_failures(dataset_path, n=46)
    coverage_tests = get_high_coverage_tests(dataset_path)

    # get execution times per test
    import pandas as pd
    df = pd.read_csv(dataset_path)
    exec_times = df.groupby("Test")["Duration"].mean().reset_index()
    exec_times.columns = ["test", "avg_duration"]
    exec_times = exec_times.sort_values("avg_duration", ascending=False).to_dict("records")

    # get which tests failed in the most recent failed build
    if recent_failures:
        latest_build = recent_failures[0]["build"]
        latest_failures = get_build_failure_summary(dataset_path, latest_build)
    else:
        latest_failures = {"failed_tests": []}

    # json.dumps converts the list of dicts into readable JSON text so the LLM can parse it
    context = f"""
    Recent failed builds:
    {json.dumps(recent_failures, indent=2)}

    Which tests failed in the most recent failed build:
    {json.dumps(latest_failures, indent=2)}

    Tests with highest failure rates:
    {json.dumps(failure_rates, indent=2)}

    Tests with highest coverage scores:
    {json.dumps(coverage_tests, indent=2)}

    Tests with longest average execution times:
    {json.dumps(exec_times, indent=2)}
    """

    # call Claude with the system prompt and context
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}]
    )

    # parse Claude's JSON response into a Python list
    raw = response.content[0].text

    # sometimes Claude wraps JSON in ```json ... ``` so we clean that up
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "")

    # grab just the JSON array between [ and ]
    raw = raw[raw.find("[") : raw.rfind("]") + 1]

    ranked = json.loads(raw)
    return ranked
