import os
import json
import anthropic
from tcp_agent.tools.history_tool import get_test_history, get_recent_failures
from tcp_agent.tools.log_tool import get_failed_builds, get_build_failure_summary
from tcp_agent.tools.dependency_tool import get_high_coverage_tests

# the system prompt tells Claude what it is and what to output
SYSTEM_PROMPT = """
You are a CI/CD test prioritization agent.

Given information about recent build failures, test failure rates, and coverage scores,
your job is to rank tests by how likely they are to fail in the next build.

You will receive:
- recent_failures: recent builds that failed and how many tests failed in each
- test_failure_rates: the top tests ranked by historical failure rate
- high_coverage_tests: tests that cover the most code

Output ONLY a JSON array ranked from most likely to fail first. No extra text.
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
    #call functions to store data
    recent_failures = get_failed_builds(dataset_path)
    failure_rates = get_recent_failures(dataset_path)
    coverage_tests = get_high_coverage_tests(dataset_path)


    #json.dumps converts the list of dicts into readable JSON text so Claude can parse it.
    context = f"""
    Recent failed builds:
    {json.dumps(recent_failures, indent=2)}

    Tests with highest failure rates:
    {json.dumps(failure_rates, indent=2)}

    Tests with highest coverage scores:
    {json.dumps(coverage_tests, indent=2)}
    """
    
    # call Claude with the system prompt and context
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}]
    )

    # parse Claude's JSON response into a Python list
    raw = response.content[0].text
    ranked = json.loads(raw)
    return ranked
