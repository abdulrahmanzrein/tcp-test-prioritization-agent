from tcp_agent.tools.history_tool import get_test_history, get_all_failure_rates
from tcp_agent.tools.log_tool import get_failed_builds, get_build_failure_summary
from tcp_agent.tools.dependency_tool import get_high_coverage_tests
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.messages import AnyMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END

SYSTEM_PROMPT = """You are a CI/CD test prioritization agent.

Your job is to rank tests by how likely they are to fail in the next build.

You have tools available to gather information about test history, build failures, 
coverage scores, and execution times. Call whichever tools you need to make your decision.

When you have enough information, output ONLY a JSON array ranking ALL tests from most
likely to fail first. You must include every test from the dataset. No extra text.

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

#MessageState holds all previous messages and llm call amount
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def run_agent(dataset_path):
    model = init_chat_model(
        "claude-sonnet-4-6",
        temperature=0
    )

    tools = [get_test_history, get_all_failure_rates, get_failed_builds, get_build_failure_summary, get_high_coverage_tests]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    def llm_call(state):
        return {
            "messages": [
                model_with_tools.invoke(
                    [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    def tool_node(state):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            t = tools_by_name[tool_call["name"]]
            observation = t.invoke(tool_call["args"])
            result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        return {"messages": result}

    def should_continue(state):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        return END

    # build the graph
    agent = StateGraph(MessagesState)
    agent.add_node("llm_call", llm_call)
    agent.add_node("tool_node", tool_node)
    agent.add_edge(START, "llm_call")
    agent.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent.add_edge("tool_node", "llm_call")

    # compile and run
    compiled = agent.compile()
    from langchain.messages import HumanMessage
    result = compiled.invoke({"messages": [HumanMessage(content=f"Prioritize tests for the next build. The dataset is at: {dataset_path}")]})

    # parse the final ranking from Claude's last message
    import json
    raw = result["messages"][-1].content
    raw = raw.strip().replace("```json", "").replace("```", "")
    raw = raw[raw.find("["):raw.rfind("]") + 1]
    ranked = json.loads(raw)
    return ranked
