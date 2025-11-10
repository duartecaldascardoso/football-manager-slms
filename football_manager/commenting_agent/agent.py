from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from typing import cast, Any
from langgraph.graph import MessagesState
from football_manager.commenting_agent.prompts import COMMENTING_AGENT_PROMPT
from football_manager.commenting_agent.tools import (
    generate_random_idea,
    obtain_active_players,
    obtain_expelled_players,
    obtain_coaches,
)

# Commented, this uses the Google Gemini model. Speculated to have around 40b parameters.
# Not exactly an SLM, but shows a fast and smaller model.
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# Using Qwen3, this has performance according to the GPU.
llm = ChatOllama(model="qwen3:8b", temperature=0.3)

commenting_agent = create_agent(
    llm,
    tools=[
        generate_random_idea,
        obtain_active_players,
        obtain_expelled_players,
        obtain_coaches,
    ],
)


@tool
def commenting_agent_tool(request: str) -> str:
    """
    Tool to delegate commentary tasks to the specialized commenting agent.
    Use this tool for any requests related to game commentary, player information, or coach details.

    Args:
        request: The commentary request or question to pass to the commenting agent

    Returns:
        str: The commenting agent's response
    """
    print(
        f"[TOOL CALL] commenting_agent_tool(request='{request}') called by Supervisor Agent"
    )

    messages = [
        SystemMessage(content=COMMENTING_AGENT_PROMPT),
        HumanMessage(content=request),
    ]

    state: MessagesState = {"messages": messages}

    result = commenting_agent.invoke(cast(Any, state))
    return result["messages"][-1].content
