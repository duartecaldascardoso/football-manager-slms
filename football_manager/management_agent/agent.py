from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from typing import cast, Any
from langgraph.graph import MessagesState
from football_manager.management_agent.prompts import GAME_MANAGER_AGENT_PROMPT
from football_manager.management_agent.tools import (
    create_game,
    end_game,
    update_game_score,
    kick_player_out,
)

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

game_manager_agent = create_agent(
    llm,
    tools=[
        create_game,
        end_game,
        update_game_score,
        kick_player_out,
    ],
)


@tool
def game_manager_agent_tool(request: str) -> str:
    """
    Tool to delegate game management tasks to the specialized game manager agent.
    Use this tool for creating games, updating scores, ending games, or expelling players.

    Args:
        request: The game management request or command to pass to the game manager agent

    Returns:
        str: The game manager agent's response
    """
    print(
        f"[TOOL CALL] game_manager_agent_tool(request='{request}') called by Supervisor Agent"
    )

    messages = [
        SystemMessage(content=GAME_MANAGER_AGENT_PROMPT),
        HumanMessage(content=request),
    ]

    state: MessagesState = {"messages": messages}

    result = game_manager_agent.invoke(cast(Any, state))
    return result["messages"][-1].content
