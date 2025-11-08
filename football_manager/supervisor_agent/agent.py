from dotenv import load_dotenv
from typing import Any, cast
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from football_manager.commenting_agent.agent import commenting_agent_tool
from football_manager.management_agent.agent import game_manager_agent_tool
from football_manager.supervisor_agent.prompts import SUPERVISOR_AGENT_PROMPT
from football_manager.supervisor_agent.tools import obtain_current_date, sum_numbers

load_dotenv()

llm = ChatOllama(model="qwen3:8b", temperature=0.3)


def create_supervisor():
    supervisor_tools = [
        obtain_current_date,
        sum_numbers,
        # These are the sub agents. Using it as recommended by LangChain 1.0 docs.
        commenting_agent_tool,
        game_manager_agent_tool,
    ]

    supervisor = create_agent(
        llm,
        tools=supervisor_tools,
    )

    return supervisor


def main():
    supervisor = create_supervisor()

    state: MessagesState = {
        "messages": [SystemMessage(content=SUPERVISOR_AGENT_PROMPT)]
    }

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        try:
            state["messages"].append(HumanMessage(content=user_input))

            result = supervisor.invoke(cast(Any, state))

            state = cast(MessagesState, result)

            assistant_message = state["messages"][-1]

            print(f"\nAssistant: {assistant_message.content}")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
