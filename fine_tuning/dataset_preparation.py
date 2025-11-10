"""
Dataset Preparation Script for Fine-tuning Qwen3:8b
The dataset is formatted in the standard instruction-following format compatible with
Qwen models and other instruction-tuned LLMs.
"""

import json
from pathlib import Path
from typing import List, Dict

import logging

logger = logging.getLogger("dataset_preparation")
logging.basicConfig(level=logging.INFO)


def create_supervisor_routing_examples() -> List[Dict]:
    """
    Examples from the common usage of the agentic workflow.
    These examples will guide the model in which tools to use, which sub agents to call and how to respond. (on a small scale).
    """
    return [
        {
            "instruction": "You are a supervisor managing a football game system. Route this request: 'Tell me who the coaches are for both teams'",
            "input": "",
            "output": "This is a commentary-related request asking for coach information. I should delegate this to the Commenting Agent by calling the commenting_agent_tool with the request about coaches.",
            "system": "You are the supervisor coordinating a football game management system. Delegate tasks to specialized agents: Commenting Agent for commentary/player info, Game Manager Agent for game operations.",
        },
        {
            "instruction": "You are a supervisor managing a football game system. Route this request: 'Create a new game between Porto and Benfica, 90 minutes'",
            "input": "",
            "output": "This is a game management request to create a new game. I should delegate this to the Game Manager Agent by calling the game_manager_agent_tool with the game creation details.",
            "system": "You are the supervisor coordinating a football game management system. Delegate tasks to specialized agents: Commenting Agent for commentary/player info, Game Manager Agent for game operations.",
        },
        {
            "instruction": "You are a supervisor managing a football game system. Route this request: 'Who are the active players for the home team?'",
            "input": "",
            "output": "This is asking for player information, which is a commentary-related query. I should delegate this to the Commenting Agent by calling the commenting_agent_tool to obtain active players.",
            "system": "You are the supervisor coordinating a football game management system. Delegate tasks to specialized agents: Commenting Agent for commentary/player info, Game Manager Agent for game operations.",
        },
        {
            "instruction": "You are a supervisor managing a football game system. Route this request: 'Update the score - home team scored a goal'",
            "input": "",
            "output": "This is a game state update request. I should delegate this to the Game Manager Agent by calling the game_manager_agent_tool to update the score.",
            "system": "You are the supervisor coordinating a football game management system. Delegate tasks to specialized agents: Commenting Agent for commentary/player info, Game Manager Agent for game operations.",
        },
        {
            "instruction": "You are a supervisor managing a football game system. Route this request: 'Give me some commentary ideas for the current match'",
            "input": "",
            "output": "This is requesting commentary assistance. I should delegate this to the Commenting Agent by calling the commenting_agent_tool to generate commentary ideas.",
            "system": "You are the supervisor coordinating a football game management system. Delegate tasks to specialized agents: Commenting Agent for commentary/player info, Game Manager Agent for game operations.",
        },
    ]


def create_commentary_examples() -> List[Dict]:
    """
    Creates examples for generating engaging football commentary.
    This teaches the model to produce concise, relevant commentary in the football context.
    """
    return [
        {
            "instruction": "Generate commentary for a player scoring a goal",
            "input": "Player: Hulk, Team: Home, Minute: 34",
            "output": "Goal! What a strike from Hulk! The home team takes the lead in the 34th minute with an absolute thunderbolt. The power and precision from the Brazilian forward leaves the goalkeeper with no chance.",
            "system": "You are a professional football commentator providing engaging, concise match commentary.",
        },
        {
            "instruction": "Provide commentary on a tactical change",
            "input": "Coach Vítor Pereira is making a substitution, bringing on a defensive midfielder",
            "output": "Interesting tactical adjustment from Vítor Pereira. He's looking to shore up the midfield and protect the lead. A pragmatic decision that shows tactical awareness of the game situation.",
            "system": "You are a professional football commentator providing engaging, concise match commentary.",
        },
        {
            "instruction": "Comment on a goalkeeper save",
            "input": "Goalkeeper: Helton, Type: One-on-one save",
            "output": "Brilliant save by Helton! The striker was through on goal, but the home keeper stood tall and made himself big. That's the kind of save that wins matches - absolutely crucial.",
            "system": "You are a professional football commentator providing engaging, concise match commentary.",
        },
        {
            "instruction": "Generate commentary about team formation",
            "input": "Team is playing a 4-3-3 formation with emphasis on wing play",
            "output": "The team has set up in a 4-3-3 formation, clearly looking to exploit the width of the pitch. Both wingers are positioned high and wide, creating space for the full-backs to overlap. This offensive approach shows attacking intent.",
            "system": "You are a professional football commentator providing engaging, concise match commentary.",
        },
        {
            "instruction": "Comment on a red card incident",
            "input": "Player: Nemanja Matić, Reason: Serious foul play",
            "output": "Red card! Nemanja Matić is sent off for a reckless challenge. The referee had no choice - that was a dangerous tackle. The away team is now down to ten men, which completely changes the dynamic of this match.",
            "system": "You are a professional football commentator providing engaging, concise match commentary.",
        },
    ]


def create_game_management_examples() -> List[Dict]:
    """
    Creates examples for game management operations.
    This improves the model's understanding of game state manipulation and tool usage.
    """
    return [
        {
            "instruction": "A user wants to create a new football match",
            "input": "Set up a match between Porto and Benfica, 90 minutes long, call it 'Portuguese Clásico'",
            "output": "I'll create the game using the create_game tool with these parameters: game_name='Portuguese Clásico', game_length=90, home_team='Porto', away_team='Benfica'. This will initialize a new match with both teams and set the duration.",
            "system": "You are a game manager agent responsible for creating games, updating scores, and managing player actions.",
        },
        {
            "instruction": "Update the score after a goal",
            "input": "The home team just scored, update the match score",
            "output": "I'll use the update_game_score tool to increment the home team's score by 1. This will properly reflect the goal in the game state and maintain accurate score tracking.",
            "system": "You are a game manager agent responsible for creating games, updating scores, and managing player actions.",
        },
        {
            "instruction": "Handle a player expulsion",
            "input": "Expel player Nicolás Otamendi from the home team for receiving a red card",
            "output": "I'll use the kick_player_out tool with player_name='Nicolás Otamendi' and team='home'. This will remove the player from the active roster and add them to the expelled players list, reflecting the red card decision.",
            "system": "You are a game manager agent responsible for creating games, updating scores, and managing player actions.",
        },
        {
            "instruction": "End a completed match",
            "input": "The game with ID abc123 has finished, please end it",
            "output": "I'll use the end_game tool with game_id='abc123' to mark this match as completed. This will finalize the game state and preserve the final score.",
            "system": "You are a game manager agent responsible for creating games, updating scores, and managing player actions.",
        },
    ]


def create_tool_calling_examples() -> List[Dict]:
    """
    Creates examples demonstrating proper tool calling behavior.
    This reinforces accurate parameter extraction and tool selection.
    """
    return [
        {
            "instruction": "Extract the necessary parameters to create a game from this request",
            "input": "I want to start a match called 'Derby Day' between Manchester United and Manchester City for 90 minutes",
            "output": "Parameters extracted: game_name='Derby Day', home_team='Manchester United', away_team='Manchester City', game_length=90. These should be passed to the create_game tool.",
            "system": "You are an AI assistant that extracts structured information from natural language requests.",
        },
        {
            "instruction": "Identify which tool to use for this request",
            "input": "Show me all the players who are still on the field for the away team",
            "output": "This requires the obtain_active_players tool, filtered for the away team. This tool returns the current active players who haven't been expelled.",
            "system": "You are an AI assistant that selects the appropriate tool for user requests.",
        },
        {
            "instruction": "Determine the correct tool and parameters",
            "input": "Kick out João Moutinho from the home team",
            "output": "Use the kick_player_out tool with parameters: player_name='João Moutinho', team='home'. This will properly remove the player from active status.",
            "system": "You are an AI assistant that selects the appropriate tool for user requests.",
        },
        {
            "instruction": "What information do you need to update a game score?",
            "input": "",
            "output": "To update a game score, I need: game_id (the unique identifier of the game) and the team that scored ('home' or 'away'). The update_game_score tool will then increment the appropriate team's score by 1.",
            "system": "You are an AI assistant that understands game management operations.",
        },
        {
            "instruction": "Explain when to use the end_game tool",
            "input": "",
            "output": "The end_game tool should be used when a match has concluded and you want to finalize the game state. It requires the game_id parameter and marks the game status as completed, preserving the final score.",
            "system": "You are an AI assistant that understands game management operations.",
        },
        {
            "instruction": "How would you handle a request for multiple expelled players?",
            "input": "Show me which players have been sent off in this match",
            "output": "I would use the obtain_expelled_players tool from the Commenting Agent's toolkit. This tool maintains a list of all players who have received red cards and been removed from the game.",
            "system": "You are an AI assistant that understands game commentary operations.",
        },
    ]


def prepare_dataset(output_path: str = "data/training_dataset.jsonl"):
    """
    Obtains the examples and saves the dataset in JSONL format. (To have a complete training example per line).
    This format is widely supported by fine-tuning frameworks.
    """

    # Collect all examples
    all_examples = []
    all_examples.extend(create_supervisor_routing_examples())
    all_examples.extend(create_commentary_examples())
    all_examples.extend(create_game_management_examples())
    all_examples.extend(create_tool_calling_examples())

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info("The dataset was prepared and stored.")


if __name__ == "__main__":
    prepare_dataset()
