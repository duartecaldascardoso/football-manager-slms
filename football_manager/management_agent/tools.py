from langchain_core.tools import tool
from datetime import datetime
import uuid

GAMES = {}


@tool
def create_game(
    game_name: str, game_length: int, home_team: str, away_team: str
) -> str:
    """
    Create a new football game with the specified details.
    Use this tool to initialize a new game with teams, duration, and other settings.

    Args:
        game_name: The name or title of the game
        game_length: The duration of the game in minutes
        home_team: The name of the home team
        away_team: The name of the away team

    Returns:
        str: Confirmation message with the game ID and game details
    """
    print(
        f"[TOOL CALL] create_game(game_name='{game_name}', game_length={game_length}, home_team='{home_team}', away_team='{away_team}') called by Game Manager Agent"
    )
    game_id = str(uuid.uuid4())[:8]
    GAMES[game_id] = {
        "name": game_name,
        "length": game_length,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": 0,
        "away_score": 0,
        "status": "active",
        "created_at": datetime.now(),
    }
    return f"Game created successfully! Game ID: {game_id} | {game_name} | {home_team} vs {away_team}"


@tool
def end_game(game_id: str) -> str:
    """
    Tool to mark a game as completed/ended.
    Use this tool when the match is over and you want to finalize the game.

    Args:
        game_id: The unique identifier of the game to end

    Returns:
        str: Confirmation message with the final score, or an error if the game is not found
    """
    print(f"[TOOL CALL] end_game(game_id='{game_id}') called by Game Manager Agent")
    if game_id not in GAMES:
        return f"Error: Game {game_id} not found."

    game = GAMES[game_id]
    if game["status"] == "ended":
        return f"Error: Game {game_id} has already ended."

    game["status"] = "ended"
    return f"Game ended! Final score: {game['home_team']} {game['home_score']} - {game['away_score']} {game['away_team']}"


@tool
def update_game_score(game_id: str, team: str, new_score: int) -> str:
    """
    Tool to update the score for a specific team in an active game.
    Use this tool when a team scores or when you need to correct the score.

    Args:
        game_id: The unique identifier of the game
        team: Which team to update ('home' or 'away')
        new_score: The new score value for the team

    Returns:
        str: Confirmation message with updated scores, or an error if the game is not found or ended
    """
    print(
        f"[TOOL CALL] update_game_score(game_id='{game_id}', team='{team}', new_score={new_score}) called by Game Manager Agent"
    )
    if game_id not in GAMES:
        return f"Error: Game {game_id} not found."

    game = GAMES[game_id]
    if game["status"] == "ended":
        return "Error: Cannot update score for an ended game."

    if team.lower() == "home":
        game["home_score"] = new_score
    elif team.lower() == "away":
        game["away_score"] = new_score
    else:
        return f"Error: Team must be 'home' or 'away', not '{team}'."

    return f"Score updated! {game['home_team']} {game['home_score']} - {game['away_score']} {game['away_team']}"


@tool
def kick_player_out(player_name: str) -> str:
    """
    Tool to expel a player from the game (red card).
    Use this tool when a player receives a red card and must be removed from the match.

    Args:
        player_name: The name of the player to expel from the game

    Returns:
        str: Confirmation message of the expulsion, or an error if the player is not found
    """
    print(
        f"[TOOL CALL] kick_player_out(player_name='{player_name}') called by Game Manager Agent"
    )
    from football_manager.commenting_agent.tools import ACTIVE_PLAYERS, EXPELLED_PLAYERS

    all_active = ACTIVE_PLAYERS["home"] + ACTIVE_PLAYERS["away"]

    if player_name not in all_active:
        return f"Error: Player '{player_name}' not found in the game."

    if player_name in EXPELLED_PLAYERS:
        return f"Error: Player '{player_name}' has already been expelled."

    EXPELLED_PLAYERS.append(player_name)
    return f"RED CARD! {player_name} has been expelled from the game!"
