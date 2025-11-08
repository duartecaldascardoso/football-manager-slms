from langchain_core.tools import tool
import random

ACTIVE_PLAYERS = {
    "home": [
        "Helton",
        "Danilo",
        "Eliaquim Mangala",
        "Alex Sandro",
        "Nicolás Otamendi",
        "Fernando",
        "João Moutinho",
        "James Rodríguez",
        "Hulk",
        "Radamel Falcao",
        "Kelvin",
    ],
    "away": [
        "Artur Moraes",
        "Maxi Pereira",
        "Ezequiel Garay",
        "Luisão",
        "André Almeida",
        "Nemanja Matić",
        "Enzo Pérez",
        "Nicolás Gaitán",
        "Salvio",
        "Rodrigo",
        "Óscar Cardozo",
    ],
}
EXPELLED_PLAYERS = []
COACHES = {"home": "Vítor Pereira", "away": "Jorge Jesus"}


@tool
def generate_random_idea() -> str:
    """
    Tool to obtain a random idea for commenting the game.
    Use when the commenting agent has to comment a recent event in the game.

    Returns:
        str: Textual idea to help the commentator
    """
    print(
        "[TOOL CALL] generate_random_idea() called by Commenting Agent"
    )
    ideas = [
        "Discuss the tactical formation changes",
        "Highlight the intensity of the midfield battle",
        "Comment on the goalkeeper's outstanding saves",
        "Analyze the team's passing accuracy",
        "Talk about the crowd's energy and atmosphere",
        "Discuss a player's individual brilliance",
        "Comment on set-piece opportunities",
        "Analyze the defensive line positioning",
        "Highlight counter-attacking opportunities",
        "Discuss time management and game tempo",
    ]
    return random.choice(ideas)


@tool
def obtain_active_players() -> str:
    """
    Tool to obtain the active players.
    Use when the user asks for the players or when this information is needed to comment the game.

    Returns:
        str: Textual representation with all active players from both teams
    """
    print(
        "[TOOL CALL] obtain_active_players() called by Commenting Agent"
    )
    home_active = [p for p in ACTIVE_PLAYERS["home"] if p not in EXPELLED_PLAYERS]
    away_active = [p for p in ACTIVE_PLAYERS["away"] if p not in EXPELLED_PLAYERS]
    return f"Home Team Active Players: {', '.join(home_active)}\nAway Team Active Players: {', '.join(away_active)}"


@tool
def obtain_expelled_players() -> str:
    """
    Tool to obtain the expelled players.
    Use when the user asks for the expelled players or when this information is needed to comment the game.

    Returns:
        str: Textual representation with all expelled players, or a message if none have been expelled
    """
    print(
        "[TOOL CALL] obtain_expelled_players() called by Commenting Agent"
    )
    if not EXPELLED_PLAYERS:
        return "No players have been expelled yet."
    return f"Expelled Players: {', '.join(EXPELLED_PLAYERS)}"


@tool
def obtain_coaches() -> str:
    """
    Tool to obtain the names of the coaches from both teams.
    Use when you need to comment any coach decision, the user asks for a coach name or when coaches are mentioned.

    Returns:
        str: Textual representation with the names of both coaches
    """
    print("[TOOL CALL] obtain_coaches() called by Commenting Agent")
    return f"Home Team Coach: {COACHES['home']}\nAway Team Coach: {COACHES['away']}"
