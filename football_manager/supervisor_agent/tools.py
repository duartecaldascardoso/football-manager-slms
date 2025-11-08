from langchain_core.tools import tool
from datetime import datetime


@tool
def obtain_current_date() -> str:
    """
    Tool to obtain the current date and time.
    Use this tool when you need to know what time it is or what day it is.

    Returns:
        str: Current date and time in the format 'YYYY-MM-DD HH:MM:SS'
    """
    print(
        "[TOOL CALL] obtain_current_date() called by Supervisor Agent"
    )
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def sum_numbers(a: float, b: float) -> float:
    """
    Tool to add two numbers together.
    Use this tool when you need to perform addition of two numeric values.

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        float: The sum of a and b
    """
    print(f"[TOOL CALL] sum_numbers(a={a}, b={b}) called by Supervisor Agent")
    return a + b
