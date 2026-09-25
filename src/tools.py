"""Small arithmetic tool shared by SFT data creation and the future OpenEnv server."""
import ast
import math
import re
from decimal import Decimal

_ALLOWED = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Compute a basic arithmetic expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Expression using numbers, parentheses, +, -, *, or /."}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final numeric answer once",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "Final numeric answer, with no units or explanation."}},
                "required": ["answer"],
            },
        },
    },
]


class InvalidExpression(ValueError):
    """Input outside the calculator's arithmetic grammar."""


def parse_number(text: str) -> Decimal:
    """Read a numeric GSM8K answer or annotation result."""
    text = text.strip().replace(",", "").removeprefix("$").strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        raise InvalidExpression("non_numeric_result")
    return Decimal(text)


def render_number(value: Decimal) -> str:
    return format(value.normalize(), "f")


def calculate(expression: str) -> Decimal:
    """Evaluate only numeric literals, parentheses, +, -, *, and /."""
    if not isinstance(expression, str) or len(expression) > 100:
        raise InvalidExpression("invalid_arithmetic")
    try:
        tree = ast.parse(expression, mode="eval")
        if any(
            type(node) not in _ALLOWED
            or (isinstance(node, ast.Constant) and type(node.value) not in (int, float))
            for node in ast.walk(tree)
        ):
            raise ValueError("unsupported syntax")
        value = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})
        if type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError("invalid result")
        return Decimal(format(value, ".12g") if isinstance(value, float) else str(value))
    except Exception as exc:
        raise InvalidExpression("invalid_arithmetic") from exc


def calculator(expression: str) -> str:
    """Compute a basic arithmetic expression.

    Args:
        expression: A string such as "8 + (2 - 1)".

    Returns:
        The numeric result, or a generic error message if arithmetic fails.
    """
    try:
        return render_number(calculate(expression))
    except InvalidExpression:
        return "Invalid arithmetic expression."


def submit_answer(answer: str) -> str:
    """Submit the final numeric answer once the calculation is complete.

    Args:
        answer: The final numeric answer, with no units or explanation.

    Returns:
        A confirmation, or a generic error if the answer is not numeric.
    """
    try:
        parse_number(answer)
    except (InvalidExpression, TypeError, AttributeError):
        return "Invalid numeric answer."
    return "Answer submitted."
