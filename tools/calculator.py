import ast
import operator

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval(node.operand))
    # Anything else (names, calls, attributes) is rejected -> no code execution.
    raise ValueError("Unsupported or unsafe expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+ - * / ** %, parentheses).

    Use for ANY arithmetic (growth rates, ratios, differences) instead of
    computing mentally, which language models do unreliably.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        return f"Error: {exc}"