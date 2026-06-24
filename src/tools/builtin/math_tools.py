"""Safe math expression evaluator built-in tool."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

# ---------------------------------------------------------------------------
# Safe evaluator — whitelist-only AST walk
# ---------------------------------------------------------------------------

_SAFE_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_SAFE_UNARYOPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Invert: operator.invert,
}

_SAFE_NAMES: dict[str, Any] = {
    # Constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    # Functions
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "exp": math.exp,
    "expm1": math.expm1,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "log1p": math.log1p,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
    "hypot": math.hypot,
    "comb": math.comb,
    "perm": math.perm,
}


def _safe_eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, complex)):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in _SAFE_NAMES:
            raise ValueError(f"Unknown identifier: {node.id!r}")
        return _SAFE_NAMES[node.id]
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__!r}")
        return op_fn(_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__!r}")
        return op_fn(_safe_eval_node(node.operand))
    if isinstance(node, ast.Call):
        fn = _safe_eval_node(node.func)
        args = [_safe_eval_node(a) for a in node.args]
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in math expressions")
        return fn(*args)
    if isinstance(node, ast.IfExp):
        test = _safe_eval_node(node.test)
        return _safe_eval_node(node.body) if test else _safe_eval_node(node.orelse)
    if isinstance(node, (ast.List, ast.Tuple)):
        elts = node.elts if isinstance(node, ast.List) else node.elts
        return [_safe_eval_node(e) for e in elts]
    raise ValueError(f"Unsupported expression node: {type(node).__name__!r}")


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


def evaluate_math(expression: str) -> dict:
    """Safely evaluate a mathematical expression string."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval_node(tree)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return {"expression": expression, "error": str(exc)}
    except SyntaxError as exc:
        return {"expression": expression, "error": f"Syntax error: {exc}"}
    # Normalise complex numbers with zero imaginary part
    if isinstance(result, complex) and result.imag == 0:
        result = result.real
    return {"expression": expression, "result": result}


# ---------------------------------------------------------------------------
# OpenAI tool schema
# ---------------------------------------------------------------------------

EVALUATE_MATH_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "evaluate_math",
        "description": (
            "Evaluates a mathematical expression and returns the numeric result. "
            "Supports arithmetic (+, -, *, /, //, %, **), bitwise operations, "
            "and common math functions: sqrt, log, log2, log10, sin, cos, tan, "
            "asin, acos, atan, atan2, exp, ceil, floor, abs, round, factorial, "
            "gcd, lcm, comb, perm, hypot, degrees, radians, and constants pi, e, tau. "
            "Always use this tool for arithmetic or mathematical calculations instead "
            "of computing in your head."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A mathematical expression to evaluate, e.g. '2 ** 32', "
                        "'sqrt(2) * pi', 'factorial(12) / factorial(8)'."
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}
