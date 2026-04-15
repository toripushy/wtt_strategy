from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_IMPORTS = {"os", "subprocess", "requests", "socket", "pathlib", "shutil"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "__import__"}


class StrategySafetyError(ValueError):
    pass


def validate_strategy_source(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    raise StrategySafetyError(f"Forbidden import: {alias.name}")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                raise StrategySafetyError(f"Forbidden import: {node.module}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise StrategySafetyError(f"Forbidden call: {node.func.id}")
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub):
            raise StrategySafetyError("Negative indexing is blocked; use ctx.history(symbol, field, n) for past data.")


def validate_strategy_file(path: str | Path, require_strategies_dir: bool = True) -> None:
    strategy_path = Path(path)
    if require_strategies_dir and "strategies" not in strategy_path.parts:
        raise StrategySafetyError("Only files under strategies/ can be patched.")
    validate_strategy_source(strategy_path.read_text(encoding="utf-8"))
