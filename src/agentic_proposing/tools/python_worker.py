from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from typing import Any
import ast
import builtins
import io
import json
import os
import sys
import traceback


DEFAULT_ALLOWED_IMPORTS = {
    "array", "bisect", "collections", "csv", "datetime", "decimal", "fractions",
    "functools", "heapq", "itertools", "json", "math", "operator", "random", "re",
    "statistics", "string", "time", "typing",
}
OPTIONAL_ALLOWED_IMPORTS = {"numpy", "pandas", "scipy", "sympy"}
FORBIDDEN_NAMES = {
    "breakpoint", "compile", "eval", "exec", "exit", "help", "input", "open", "quit",
    "__import__",
}


def _allowed_imports() -> set[str]:
    configured = {
        value.strip() for value in os.environ.get("AP_PYTHON_ALLOWED_IMPORTS", "").split(",")
        if value.strip()
    }
    return configured or (DEFAULT_ALLOWED_IMPORTS | OPTIONAL_ALLOWED_IMPORTS)


class _SessionValidator(ast.NodeVisitor):
    def __init__(self, allowed_imports: set[str]):
        self.allowed_imports = allowed_imports

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in self.allowed_imports:
                raise PermissionError(f"Import is not allowed in this Python session: {root}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if root not in self.allowed_imports:
            raise PermissionError(f"Import is not allowed in this Python session: {root}")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
            raise PermissionError(f"Name is not allowed in this Python session: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise PermissionError("Dunder attribute access is not allowed")
        self.generic_visit(node)


def _safe_repr(value: Any, limit: int = 8000) -> str:
    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<{type(value).__name__}>"
    return rendered[:limit]


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"type": type(value).__name__, "repr": _safe_repr(value)}


class PythonWorker:
    def __init__(self):
        self.allowed_imports = _allowed_imports()
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            root = name.split(".", 1)[0]
            if root not in self.allowed_imports:
                raise PermissionError(f"Import is not allowed in this Python session: {root}")
            return original_import(name, *args, **kwargs)

        safe_builtins = dict(vars(builtins))
        for name in FORBIDDEN_NAMES:
            safe_builtins.pop(name, None)
        safe_builtins["__import__"] = guarded_import
        self.namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__name__": "__agentic_python_session__",
        }

    def execute(self, code: str, max_output_chars: int) -> dict[str, Any]:
        output = io.StringIO()
        errors = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
            _SessionValidator(self.allowed_imports).visit(tree)
            result: Any = None
            prefix = tree.body
            last_expression: ast.expr | None = None
            if prefix and isinstance(prefix[-1], ast.Expr):
                last_expression = prefix[-1].value
                prefix = prefix[:-1]
            with redirect_stdout(output), redirect_stderr(errors):
                if prefix:
                    module = ast.Module(body=prefix, type_ignores=[])
                    exec(compile(module, "<python_session>", "exec"), self.namespace, self.namespace)
                if last_expression is not None:
                    expression = ast.Expression(body=last_expression)
                    result = eval(compile(expression, "<python_session>", "eval"), self.namespace, self.namespace)
            visible = {
                name: {"type": type(value).__name__, "repr": _safe_repr(value, 500)}
                for name, value in self.namespace.items()
                if not name.startswith("_") and name != "__builtins__"
            }
            return {
                "ok": True,
                "result": _jsonable(result),
                "stdout": output.getvalue()[:max_output_chars],
                "stderr": errors.getvalue()[:max_output_chars],
                "variables": visible,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=12)[:max_output_chars],
                "stdout": output.getvalue()[:max_output_chars],
                "stderr": errors.getvalue()[:max_output_chars],
            }


def main() -> int:
    worker = PythonWorker()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("action") == "ping":
                response = {"ok": True, "pong": True}
            elif request.get("action") == "exec":
                response = worker.execute(
                    str(request.get("code", "")),
                    min(int(request.get("max_output_chars", 20000)), 100000),
                )
            else:
                response = {"ok": False, "error": "unknown action"}
        except Exception as exc:
            response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
