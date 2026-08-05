from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import json
import math
import os
import subprocess
import sys
import tempfile

from agentic_proposing.core.models import SideEffect, ToolResult
from .base import BaseTool, ToolContext, ToolSpec


SAFE_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "len": len,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
}


class _SafeExpression(ast.NodeVisitor):
    ALLOWED = {
        ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
        ast.UnaryOp, ast.UAdd, ast.USub, ast.Not, ast.BinOp, ast.Add, ast.Sub,
        ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BoolOp, ast.And,
        ast.Or, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.IfExp, ast.Name, ast.Load, ast.Call, ast.keyword, ast.Subscript, ast.Slice,
    }

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in self.ALLOWED:
            raise ValueError(f"Expression node is not allowed: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in SAFE_FUNCTIONS and node.id not in {"True", "False", "None"}:
            raise ValueError(f"Name is not allowed: {node.id}")

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            raise ValueError("Only allowlisted pure functions may be called")
        self.generic_visit(node)


class PythonEvalTool(BaseTool):
    spec = ToolSpec(
        name="python_eval",
        description="Evaluate a bounded pure mathematical expression with an AST allowlist; no imports, files, network, or process access.",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string", "maxLength": 4000}},
            "required": ["expression"],
        },
        read_only=True,
        parallelizable=True,
        deferred=True,
        tags=["python", "calculation", "deterministic", "check"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        expression = str(arguments["expression"])
        tree = ast.parse(expression, mode="eval")
        _SafeExpression().visit(tree)
        value = eval(compile(tree, "<python_eval>", "eval"), {"__builtins__": {}}, SAFE_FUNCTIONS)
        return ToolResult(call_id, self.spec.name, "Expression evaluated", data={"result": value})


class SkillPythonTool(BaseTool):
    spec = ToolSpec(
        name="skill_python",
        description=(
            "Run a trusted, versioned Python verifier shipped inside a registered Skill package. Arbitrary code is not accepted; "
            "the script path must resolve under that Skill's tools directory."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "skill_id": {"type": "string"},
                "script": {"type": "string"},
                "input": {"type": "object"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["skill_id", "script", "input"],
        },
        side_effect=SideEffect.PROCESS,
        read_only=True,
        deferred=True,
        tags=["python", "skill-tool", "verifier", "sandbox"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        registry = context.services["skill_registry"]
        package = registry.get_package(str(arguments["skill_id"]))
        tools_root = (package.root / "tools").resolve()
        script = (tools_root / str(arguments["script"])).resolve()
        if tools_root not in script.parents or script.suffix.lower() != ".py" or not script.is_file():
            raise PermissionError("Script must be an existing .py file inside the selected Skill tools directory")
        timeout = int(arguments.get("timeout_seconds", 15))
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
        payload = json.dumps(arguments["input"], ensure_ascii=False)
        completed = subprocess.run(
            [sys.executable, "-I", str(script)],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(tools_root),
            env=env,
            timeout=timeout,
            check=False,
        )
        stdout, stderr = completed.stdout[:20000], completed.stderr[:10000]
        data: dict[str, Any] = {"exit_code": completed.returncode, "stderr": stderr}
        try:
            data["result"] = json.loads(stdout)
        except json.JSONDecodeError:
            data["stdout"] = stdout
        return ToolResult(
            call_id,
            self.spec.name,
            "Skill verifier passed" if completed.returncode == 0 else "Skill verifier failed",
            data=data,
            is_error=completed.returncode != 0,
        )

