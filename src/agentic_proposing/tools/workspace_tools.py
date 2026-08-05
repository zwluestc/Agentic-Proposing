from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import re

from agentic_proposing.core.models import SideEffect, ToolResult
from .base import BaseTool, ToolContext, ToolSpec


def _allowed_roots(context: ToolContext) -> list[Path]:
    workspace_root = context.layout.root.resolve()
    roots = [workspace_root]
    for item in context.services.get("source_roots", []):
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        roots.append(candidate.resolve())
    return list(dict.fromkeys(roots))


def _resolve_path(value: str, context: ToolContext, *, require_exists: bool = True) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = context.layout.root / candidate
    resolved = candidate.resolve()
    if not any(resolved == root or root in resolved.parents for root in _allowed_roots(context)):
        raise PermissionError(f"Path is outside configured workspace roots: {resolved}")
    if require_exists and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _display_path(path: Path, context: ToolContext) -> str:
    try:
        return str(path.relative_to(context.layout.root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _iter_glob(root: Path, pattern: str) -> Iterable[Path]:
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("Glob pattern must be relative and cannot contain '..'")
    yield from root.glob(pattern)


class WorkspaceGlobTool(BaseTool):
    spec = ToolSpec(
        name="workspace_glob",
        description="Find files or directories under an allowed workspace/source root using a relative glob pattern.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "root": {"type": "string", "default": "."},
                "include_directories": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
            },
            "required": ["pattern"],
        },
        read_only=True,
        parallelizable=True,
        deferred=True,
        tags=["workspace", "glob", "files", "discover", "list"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        root = _resolve_path(str(arguments.get("root", ".")), context)
        if not root.is_dir():
            raise NotADirectoryError(root)
        limit = min(int(arguments.get("limit", 500)), 2000)
        include_directories = bool(arguments.get("include_directories", False))
        rows: list[dict[str, Any]] = []
        truncated = False
        for item in _iter_glob(root, str(arguments["pattern"])):
            resolved = _resolve_path(str(item), context)
            if resolved.is_dir() and not include_directories:
                continue
            rows.append(
                {
                    "path": _display_path(resolved, context),
                    "type": "directory" if resolved.is_dir() else "file",
                    "size": resolved.stat().st_size if resolved.is_file() else None,
                }
            )
            if len(rows) >= limit:
                truncated = True
                break
        rows.sort(key=lambda row: row["path"])
        return ToolResult(
            call_id,
            self.spec.name,
            f"Found {len(rows)} workspace entries" + (" (truncated)" if truncated else ""),
            data={"results": rows, "truncated": truncated},
        )


class WorkspaceReadTool(BaseTool):
    spec = ToolSpec(
        name="workspace_read",
        description="Read a bounded UTF-8-compatible text range from an allowed workspace/source file with line numbers.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1, "default": 1},
                "line_count": {"type": "integer", "minimum": 1, "maximum": 5000},
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 100000},
            },
            "required": ["path"],
        },
        read_only=True,
        parallelizable=True,
        deferred=True,
        tags=["workspace", "read", "file", "inspect", "text"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        path = _resolve_path(str(arguments["path"]), context)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            raise ValueError("Binary files are not supported by workspace_read")
        text = raw.decode("utf-8-sig", errors="replace")
        lines = text.splitlines()
        start = int(arguments.get("start_line", 1))
        count = min(int(arguments.get("line_count", 400)), 5000)
        max_chars = min(int(arguments.get("max_chars", 30000)), 100000)
        selected = lines[start - 1 : start - 1 + count]
        rendered = "\n".join(f"{index}: {line}" for index, line in enumerate(selected, start))
        truncated = len(rendered) > max_chars or start - 1 + count < len(lines)
        rendered = rendered[:max_chars]
        return ToolResult(
            call_id,
            self.spec.name,
            f"Read {len(selected)} lines from {_display_path(path, context)}",
            data={
                "path": _display_path(path, context),
                "start_line": start,
                "total_lines": len(lines),
                "text": rendered,
                "truncated": truncated,
            },
        )


class WorkspaceGrepTool(BaseTool):
    spec = ToolSpec(
        name="workspace_grep",
        description="Search text across allowed workspace/source files with literal or regular-expression matching.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "root": {"type": "string", "default": "."},
                "include": {"type": "string", "default": "**/*"},
                "regex": {"type": "boolean", "default": False},
                "case_sensitive": {"type": "boolean", "default": False},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 2000},
                "max_file_bytes": {"type": "integer", "minimum": 1024, "maximum": 10000000},
            },
            "required": ["query"],
        },
        read_only=True,
        parallelizable=True,
        deferred=True,
        tags=["workspace", "grep", "search", "regex", "text"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        root = _resolve_path(str(arguments.get("root", ".")), context)
        if not root.is_dir():
            raise NotADirectoryError(root)
        flags = 0 if bool(arguments.get("case_sensitive", False)) else re.IGNORECASE
        query = str(arguments["query"])
        pattern = re.compile(query if bool(arguments.get("regex", False)) else re.escape(query), flags)
        max_matches = min(int(arguments.get("max_matches", 200)), 2000)
        max_file_bytes = min(int(arguments.get("max_file_bytes", 2_000_000)), 10_000_000)
        matches: list[dict[str, Any]] = []
        scanned = 0
        skipped = 0
        for item in _iter_glob(root, str(arguments.get("include", "**/*"))):
            if not item.is_file():
                continue
            resolved = _resolve_path(str(item), context)
            if resolved.stat().st_size > max_file_bytes:
                skipped += 1
                continue
            raw = resolved.read_bytes()
            if b"\x00" in raw[:8192]:
                skipped += 1
                continue
            scanned += 1
            for line_number, line in enumerate(raw.decode("utf-8-sig", errors="replace").splitlines(), 1):
                match = pattern.search(line)
                if not match:
                    continue
                matches.append(
                    {
                        "path": _display_path(resolved, context),
                        "line": line_number,
                        "column": match.start() + 1,
                        "text": line[:1000],
                    }
                )
                if len(matches) >= max_matches:
                    return ToolResult(
                        call_id,
                        self.spec.name,
                        f"Found {len(matches)} matches (truncated)",
                        data={"matches": matches, "scanned_files": scanned, "skipped_files": skipped, "truncated": True},
                    )
        return ToolResult(
            call_id,
            self.spec.name,
            f"Found {len(matches)} matches",
            data={"matches": matches, "scanned_files": scanned, "skipped_files": skipped, "truncated": False},
        )


class ArtifactWriteTool(BaseTool):
    spec = ToolSpec(
        name="artifact_write",
        description="Write a text or JSON artifact into the current run's managed artifacts directory.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "content": {},
                "category": {"type": "string", "enum": ["artifacts", "tool_outputs"]},
            },
            "required": ["name", "content"],
        },
        side_effect=SideEffect.WORKSPACE,
        read_only=False,
        deferred=True,
        tags=["workspace", "artifact", "write", "persist", "manage"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        category = str(arguments.get("category", "artifacts"))
        if category not in {"artifacts", "tool_outputs"}:
            raise ValueError("category must be artifacts or tool_outputs")
        content = arguments["content"]
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)
        ref = context.run_store.write_artifact(str(arguments["name"]), content, category)
        return ToolResult(
            call_id,
            self.spec.name,
            "Artifact written",
            data={"artifact_ref": ref},
            artifact_refs=[ref],
        )
