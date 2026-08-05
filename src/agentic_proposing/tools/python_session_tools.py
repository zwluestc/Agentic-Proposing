from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
import json
import os
import subprocess
import sys
import uuid

from agentic_proposing.core.models import SideEffect, ToolResult
from .base import BaseTool, ToolContext, ToolSpec


@dataclass(slots=True)
class _PythonSession:
    run_id: str
    session_id: str
    root: Path
    process: subprocess.Popen[str]
    lock: RLock


class PythonSessionManager:
    """Manages per-run isolated Python workers with persistent in-memory namespaces."""

    def __init__(self, allowed_imports: list[str] | None = None, default_timeout: int = 20):
        self.allowed_imports = list(allowed_imports or [])
        self.default_timeout = default_timeout
        self._sessions: dict[tuple[str, str], _PythonSession] = {}
        self._lock = RLock()

    def create(self, context: ToolContext, requested_id: str | None = None) -> _PythonSession:
        session_id = requested_id or f"py_{uuid.uuid4().hex[:12]}"
        if not all(char.isalnum() or char in "-_" for char in session_id):
            raise ValueError("session_id may contain only letters, digits, '-' and '_'")
        key = (context.state.run_id, session_id)
        with self._lock:
            if key in self._sessions:
                raise FileExistsError(f"Python session already exists: {session_id}")
            root = context.run_store.run_dir / "python_sessions" / session_id
            root.mkdir(parents=True, exist_ok=False)
            worker = Path(__file__).with_name("python_worker.py").resolve()
            env: dict[str, str] = {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "PYTHONHASHSEED": "0",
                "TEMP": str(root),
                "TMP": str(root),
            }
            for name in ("SYSTEMROOT", "WINDIR"):
                if os.environ.get(name):
                    env[name] = os.environ[name]
            if self.allowed_imports:
                env["AP_PYTHON_ALLOWED_IMPORTS"] = ",".join(self.allowed_imports)
            process = subprocess.Popen(
                [sys.executable, "-I", "-u", str(worker)],
                cwd=str(root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            session = _PythonSession(context.state.run_id, session_id, root, process, RLock())
            self._sessions[key] = session
        try:
            response = self._exchange(session, {"action": "ping"}, 5)
            if not response.get("ok"):
                raise RuntimeError(f"Python worker failed to start: {response}")
        except Exception:
            self.close(context.state.run_id, session_id)
            raise
        return session

    def _get(self, run_id: str, session_id: str) -> _PythonSession:
        try:
            return self._sessions[(run_id, session_id)]
        except KeyError as exc:
            raise KeyError(f"Python session not found: {session_id}") from exc

    def _exchange(self, session: _PythonSession, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        with session.lock:
            process = session.process
            if process.poll() is not None or process.stdin is None or process.stdout is None:
                stderr = process.stderr.read()[:4000] if process.stderr else ""
                raise RuntimeError(f"Python worker is not running: {stderr}")
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(process.stdout.readline)
                try:
                    line = future.result(timeout=timeout)
                except FutureTimeout as exc:
                    process.kill()
                    raise TimeoutError(f"Python session exceeded {timeout} seconds and was terminated") from exc
            if not line:
                stderr = process.stderr.read()[:4000] if process.stderr else ""
                raise RuntimeError(f"Python worker closed without a response: {stderr}")
            response = json.loads(line)
            if not isinstance(response, dict):
                raise ValueError("Python worker response must be an object")
            return response

    def execute(
        self,
        run_id: str,
        session_id: str,
        code: str,
        timeout: int | None = None,
        max_output_chars: int = 20_000,
    ) -> dict[str, Any]:
        session = self._get(run_id, session_id)
        try:
            return self._exchange(
                session,
                {"action": "exec", "code": code, "max_output_chars": max_output_chars},
                timeout or self.default_timeout,
            )
        except Exception:
            if session.process.poll() is not None:
                with self._lock:
                    self._sessions.pop((run_id, session_id), None)
            raise

    def close(self, run_id: str, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop((run_id, session_id), None)
        if session is None:
            return False
        if session.process.poll() is None:
            session.process.terminate()
            try:
                session.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                session.process.kill()
                session.process.wait(timeout=3)
        for stream in (session.process.stdin, session.process.stdout, session.process.stderr):
            if stream is not None:
                stream.close()
        return True

    def close_run(self, run_id: str) -> None:
        with self._lock:
            ids = [session_id for current_run, session_id in self._sessions if current_run == run_id]
        for session_id in ids:
            self.close(run_id, session_id)

    def list_for_run(self, run_id: str) -> list[str]:
        with self._lock:
            return sorted(session_id for current_run, session_id in self._sessions if current_run == run_id)


class PythonSessionCreateTool(BaseTool):
    spec = ToolSpec(
        name="python_session_create",
        description="Create a persistent restricted Python worker for multi-cell analysis within the current run.",
        input_schema={
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
        },
        side_effect=SideEffect.PROCESS,
        read_only=False,
        deferred=True,
        tags=["python", "session", "analysis", "code", "create"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        session = context.services["python_session_manager"].create(
            context, str(arguments["session_id"]) if arguments.get("session_id") else None
        )
        return ToolResult(
            call_id,
            self.spec.name,
            f"Python session created: {session.session_id}",
            data={"session_id": session.session_id, "workspace": str(session.root)},
            state_updates={"python_sessions": context.services["python_session_manager"].list_for_run(context.state.run_id)},
        )


class PythonSessionExecTool(BaseTool):
    spec = ToolSpec(
        name="python_session_exec",
        description="Execute one code cell in a persistent restricted Python session and return stdout, result, errors, and visible variables.",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "code": {"type": "string", "maxLength": 100000},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
                "max_output_chars": {"type": "integer", "minimum": 1000, "maximum": 100000},
            },
            "required": ["session_id", "code"],
        },
        side_effect=SideEffect.PROCESS,
        read_only=False,
        deferred=True,
        tags=["python", "session", "execute", "analysis", "self-check"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        response = context.services["python_session_manager"].execute(
            context.state.run_id,
            str(arguments["session_id"]),
            str(arguments["code"]),
            int(arguments["timeout_seconds"]) if arguments.get("timeout_seconds") is not None else None,
            int(arguments.get("max_output_chars", 20_000)),
        )
        return ToolResult(
            call_id,
            self.spec.name,
            "Python cell executed" if response.get("ok") else "Python cell failed",
            data=response,
            is_error=not bool(response.get("ok")),
        )


class PythonSessionCloseTool(BaseTool):
    spec = ToolSpec(
        name="python_session_close",
        description="Terminate one persistent Python session and release its process resources.",
        input_schema={
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
        side_effect=SideEffect.PROCESS,
        read_only=False,
        deferred=True,
        tags=["python", "session", "close", "manage"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        session_id = str(arguments["session_id"])
        closed = context.services["python_session_manager"].close(context.state.run_id, session_id)
        return ToolResult(
            call_id,
            self.spec.name,
            "Python session closed" if closed else "Python session was already absent",
            data={"session_id": session_id, "closed": closed},
            state_updates={"python_sessions": context.services["python_session_manager"].list_for_run(context.state.run_id)},
        )
