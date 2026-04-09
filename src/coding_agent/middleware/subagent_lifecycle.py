"""Lifecycle tracking for async subagent tool calls."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from coding_agent.async_subagent_manager import LocalAsyncSubagentManager


class SubAgentLifecycleMiddleware(AgentMiddleware):
    def __init__(self, runtime: LocalAsyncSubagentManager) -> None:
        super().__init__()
        self._runtime = runtime

    @staticmethod
    def _tool_content(result: ToolMessage | Command[Any]) -> str:
        content = getattr(result, "content", "")
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _task_id(text: str) -> str:
        payload = SubAgentLifecycleMiddleware._parse_json(text)
        for key in ("task_id", "thread_id"):
            value = str(payload.get(key, "") or "").strip()
            if value:
                return value
        match = re.search(r"task_id:\s*([a-f0-9-]{8,})", text, flags=re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if not text.startswith("{"):
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _after(self, request: ToolCallRequest, result: ToolMessage | Command[Any]) -> ToolMessage | Command[Any]:
        tool_name = request.tool_call.get("name")
        args = request.tool_call.get("args") or {}
        content = self._tool_content(result)

        if tool_name == "start_async_task" and isinstance(args, dict):
            role = str(args.get("subagent_type", "")).strip()
            task_id = self._task_id(content)
            payload = self._parse_json(content)
            run_id = str(payload.get("run_id", "") or "").strip() or None
            if task_id and role:
                self._runtime.bind_task(task_id, role=role, run_id=run_id)
            elif getattr(result, "status", "") == "error" and role:
                self._runtime.note_runtime_state(role, state="failed", error=content)

        elif tool_name == "check_async_task" and isinstance(args, dict):
            task_id = str(args.get("task_id", "")).strip()
            payload = self._parse_json(content)
            status = str(payload.get("status", "")).lower()
            if task_id and status in {"success", "completed"}:
                self._runtime.update_task_state(task_id=task_id, state="completed", detail=str(payload.get("result", "")))
            elif task_id and status == "cancelled":
                self._runtime.update_task_state(task_id=task_id, state="cancelled", detail=str(payload.get("error", "")))
            elif task_id and status == "error":
                self._runtime.update_task_state(task_id=task_id, state="failed", detail=str(payload.get("error", "")))

        elif tool_name == "cancel_async_task" and isinstance(args, dict):
            task_id = str(args.get("task_id", "")).strip()
            if task_id:
                self._runtime.update_task_state(task_id=task_id, state="cancelled", detail="cancel_async_task")

        elif tool_name == "update_async_task" and isinstance(args, dict):
            task_id = str(args.get("task_id", "")).strip()
            message = str(args.get("message", "")).strip()
            if task_id:
                self._runtime.update_task_state(task_id=task_id, state="assigned", detail=message)

        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return self._after(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return self._after(request, await handler(request))
