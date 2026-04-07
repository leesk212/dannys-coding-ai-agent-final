"""Dynamic SubAgent Lifecycle Management.

Provides tools for the main agent to spawn specialized sub-agents
(code_writer, researcher, reviewer, debugger, general) at runtime.
Sub-agents execute as independent LLM calls with specialized system prompts.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── SubAgent status lifecycle ──────────────────────────────────────────

class SubAgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── SubAgent type templates ────────────────────────────────────────────

SUBAGENT_TEMPLATES: dict[str, dict[str, str]] = {
    "code_writer": {
        "system_prompt": (
            "You are a code writing specialist. Write clean, efficient, well-tested code. "
            "Follow the existing code style and patterns in the project. "
            "Always explain your design decisions briefly."
        ),
    },
    "researcher": {
        "system_prompt": (
            "You are a research specialist. Thoroughly investigate the topic, "
            "search through files, documentation, and code to gather comprehensive information. "
            "Provide structured findings with sources."
        ),
    },
    "reviewer": {
        "system_prompt": (
            "You are a code review specialist. Review code for bugs, security issues, "
            "performance problems, and style violations. Provide specific, actionable feedback "
            "with line references."
        ),
    },
    "debugger": {
        "system_prompt": (
            "You are a debugging specialist. Systematically identify the root cause of issues. "
            "Read error messages carefully, trace execution flow, check edge cases, "
            "and propose targeted fixes."
        ),
    },
    "general": {
        "system_prompt": (
            "You are a general-purpose assistant. Help with any coding task as needed. "
            "Be thorough and precise."
        ),
    },
}


# ── SubAgent task data ─────────────────────────────────────────────────

@dataclass
class SubAgentTask:
    """Represents a dynamically created sub-agent task."""

    id: str
    task_description: str
    agent_type: str
    status: SubAgentStatus
    result: str | None = None
    error: str | None = None
    model_used: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    @classmethod
    def create(cls, task_description: str, agent_type: str = "general") -> SubAgentTask:
        return cls(
            id=uuid.uuid4().hex[:8],
            task_description=task_description,
            agent_type=agent_type,
            status=SubAgentStatus.PENDING,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_description": self.task_description,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "model_used": self.model_used,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# ── SubAgent Event (for UI timeline) ─────────────────────────────────

@dataclass
class SubAgentEvent:
    """An event in the sub-agent lifecycle, used by the WebUI to render updates."""

    timestamp: float
    task_id: str
    agent_type: str
    event_type: str  # "spawned", "started", "completed", "failed"
    description: str
    result: str | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "event_type": self.event_type,
            "description": self.description,
            "result": self.result,
        }


# ── SubAgent Registry ──────────────────────────────────────────────────

class SubAgentRegistry:
    """Registry for tracking active sub-agents and their lifecycle."""

    def __init__(self, max_concurrent: int = 3) -> None:
        self.tasks: dict[str, SubAgentTask] = {}
        self.history: list[SubAgentTask] = []
        self.max_concurrent = max_concurrent
        self.events: list[SubAgentEvent] = []

    def _emit(self, task: SubAgentTask, event_type: str, result: str | None = None) -> None:
        self.events.append(SubAgentEvent(
            timestamp=time.time(),
            task_id=task.id,
            agent_type=task.agent_type,
            event_type=event_type,
            description=task.task_description,
            result=result,
        ))

    def spawn(self, description: str, agent_type: str = "general") -> SubAgentTask:
        """Create and register a new sub-agent task."""
        active = sum(
            1 for t in self.tasks.values()
            if t.status in (SubAgentStatus.PENDING, SubAgentStatus.RUNNING)
        )
        if active >= self.max_concurrent:
            raise RuntimeError(
                f"Max concurrent sub-agents ({self.max_concurrent}) reached. "
                f"Wait for active tasks to complete."
            )

        if agent_type not in SUBAGENT_TEMPLATES:
            agent_type = "general"

        task = SubAgentTask.create(description, agent_type)
        self.tasks[task.id] = task
        self._emit(task, "spawned")
        logger.info("Spawned sub-agent [%s] type=%s: %s", task.id, agent_type, description[:80])
        return task

    def start(self, task_id: str) -> None:
        task = self.tasks[task_id]
        task.status = SubAgentStatus.RUNNING
        self._emit(task, "started")

    def complete(self, task_id: str, result: str, model_used: str | None = None) -> None:
        task = self.tasks[task_id]
        task.status = SubAgentStatus.COMPLETED
        task.result = result
        task.model_used = model_used
        task.completed_at = time.time()
        self._emit(task, "completed", result[:200] if result else None)
        logger.info("Sub-agent [%s] completed in %.1fs", task_id, task.completed_at - task.created_at)

    def fail(self, task_id: str, error: str) -> None:
        task = self.tasks[task_id]
        task.status = SubAgentStatus.FAILED
        task.error = error
        task.completed_at = time.time()
        self._emit(task, "failed", error)
        logger.warning("Sub-agent [%s] failed: %s", task_id, error)

    def cleanup(self) -> list[SubAgentTask]:
        """Move completed/failed tasks to history and remove from active."""
        done = [
            t for t in self.tasks.values()
            if t.status in (SubAgentStatus.COMPLETED, SubAgentStatus.FAILED)
        ]
        for t in done:
            self.history.append(t)
            del self.tasks[t.id]
        return done

    def get_active_summary(self) -> str:
        """Summary of active sub-agents for system prompt injection."""
        active = [
            t for t in self.tasks.values()
            if t.status in (SubAgentStatus.PENDING, SubAgentStatus.RUNNING)
        ]
        if not active:
            return "(no active sub-agents)"
        lines = []
        for t in active:
            elapsed = time.time() - t.created_at
            lines.append(
                f"[{t.id}] {t.status.value} ({t.agent_type}, {elapsed:.0f}s): "
                f"{t.task_description[:80]}"
            )
        return "\n".join(lines)

    def get_all_tasks(self) -> list[dict]:
        """Get all tasks (active + history) as dicts."""
        all_tasks = list(self.tasks.values()) + self.history
        all_tasks.sort(key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in all_tasks]

    def get_events_since(self, since: float = 0) -> list[dict]:
        """Get events since a timestamp (for incremental UI updates)."""
        return [e.to_dict() for e in self.events if e.timestamp > since]

    def clear_events(self) -> None:
        self.events.clear()


# ── SubAgent Lifecycle Middleware ──────────────────────────────────────

class SubAgentLifecycleMiddleware:
    """Manages dynamic sub-agent creation, execution, and cleanup.

    Provides tools:
    - spawn_subagent: Create and run a new sub-agent
    - list_subagents: View active and historical sub-agents
    """

    def __init__(
        self,
        model: BaseChatModel | None = None,
        max_concurrent: int = 3,
    ) -> None:
        self.registry = SubAgentRegistry(max_concurrent=max_concurrent)
        self._model = model

    def set_model(self, model: BaseChatModel) -> None:
        """Update the model used for sub-agent execution."""
        self._model = model

    def _execute_subagent(self, task: SubAgentTask) -> str:
        """Execute a sub-agent task synchronously."""
        if self._model is None:
            return "Error: No model configured for sub-agent execution"

        self.registry.start(task.id)

        template = SUBAGENT_TEMPLATES.get(task.agent_type, SUBAGENT_TEMPLATES["general"])
        system_prompt = template["system_prompt"]

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task.task_description),
            ]
            response = self._model.invoke(messages)
            result = response.content if isinstance(response.content, str) else str(response.content)
            # Capture which model was used
            model_name = (
                getattr(self._model, "model_name", None)
                or getattr(self._model, "model", None)
                or "unknown"
            )
            self.registry.complete(task.id, result, model_used=str(model_name))
            return result
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self.registry.fail(task.id, error)
            return f"Sub-agent failed: {error}"

    async def _aexecute_subagent(self, task: SubAgentTask) -> str:
        """Execute a sub-agent task asynchronously."""
        if self._model is None:
            return "Error: No model configured for sub-agent execution"

        self.registry.start(task.id)

        template = SUBAGENT_TEMPLATES.get(task.agent_type, SUBAGENT_TEMPLATES["general"])
        system_prompt = template["system_prompt"]

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task.task_description),
            ]
            response = await self._model.ainvoke(messages)
            result = response.content if isinstance(response.content, str) else str(response.content)
            # Capture which model was used
            model_name = (
                getattr(self._model, "model_name", None)
                or getattr(self._model, "model", None)
                or "unknown"
            )
            self.registry.complete(task.id, result, model_used=str(model_name))
            return result
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self.registry.fail(task.id, error)
            return f"Sub-agent failed: {error}"

    def get_tools(self) -> list:
        """Return sub-agent management tools for LangGraph."""
        middleware = self

        @tool
        def spawn_subagent(task_description: str, agent_type: str = "general") -> str:
            """Spawn a new sub-agent to handle a specific task.

            Args:
                task_description: Detailed description of the task for the sub-agent.
                agent_type: Type of sub-agent. One of: code_writer, researcher, reviewer, debugger, general.
            """
            try:
                task = middleware.registry.spawn(task_description, agent_type)
                result = middleware._execute_subagent(task)
                middleware.registry.cleanup()
                return f"[Sub-agent {task.id} ({agent_type})] Result:\n{result}"
            except RuntimeError as e:
                return str(e)

        @tool
        def list_subagents() -> str:
            """List all sub-agents (active and historical) with their status and results."""
            tasks = middleware.registry.get_all_tasks()
            if not tasks:
                return "No sub-agents have been created yet."

            lines = []
            for t in tasks:
                status = t["status"]
                elapsed = ""
                if t["completed_at"] and t["created_at"]:
                    elapsed = f" ({t['completed_at'] - t['created_at']:.1f}s)"
                lines.append(
                    f"[{t['id']}] {status}{elapsed} ({t['agent_type']}): "
                    f"{t['task_description'][:80]}"
                )
                if t.get("error"):
                    lines.append(f"  Error: {t['error']}")
            return "\n".join(lines)

        return [spawn_subagent, list_subagents]
