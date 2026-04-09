"""Durable state models for memory, subagent lifecycle, and loop runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class MemoryRecord:
    record_id: str
    layer: str
    content: str
    scope_key: str = "global"
    source: str = "agent"
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    correction_of: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class SubAgentRecord:
    agent_id: str
    role: str
    task_summary: str
    parent_id: str
    state: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    task_id: str | None = None
    run_id: str | None = None
    endpoint: str | None = None
    pid: int | None = None
    model: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoopRunRecord:
    run_id: str
    thread_id: str
    status: str
    current_step: str
    retries: int = 0
    failure_reason: str | None = None
    next_action: str | None = None
    model: str | None = None
    fallback_model: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
