"""Local Agent Protocol server for DeepAgents async subagents."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage

from coding_agent.config import settings
from coding_agent.middleware.long_term_memory import LongTermMemoryMiddleware
from coding_agent.middleware.model_fallback import ModelFallbackMiddleware, create_model

logger = logging.getLogger(__name__)

_ARGS: argparse.Namespace | None = None
_AGENT = None
_CONN = sqlite3.connect(":memory:", check_same_thread=False)
_CONN.row_factory = sqlite3.Row
_DB_LOCK = threading.RLock()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local async subagent server")
    parser.add_argument("--agent-type", required=True)
    parser.add_argument("--graph-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", required=True)
    return parser.parse_args()


def _init_db() -> None:
    with _DB_LOCK:
        _CONN.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id  TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            messages   TEXT NOT NULL DEFAULT '[]',
            values_    TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id       TEXT PRIMARY KEY,
            thread_id    TEXT NOT NULL REFERENCES threads(thread_id),
            assistant_id TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            created_at   TEXT NOT NULL,
            error        TEXT,
            partial_output TEXT NOT NULL DEFAULT ''
        );
        """)
        _CONN.commit()


def _get_thread(thread_id: str) -> dict[str, Any] | None:
    with _DB_LOCK:
        row = _CONN.execute(
            "SELECT thread_id, created_at, messages, values_ FROM threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "thread_id": row["thread_id"],
        "created_at": row["created_at"],
        "messages": json.loads(row["messages"]),
        "values": json.loads(row["values_"]),
    }


def _get_run(run_id: str) -> dict[str, Any] | None:
    with _DB_LOCK:
        row = _CONN.execute(
            "SELECT run_id, thread_id, assistant_id, status, created_at, error, partial_output FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def _message_text_delta(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(content) if content else ""

    blocks = getattr(message, "content_blocks", None)
    if blocks:
        return "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and "text" in block
        )
    return ""


def _msg_value(msg: Any, key: str, default=None):
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _msg_type(msg: Any) -> str:
    return str(_msg_value(msg, "type", "") or "")


def _msg_name(msg: Any) -> str:
    return str(_msg_value(msg, "name", "") or "")


def _msg_tool_calls(msg: Any) -> list[Any]:
    value = _msg_value(msg, "tool_calls", []) or []
    return value if isinstance(value, list) else []


def _tool_call_value(tool_call: Any, key: str, default=None):
    if isinstance(tool_call, dict):
        return tool_call.get(key, default)
    return getattr(tool_call, key, default)


def _truncate_line(text: str, limit: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _set_partial_output(run_id: str, text: str) -> None:
    with _DB_LOCK:
        _CONN.execute(
            "UPDATE runs SET partial_output = ? WHERE run_id = ?",
            (text, run_id),
        )
        _CONN.commit()


def _render_partial_output(progress_lines: list[str], output: str) -> str:
    parts: list[str] = []
    if progress_lines:
        parts.append("\n".join(progress_lines))
    if output:
        if parts:
            parts.append("")
        parts.append(output)
    return "\n".join(parts).strip()


def _extract_update_lines(chunk_data: Any) -> list[str]:
    if not isinstance(chunk_data, dict):
        return []
    lines: list[str] = []
    for _node, node_output in chunk_data.items():
        if not isinstance(node_output, dict):
            node_output = getattr(node_output, "value", None) or {}
        if not isinstance(node_output, dict):
            continue
        raw_msgs = node_output.get("messages", [])
        if not isinstance(raw_msgs, list):
            raw_msgs = getattr(raw_msgs, "value", None) or []
        if not isinstance(raw_msgs, list):
            raw_msgs = [raw_msgs] if raw_msgs else []
        for msg in raw_msgs:
            msg_type = _msg_type(msg).lower()
            if msg_type == "ai":
                tool_calls = _msg_tool_calls(msg)
                for tc in tool_calls:
                    tool_name = str(_tool_call_value(tc, "name", "") or "")
                    if tool_name:
                        lines.append(f"[tool] {tool_name}")
            elif msg_type == "tool":
                tool_name = _msg_name(msg)
                content = _truncate_line(_msg_value(msg, "content", ""))
                if tool_name and content:
                    lines.append(f"[tool-result] {tool_name}: {content}")
                elif tool_name:
                    lines.append(f"[tool-result] {tool_name}")
    return lines


def _bootstrap_agent():
    if _ARGS is None:
        raise RuntimeError("Server arguments were not initialized.")

    from deepagents.backends import LocalShellBackend

    model_spec = next(
        (spec for spec in settings.get_all_models() if spec.to_model_string() == _ARGS.model),
        settings.get_all_models()[0],
    )
    backend = LocalShellBackend(
        root_dir=str(Path(_ARGS.root_dir).resolve()),
        inherit_env=True,
        virtual_mode=False,
    )

    fallback = ModelFallbackMiddleware()
    ltm = LongTermMemoryMiddleware()
    root_dir = Path(_ARGS.root_dir).resolve()
    system_prompt = (
        f"{_ARGS.system_prompt}\n\n"
        "## Runtime Context\n"
        f"- Current working directory: `{root_dir}`\n"
        f"- Model: `{_ARGS.model}`\n"
        "- You are running in a local split-topology async subagent runtime.\n\n"
        "## File System Rules\n"
        f"- Use `{root_dir}` as your working directory.\n"
        f"- Prefer absolute paths rooted under `{root_dir}`.\n"
        "- File read/write and shell execution are allowed through the backend tools.\n"
        "- When creating or editing files, target the current project directory instead of temporary unrelated paths.\n"
    )

    return create_deep_agent(
        model=create_model(model_spec),
        system_prompt=system_prompt,
        middleware=[fallback, ltm],
        tools=ltm.get_tools(),
        backend=backend,
        memory=[],
        skills=[],
        debug=False,
        name=str(_ARGS.graph_id or _ARGS.agent_type),
    )


async def _execute_run(run_id: str, thread_id: str, user_message: str) -> None:
    with _DB_LOCK:
        _CONN.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", (run_id,))
        _CONN.commit()
    try:
        output = ""
        progress_lines = ["[status] subagent started", "[status] waiting for model output"]
        _set_partial_output(run_id, _render_partial_output(progress_lines, output))
        streamed = False
        if hasattr(_AGENT, "astream"):
            try:
                async for chunk in _AGENT.astream(
                    {"messages": [HumanMessage(user_message)]},
                    stream_mode=["messages", "updates"],
                ):
                    current_stream_mode = ""
                    payload = None
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        current_stream_mode, payload = chunk
                    elif isinstance(chunk, tuple) and len(chunk) == 3:
                        _namespace, current_stream_mode, payload = chunk

                    if current_stream_mode == "messages":
                        message = None
                        if isinstance(payload, tuple) and len(payload) == 2:
                            message, _metadata = payload
                        else:
                            message = payload
                        if message is None:
                            continue
                        delta = _message_text_delta(message)
                        if not delta:
                            continue
                        streamed = True
                        output += delta
                        _set_partial_output(run_id, _render_partial_output(progress_lines, output))
                    elif current_stream_mode == "updates":
                        new_lines = _extract_update_lines(payload)
                        changed = False
                        for line in new_lines:
                            if line and line not in progress_lines:
                                progress_lines.append(line)
                                changed = True
                        if changed:
                            _set_partial_output(run_id, _render_partial_output(progress_lines, output))
            except Exception:
                streamed = False
                output = ""

        if not streamed:
            progress_lines.append("[status] awaiting final response")
            _set_partial_output(run_id, _render_partial_output(progress_lines, output))
            result = await _AGENT.ainvoke({"messages": [HumanMessage(user_message)]})
            last = result["messages"][-1]
            output = last.content if isinstance(last.content, str) else json.dumps(last.content)

        assistant_msg = {"role": "assistant", "content": output}
        with _DB_LOCK:
            row = _CONN.execute(
                "SELECT messages FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        msgs = json.loads(row[0]) if row else []
        msgs.append(assistant_msg)
        serialized = json.dumps(msgs)
        with _DB_LOCK:
            _CONN.execute(
                "UPDATE threads SET messages = ?, values_ = ? WHERE thread_id = ?",
                (serialized, json.dumps({"messages": msgs}), thread_id),
            )
            _CONN.execute(
                "UPDATE runs SET status = 'success', partial_output = ? WHERE run_id = ?",
                (_render_partial_output(progress_lines, output), run_id),
            )
            _CONN.commit()
    except Exception as exc:  # noqa: BLE001
        progress_lines = locals().get("progress_lines", [])
        output = locals().get("output", "")
        if progress_lines is None:
            progress_lines = []
        progress_lines.append(f"[error] {_truncate_line(str(exc), 200)}")
        with _DB_LOCK:
            _CONN.execute(
                "UPDATE runs SET status = 'error', error = ?, partial_output = ? WHERE run_id = ?",
                (str(exc), _render_partial_output(progress_lines, output), run_id),
            )
            _CONN.commit()


@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _AGENT
    _init_db()
    _AGENT = _bootstrap_agent()
    yield


app = FastAPI(lifespan=_lifespan)


@app.get("/ok")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/threads")
async def create_thread() -> dict[str, Any]:
    thread_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    with _DB_LOCK:
        _CONN.execute(
            "INSERT INTO threads (thread_id, created_at) VALUES (?, ?)",
            (thread_id, now),
        )
        _CONN.commit()
    return {"thread_id": thread_id, "created_at": now, "messages": [], "values": {}}


@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request) -> dict[str, Any]:
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    body = await request.json()
    multitask_strategy = body.get("multitask_strategy")
    if multitask_strategy == "interrupt":
        with _DB_LOCK:
            _CONN.execute(
                "UPDATE runs SET status = 'cancelled' WHERE thread_id = ? AND status = 'running'",
                (thread_id,),
            )
            _CONN.execute(
                "UPDATE threads SET values_ = '{}' WHERE thread_id = ?",
                (thread_id,),
            )
            _CONN.commit()

    messages = (body.get("input") or {}).get("messages") or []
    user_message = next((m["content"] for m in messages if m.get("role") == "user"), "")

    if user_message:
        with _DB_LOCK:
            existing = json.loads(
                _CONN.execute(
                    "SELECT messages FROM threads WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()[0]
            )
        existing.append({"role": "user", "content": user_message})
        with _DB_LOCK:
            _CONN.execute(
                "UPDATE threads SET messages = ? WHERE thread_id = ?",
                (json.dumps(existing), thread_id),
            )
            _CONN.commit()

    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    assistant_id = body.get("assistant_id") or (_ARGS.graph_id or _ARGS.agent_type if _ARGS else "async-subagent")
    with _DB_LOCK:
        _CONN.execute(
            "INSERT INTO runs (run_id, thread_id, assistant_id, created_at) VALUES (?, ?, ?, ?)",
            (run_id, thread_id, assistant_id, now),
        )
        _CONN.commit()

    asyncio.ensure_future(_execute_run(run_id, thread_id, user_message))
    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "status": "pending",
        "created_at": now,
        "error": None,
    }


@app.get("/threads/{thread_id}/runs/{run_id}")
async def get_run(thread_id: str, run_id: str) -> dict[str, Any]:
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@app.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str) -> dict[str, Any]:
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    with _DB_LOCK:
        _CONN.execute("UPDATE runs SET status = 'cancelled' WHERE run_id = ?", (run_id,))
        _CONN.commit()
    return {**run, "status": "cancelled"}


def main() -> None:
    global _ARGS
    _ARGS = _parse_args()
    import uvicorn

    uvicorn.run(
        app,
        host=_ARGS.host,
        port=_ARGS.port,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
