"""Chat page - split-panel UI with real-time SubAgent monitoring.

Left panel: Chat conversation
Right panel: Auto-appears when SubAgents are triggered, showing live status
"""

import time
import streamlit as st
from langchain_core.messages import HumanMessage

# ── Custom CSS for split-panel styling ─────────────────────────────────

CHAT_CSS = """
<style>
/* SubAgent panel cards */
.subagent-card {
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    background: #1a1a2e;
}
.subagent-card.running {
    border-color: #f0ad4e;
    animation: pulse 2s infinite;
}
.subagent-card.completed {
    border-color: #5cb85c;
}
.subagent-card.failed {
    border-color: #d9534f;
}
.subagent-card.pending {
    border-color: #5bc0de;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(240, 173, 78, 0.3); }
    50% { box-shadow: 0 0 12px 4px rgba(240, 173, 78, 0.3); }
}

/* Event timeline */
.event-item {
    padding: 4px 0;
    border-left: 2px solid #444;
    padding-left: 12px;
    margin-left: 8px;
    font-size: 0.85em;
}
.event-item.spawned { border-left-color: #5bc0de; }
.event-item.started { border-left-color: #f0ad4e; }
.event-item.completed { border-left-color: #5cb85c; }
.event-item.failed { border-left-color: #d9534f; }
</style>
"""

AGENT_TYPE_ICONS = {
    "code_writer": "✍️",
    "researcher": "🔍",
    "reviewer": "📋",
    "debugger": "🐛",
    "general": "🤖",
}

STATUS_ICONS = {
    "pending": "⏳",
    "running": "🔄",
    "completed": "✅",
    "failed": "❌",
}


def _render_subagent_panel(sa_mw) -> None:
    """Render the SubAgent monitoring panel (right side)."""
    registry = sa_mw.registry

    all_tasks = registry.get_all_tasks()
    active_tasks = [t for t in all_tasks if t["status"] in ("pending", "running")]
    done_tasks = [t for t in all_tasks if t["status"] in ("completed", "failed")]

    # Header with count
    active_count = len(active_tasks)
    st.markdown(f"### 🤖 SubAgents ({active_count} active)")

    if not all_tasks:
        st.info("No sub-agents triggered yet.\n\nSub-agents appear here automatically when the main agent delegates tasks.")
        return

    # ── Active agents (prominent) ──────────────────────────────────
    for task in active_tasks:
        icon = AGENT_TYPE_ICONS.get(task["agent_type"], "🤖")
        status_icon = STATUS_ICONS.get(task["status"], "❓")
        elapsed = time.time() - task["created_at"]

        with st.container(border=True):
            # Header row
            cols = st.columns([0.6, 0.4])
            cols[0].markdown(f"{icon} **{task['agent_type']}**")
            cols[1].markdown(f"{status_icon} `{task['status']}`")

            # Task description
            st.caption(task["task_description"][:120])

            # Progress indicator
            if task["status"] == "running":
                st.progress(min(elapsed / 30, 0.95), text=f"Running {elapsed:.0f}s...")
            else:
                st.progress(0.0, text="Waiting to start...")

            st.caption(f"ID: `{task['id']}`")

    # ── Completed agents (collapsed) ───────────────────────────────
    if done_tasks:
        st.markdown("---")
        st.markdown(f"**Completed** ({len(done_tasks)})")
        for task in done_tasks[:5]:  # Show last 5
            icon = AGENT_TYPE_ICONS.get(task["agent_type"], "🤖")
            status_icon = STATUS_ICONS[task["status"]]
            duration = ""
            if task.get("completed_at") and task.get("created_at"):
                duration = f" ({task['completed_at'] - task['created_at']:.1f}s)"

            with st.expander(f"{status_icon} {icon} {task['agent_type']}{duration}", expanded=False):
                st.caption(task["task_description"][:120])
                if task.get("result"):
                    st.code(task["result"][:500], language="text")
                if task.get("error"):
                    st.error(task["error"][:300])

    # ── Event timeline ─────────────────────────────────────────────
    events = registry.get_events_since(0)
    if events:
        st.markdown("---")
        with st.expander(f"📜 Timeline ({len(events)} events)", expanded=False):
            for ev in reversed(events[-15:]):  # Last 15 events
                event_icon = {
                    "spawned": "🆕", "started": "▶️",
                    "completed": "✅", "failed": "❌",
                }.get(ev["event_type"], "•")
                agent_icon = AGENT_TYPE_ICONS.get(ev["agent_type"], "🤖")
                ts = time.strftime("%H:%M:%S", time.localtime(ev["timestamp"]))
                st.markdown(
                    f"`{ts}` {event_icon} {agent_icon} **{ev['agent_type']}** "
                    f"`{ev['task_id']}` — {ev['event_type']}"
                )


def _render_tool_event(tool_name: str, tool_result: str, sa_mw) -> dict | None:
    """Process a tool event and return display info."""
    is_subagent_tool = tool_name in ("spawn_subagent", "list_subagents")
    return {
        "name": tool_name,
        "result": tool_result[:200],
        "is_subagent": is_subagent_tool,
    }


def render_chat() -> None:
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    components = st.session_state.get("agent_components")
    if not components:
        st.warning("Agent not initialized. Check Settings.")
        return

    agent = components["agent"]
    fallback_mw = components["fallback_middleware"]
    sa_mw = components["subagent_middleware"]
    loop_guard = components["loop_guard"]

    # Initialize subagent-related session state
    if "subagent_panel_visible" not in st.session_state:
        st.session_state.subagent_panel_visible = False
    if "current_tools" not in st.session_state:
        st.session_state.current_tools = []

    # Determine if subagent panel should be visible
    has_subagent_activity = bool(sa_mw.registry.get_all_tasks())
    show_panel = st.session_state.subagent_panel_visible or has_subagent_activity

    # ── Layout: split when subagents active ────────────────────────
    if show_panel:
        chat_col, agent_col = st.columns([3, 2])
    else:
        chat_col = st.container()
        agent_col = None

    # ── LEFT: Chat panel ───────────────────────────────────────────
    with chat_col:
        st.markdown("### 💬 Chat")

        # Display chat history
        for msg in st.session_state.chat_messages:
            role = msg["role"]
            content = msg["content"]
            with st.chat_message(role):
                st.markdown(content)
                if msg.get("model"):
                    st.caption(f"🧠 {msg['model']}")
                if msg.get("tools_used"):
                    tool_items = msg["tools_used"]
                    subagent_tools = [t for t in tool_items if t.get("is_subagent")]
                    other_tools = [t for t in tool_items if not t.get("is_subagent")]

                    if subagent_tools:
                        with st.expander(f"🤖 SubAgent calls ({len(subagent_tools)})"):
                            for t in subagent_tools:
                                st.code(f"{t['name']}: {t['result']}", language="text")
                    if other_tools:
                        with st.expander(f"🔧 Tools ({len(other_tools)})"):
                            for t in other_tools:
                                st.code(f"{t['name']}: {t['result']}", language="text")

        # Chat input
        if prompt := st.chat_input("Ask me anything about coding..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                tools_used = []
                response_text = ""

                loop_guard.reset()

                try:
                    config = {"configurable": {"thread_id": "webui-session"}}
                    inputs = {"messages": [HumanMessage(content=prompt)]}

                    with st.spinner("🧠 Agent is thinking..."):
                        result = agent.invoke(inputs, config=config)

                        messages = result.get("messages", [])
                        for msg in messages:
                            if hasattr(msg, "type"):
                                if msg.type == "ai" and msg.content:
                                    response_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                                elif msg.type == "tool":
                                    tool_name = getattr(msg, "name", "unknown")
                                    tool_content = str(msg.content)[:200] if msg.content else ""
                                    is_sa = tool_name in ("spawn_subagent", "list_subagents")
                                    tools_used.append({
                                        "name": tool_name,
                                        "result": tool_content,
                                        "is_subagent": is_sa,
                                    })

                    # Check if subagents were used → show panel
                    if any(t["is_subagent"] for t in tools_used):
                        st.session_state.subagent_panel_visible = True

                    if not response_text:
                        response_text = "(No response generated)"

                    response_placeholder.markdown(response_text)

                    model_name = fallback_mw.current_model or "unknown"
                    st.caption(f"🧠 {model_name}")

                    subagent_tools = [t for t in tools_used if t["is_subagent"]]
                    other_tools = [t for t in tools_used if not t["is_subagent"]]

                    if subagent_tools:
                        with st.expander(f"🤖 SubAgent calls ({len(subagent_tools)})"):
                            for t in subagent_tools:
                                st.code(f"{t['name']}: {t['result']}", language="text")
                    if other_tools:
                        with st.expander(f"🔧 Tools ({len(other_tools)})"):
                            for t in other_tools:
                                st.code(f"{t['name']}: {t['result']}", language="text")

                except Exception as e:
                    response_text = f"Error: {e}"
                    response_placeholder.error(response_text)

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "model": fallback_mw.current_model,
                    "tools_used": tools_used,
                })

                # Force rerun to update subagent panel
                if st.session_state.subagent_panel_visible:
                    st.rerun()

    # ── RIGHT: SubAgent panel (auto-appears) ───────────────────────
    if agent_col is not None:
        with agent_col:
            _render_subagent_panel(sa_mw)

    # ── Sidebar controls ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Chat Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.session_state.current_tools = []
                st.rerun()
        with col2:
            if st.button("📊 Toggle Panel"):
                st.session_state.subagent_panel_visible = not st.session_state.subagent_panel_visible
                st.rerun()

        st.markdown("### Model Status")
        status = fallback_mw.get_status()
        for m in status["models"]:
            icon = "🟢" if m["circuit_state"] == "closed" else "🔴" if m["circuit_state"] == "open" else "🟡"
            st.text(f"{icon} {m['name'][:30]}")

        # Quick test prompts
        st.markdown("### 🧪 Test Prompts")
        st.caption("Click to auto-fill chat input")

        test_prompts = [
            ("SubAgent Test", "Analyze the following task by spawning sub-agents: 1) A researcher to investigate best practices for Python error handling, 2) A code_writer to write an example implementation"),
            ("Memory Test", "Remember that I prefer Python type hints and Google-style docstrings. Then search memory to confirm it was saved."),
            ("Multi-Agent Review", "I need you to: spawn a code_writer to create a fibonacci function, then spawn a reviewer to review the code quality"),
            ("Fallback Test", "Write a simple hello world in Python"),
        ]
        for label, prompt in test_prompts:
            if st.button(f"▶ {label}", key=f"test_{label}"):
                st.session_state._test_prompt = prompt
                st.rerun()
