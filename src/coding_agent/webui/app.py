"""Danny's Coding AI Agent — Single-page WebUI.

Run with: python -m coding_agent --webui
"""

import logging
import time
import traceback

import streamlit as st

st.set_page_config(
    page_title="Danny's Coding AI Agent",
    page_icon="data:,",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.getLogger("coding_agent").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    /* Sidebar collapsed by default */
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 0px;
    }
</style>
""", unsafe_allow_html=True)


def _init_state():
    defaults = {
        "agent_components": None,
        "chat_messages": [],
        "initialized": False,
        "init_error": None,
        "page": "chat",
        "mem_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _init_agent():
    if st.session_state.agent_components is not None:
        return

    init_area = st.empty()
    with init_area.container():
        st.markdown(
            "<h2 style='text-align:center'>Danny's Coding AI Agent</h2>"
            "<p style='text-align:center; color:#888'>Initializing...</p>",
            unsafe_allow_html=True,
        )
        progress = st.progress(0)
        log_area = st.empty()
        logs = []

        t_init_start = time.time()

        def log(icon, msg, pct):
            ts = time.strftime("%H:%M:%S")
            elapsed = f"{time.time() - t_init_start:.1f}s"
            logs.append(f"[{ts}] (+{elapsed}) {icon} {msg}")
            progress.progress(pct)
            log_area.code("\n".join(logs), language="text")

        try:
            log("⚙️", "Loading configuration...", 10)
            from coding_agent.config import settings
            key_ok = "✓" if settings.openrouter_api_key else "✗ NOT SET"
            log("🔑", f"API Key: {key_ok}", 15)
            models = settings.get_all_models()
            log("🧪", f"Models: {len(models)} configured", 20)

            t0 = time.time()
            log("🧠", "Initializing ChromaDB memory...", 35)
            from coding_agent.middleware.long_term_memory import LongTermMemoryMiddleware
            ltm_mw = LongTermMemoryMiddleware(memory_dir=str(settings.memory_dir))
            total = sum(ltm_mw.store.get_stats().values())
            st.session_state.mem_count = total
            log("✅", f"Memory ready ({total} entries) — {time.time()-t0:.1f}s", 40)

            t0 = time.time()
            log("🔄", "Building model fallback chain...", 55)
            from coding_agent.middleware.model_fallback import ModelFallbackMiddleware
            fallback_mw = ModelFallbackMiddleware(models=models, timeout=settings.model_timeout)
            log("✅", f"Fallback chain ready — {time.time()-t0:.1f}s", 60)

            t0 = time.time()
            log("🤖", "Setting up SubAgent system...", 70)
            from coding_agent.middleware.subagent_lifecycle import SubAgentLifecycleMiddleware
            primary_model = fallback_mw.get_model_with_fallback()
            subagent_mw = SubAgentLifecycleMiddleware(model=primary_model, max_concurrent=settings.max_subagents)
            log("✅", f"SubAgents ready (max {settings.max_subagents}) — {time.time()-t0:.1f}s", 75)

            t0 = time.time()
            log("🏗️", "Creating agent...", 85)
            from coding_agent.agent import AgentLoopGuard
            loop_guard = AgentLoopGuard(max_iterations=settings.max_iterations)
            custom_tools = ltm_mw.get_tools() + subagent_mw.get_tools()
            log("🔧", f"Tools: {', '.join(t.name for t in custom_tools)}", 88)

            agent, backend = None, None
            _deepagents_error: str | None = None

            # ── Step 1: verify deepagents_cli can be imported ────────
            try:
                import inspect
                from deepagents_cli.agent import create_cli_agent  # noqa: PLC0415
                _deepagents_import_ok = True
            except ImportError as _ie:
                _deepagents_import_ok = False
                _deepagents_error = (
                    f"ImportError: {_ie}\n"
                    f"Python: {__import__('sys').version.split()[0]}  "
                    f"(deepagents-cli requires ≥ 3.11)"
                )
                log("⚠️", f"deepagents_cli import failed: {_ie}", 92)

            # ── Step 2: try to build the agent ────────────────────────
            if _deepagents_import_ok:
                try:
                    kwargs = {
                        "model": settings.primary_model_string,
                        "assistant_id": "coding-ai-agent",
                        "tools": custom_tools,
                        "system_prompt": "",
                        "interactive": False,
                        "auto_approve": True,
                        "enable_memory": True,
                        "enable_skills": True,
                        "enable_shell": True,
                        "cwd": str(settings.memory_dir.parent),
                    }
                    try:
                        sig = inspect.signature(create_cli_agent)
                        supported = set(sig.parameters.keys())
                        if not any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in sig.parameters.values()
                        ):
                            kwargs = {k: v for k, v in kwargs.items() if k in supported}
                    except (ValueError, TypeError):
                        pass
                    agent, backend = create_cli_agent(**kwargs)
                    log("✅", f"DeepAgents CLI agent created — {time.time()-t0:.1f}s", 95)
                except Exception as _ce:
                    _deepagents_error = f"{type(_ce).__name__}: {_ce}"
                    log("⚠️", f"create_cli_agent failed: {_ce}", 92)

            # ── Step 3: LangGraph fallback if anything above failed ───
            if agent is None:
                _reason = _deepagents_error or "unknown error"
                log("🔄", f"LangGraph fallback (reason: {_reason[:80]})", 92)
                from langgraph.prebuilt import create_react_agent  # noqa: PLC0415
                agent = create_react_agent(
                    model=primary_model,
                    tools=custom_tools,
                    prompt="You are Danny's Coding AI Agent.",
                )
                log("✅", f"LangGraph agent created — {time.time()-t0:.1f}s", 95)

            total_elapsed = time.time() - t_init_start
            log("🚀", f"Ready! (total {total_elapsed:.1f}s)", 100)

            st.session_state.agent_components = {
                "agent": agent, "backend": backend,
                "fallback_middleware": fallback_mw, "memory_middleware": ltm_mw,
                "subagent_middleware": subagent_mw, "loop_guard": loop_guard,
            }
            st.session_state.initialized = True

        except Exception as e:
            st.session_state.init_error = traceback.format_exc()
            st.session_state.initialized = False
            st.error(f"Failed: {e}")
            st.code(st.session_state.init_error, language="python")
            return

    init_area.empty()
    # Init 완료 → 즉시 rerun하여 clean render (Init UI 잔재 없이 chat 진입)
    if st.session_state.initialized:
        st.rerun()


def _render_home():
    st.markdown(
        "<h1 style='text-align:center; margin-top:40px'>Danny's Coding AI Agent</h1>"
        "<p style='text-align:center; color:#64748b; font-size:1.1em'>"
        "Agentic Loop &middot; Long-Term Memory &middot; Dynamic SubAgents &middot; Model Fallback</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    comp = st.session_state.agent_components
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Model", (comp["fallback_middleware"].current_model or "ready")[:25])
    with col2:
        st.metric("Memory Entries", st.session_state.mem_count)
    with col3:
        st.metric("SubAgent Tasks", len(comp["subagent_middleware"].registry.get_all_tasks()))
    with col4:
        st.metric("Models Available", len(comp["fallback_middleware"].models))

    st.markdown("<br>", unsafe_allow_html=True)
    _, c2, _ = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            "<div style='text-align:center; padding:20px; "
            "background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0'>"
            "<p style='font-size:1.2em'>Ready to start coding?</p></div>",
            unsafe_allow_html=True,
        )
        # No st.rerun() — just set page, Streamlit handles via query_params
        if st.button("💬 Start Chatting", use_container_width=True, type="primary"):
            st.session_state.page = "chat"
            st.rerun()  # Only rerun for explicit page navigation

    st.markdown("---")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**Model Fallback**\n\nOpenRouter models with circuit breaker auto-fallback.")
    with a2:
        st.markdown("**Long-Term Memory**\n\nChromaDB vector store with 4 knowledge categories.")
    with a3:
        st.markdown("**Dynamic SubAgents**\n\nSpecialized agents: code_writer, researcher, reviewer, debugger.")


def _render_history():
    """History page — session flow visualization with Mermaid FlowChart."""
    st.markdown("### 📜 Chat History")

    msgs = st.session_state.chat_messages
    if not msgs:
        st.info("No chat history yet. Start chatting first!")
        return

    if st.button("🗑️ Clear All History"):
        st.session_state.chat_messages = []
        st.rerun()

    # Group into sessions (user + assistant pairs)
    sessions = []
    current_session = []
    for msg in msgs:
        current_session.append(msg)
        if msg["role"] == "assistant":
            sessions.append(current_session)
            current_session = []
    if current_session:
        sessions.append(current_session)

    # Import Mermaid renderer from chat page
    from coding_agent.webui._pages.chat import _build_page_html

    for si, session in enumerate(reversed(sessions)):
        user_msg = next((m for m in session if m["role"] == "user"), None)
        asst_msg = next((m for m in session if m["role"] == "assistant"), None)

        prompt_preview = (user_msg["content"][:60] + "…") if user_msg and len(user_msg["content"]) > 60 else (user_msg["content"] if user_msg else "?")
        st.markdown(f"#### Session {len(sessions) - si} — {prompt_preview}")

        # ── FlowChart: Mermaid 스냅샷 (저장된 경우) ───────
        if asst_msg and asst_msg.get("mermaid_def"):
            mermaid_def = asst_msg["mermaid_def"]
            mermaid_tips = asst_msg.get("mermaid_tooltips", {})
            mermaid_evts = asst_msg.get("mermaid_events", [])
            num_agents = asst_msg.get("num_agents", 0)

            with st.expander("📊 FlowChart", expanded=False):
                html = _build_page_html(mermaid_def, mermaid_evts, False, tooltips=mermaid_tips)
                h = max(420, 260 + num_agents * 70)
                st.iframe(html, height=h)

        # ── User + Agent 결과 ─────────────────────────────
        c1, c2 = st.columns([2, 3])

        with c1:
            st.markdown("**👤 User**")
            st.info(user_msg["content"] if user_msg else "")

        with c2:
            st.markdown("**🤖 Agent**")
            if asst_msg:
                # 전체 결과 표시 (잘림 없음)
                content = asst_msg["content"] or ""
                if len(content) > 500:
                    # 긴 결과: 접힌 형태로
                    st.success(content[:500] + "…")
                    with st.expander("📄 Full Response"):
                        st.markdown(content)
                else:
                    st.success(content)
                if asst_msg.get("model"):
                    st.caption(f"🧠 {asst_msg['model']}")
                tools = asst_msg.get("tools_used", [])
                if tools:
                    tool_names = ", ".join(t["name"] for t in tools)
                    st.caption(f"🔧 Tools: {tool_names}")

        # ── Activity Log (접힌 형태) ──────────────────────
        if asst_msg and asst_msg.get("activity_log"):
            with st.expander("📡 Event Feed", expanded=False):
                for icon, text in asst_msg["activity_log"]:
                    st.markdown(f"{icon} {text}", unsafe_allow_html=True)

        st.markdown("---")


def main():
    _init_state()
    _init_agent()

    if not st.session_state.initialized:
        st.stop()

    # ── Sidebar: navigation only ─────────────────────────────────────
    with st.sidebar:
        st.markdown("### Danny's Coding AI Agent")
        st.markdown("---")

        # Navigation buttons — only rerun on actual page change
        for label, page_id in [("🏠 Home", "home"), ("💬 Chat", "chat"),
                                ("📜 History", "history"), ("⚙️ Settings", "settings")]:
            btn_type = "primary" if st.session_state.page == page_id else "secondary"
            if st.button(label, use_container_width=True, type=btn_type, key=f"nav_{page_id}"):
                if st.session_state.page != page_id:
                    st.session_state.page = page_id
                    st.rerun()  # Only rerun for page navigation

        st.markdown("---")
        comp = st.session_state.agent_components
        st.success(f"Model: {comp['fallback_middleware'].current_model or 'ready'}")
        st.info(f"Memory: {st.session_state.mem_count} entries")

        with st.expander("🔒 Admin"):
            if st.button("🔄 Reinitialize", use_container_width=True):
                st.session_state.agent_components = None
                st.session_state.initialized = False
                st.rerun()

    # ── Page routing ─────────────────────────────────────────────────
    page = st.session_state.page
    if page == "home":
        _render_home()
    elif page == "history":
        _render_history()
    elif page == "settings":
        from coding_agent.webui._pages.settings import render_settings
        render_settings()
    else:
        from coding_agent.webui._pages.chat import render_chat
        render_chat()


if __name__ == "__main__":
    main()
