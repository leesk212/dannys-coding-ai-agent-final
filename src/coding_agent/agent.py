"""Agent assembly - built on top of DeepAgents CLI's create_cli_agent().

This module wraps the DeepAgents CLI to create a coding agent with:
1. ModelFallbackMiddleware - OpenRouter open-source models -> Ollama with circuit breaker
2. LongTermMemoryMiddleware - ChromaDB vector memory
3. SubAgentLifecycleMiddleware - dynamic sub-agent management
4. AgentLoopGuard - iteration guards, stuck detection

Architecture:
  create_cli_agent(
      model=openrouter:qwen/qwen3.5-35b-a3b,
      tools=[memory_store, memory_search, spawn_subagent, list_subagents],
      system_prompt=SYSTEM_PROMPT,
      enable_memory=True,    # DeepAgents native AGENTS.md memory
      enable_skills=True,    # DeepAgents skills system
      enable_shell=True,     # Filesystem + shell backend
  )
  +
  Custom middleware: [fallback_mw, ltm_mw, subagent_mw]
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from coding_agent.config import Settings, settings
from coding_agent.middleware.long_term_memory import LongTermMemoryMiddleware
from coding_agent.middleware.model_fallback import ModelFallbackMiddleware, create_model
from coding_agent.middleware.subagent_lifecycle import SubAgentLifecycleMiddleware

logger = logging.getLogger(__name__)

# ── System prompt (prepended before DeepAgents' BASE_AGENT_PROMPT) ─────

SYSTEM_PROMPT = """You are a Coding AI Agent, an advanced AI assistant specialized in software engineering tasks.

## Extended Capabilities (on top of DeepAgents)
- Store and recall knowledge from long-term vector memory (ChromaDB)
- Spawn specialized sub-agents dynamically for complex tasks
- Automatically switch between AI models for reliability (OpenRouter → Ollama)

## Memory Usage
- Use `memory_store` to save important learnings (user preferences, code patterns, domain knowledge)
- Use `memory_search` to find relevant past knowledge before starting tasks
- Categories: domain_knowledge, user_preferences, code_patterns, project_context

## Sub-Agent Usage
- Use `spawn_subagent` to delegate tasks to specialized agents:
  - `code_writer`: Writing new code or functions
  - `researcher`: Investigating codebases, documentation
  - `reviewer`: Code review and quality analysis
  - `debugger`: Root cause analysis and bug fixing
  - `general`: Any other task
- Use `list_subagents` to check status of previous sub-agent tasks
"""


class AgentLoopGuard:
    """Defense mechanisms for the agentic loop.

    Prevents:
    1. Infinite loops via max iteration count
    2. Empty response loops via consecutive empty response detection
    3. Stuck loops via repeated identical tool calls (MD5 hash comparison)
    """

    def __init__(self, max_iterations: int = 25, max_retries: int = 3) -> None:
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.iteration_count = 0
        self.empty_response_count = 0
        self.tool_call_history: list[tuple[str, str]] = []

    def check_iteration(self) -> str | None:
        """Check if max iterations exceeded. Returns warning message or None."""
        self.iteration_count += 1
        if self.iteration_count >= self.max_iterations:
            return (
                f"Reached maximum iterations ({self.max_iterations}). "
                "Stopping to prevent infinite loop. Here's a summary of what was accomplished."
            )
        return None

    def check_empty_response(self, response: str) -> bool:
        """Check for consecutive empty responses.

        Returns True if should retry (under limit), False if should stop.
        """
        if not response or not response.strip():
            self.empty_response_count += 1
            if self.empty_response_count >= self.max_retries:
                logger.warning("Max empty responses reached (%d)", self.max_retries)
                return False
            logger.warning(
                "Empty response detected (attempt %d/%d)",
                self.empty_response_count,
                self.max_retries,
            )
            return True
        self.empty_response_count = 0
        return False

    def check_stuck(self, tool_name: str, args: str) -> str | None:
        """Detect if agent is stuck calling the same tool repeatedly.

        Returns warning message if stuck, None otherwise.
        """
        args_hash = hashlib.md5(args.encode()).hexdigest()[:8]
        entry = (tool_name, args_hash)
        self.tool_call_history.append(entry)
        if len(self.tool_call_history) >= 3:
            last_3 = self.tool_call_history[-3:]
            if len(set(last_3)) == 1:
                self.tool_call_history.clear()
                return (
                    f"WARNING: You've called `{tool_name}` with the same arguments 3 times. "
                    "This suggests you're stuck. Try a different approach."
                )
        return None

    def reset(self) -> None:
        """Reset all counters for a new conversation turn."""
        self.iteration_count = 0
        self.empty_response_count = 0
        self.tool_call_history.clear()


def _setup_agents_md(agent_id: str = "coding-agent") -> list[str]:
    """Setup AGENTS.md files for DeepAgents MemoryMiddleware.

    Returns list of paths for MemoryMiddleware sources.
    Mirrors deepagents_cli/agent.py AGENTS.md setup pattern.
    """
    agent_dir = Path.home() / ".deepagents" / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "AGENTS.md"
    if not agent_md.exists():
        agent_md.write_text(
            "# Coding AI Agent Memory\n\n"
            "This file stores persistent memory for the agent.\n"
            "The agent can edit this file to remember information.\n"
        )

    # Also check for project-level AGENTS.md
    sources = [str(agent_md)]
    project_md = Path.cwd() / ".deepagents" / "AGENTS.md"
    if project_md.exists():
        sources.append(str(project_md))
    project_agents_md = Path.cwd() / ".agents" / "AGENTS.md"
    if project_agents_md.exists():
        sources.append(str(project_agents_md))

    return sources


def create_coding_agent(
    custom_settings: Settings | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Create the coding agent using DeepAgents CLI's create_cli_agent().

    This builds on top of the full DeepAgents stack:
    - FilesystemMiddleware (ls, read_file, write_file, edit_file, glob, grep)
    - SubAgentMiddleware (task tool)
    - SummarizationMiddleware
    - MemoryMiddleware (AGENTS.md)
    - AskUserMiddleware

    And adds our custom extensions:
    - ModelFallbackMiddleware (OpenRouter open-source models → Ollama)
    - LongTermMemoryMiddleware (ChromaDB vector memory)
    - SubAgentLifecycleMiddleware (dynamic sub-agents)
    - AgentLoopGuard (loop defense)

    Returns a dict with:
    - agent: The compiled LangGraph Pregel graph
    - backend: CompositeBackend for file operations
    - fallback_middleware: For status monitoring
    - memory_middleware: For memory access
    - subagent_middleware: For sub-agent monitoring
    - loop_guard: For loop defense
    """
    cfg = custom_settings or settings

    # 1. Initialize our custom middleware components
    fallback_mw = ModelFallbackMiddleware(
        models=cfg.get_all_models(),
        timeout=cfg.model_timeout,
    )

    ltm_mw = LongTermMemoryMiddleware(
        memory_dir=str(cfg.memory_dir),
    )

    # Get the best available model for sub-agent execution
    primary_model = fallback_mw.get_model_with_fallback()

    subagent_mw = SubAgentLifecycleMiddleware(
        model=primary_model,
        max_concurrent=cfg.max_subagents,
    )

    loop_guard = AgentLoopGuard(max_iterations=cfg.max_iterations)

    # 2. Collect custom tools from our middleware
    custom_tools = ltm_mw.get_tools() + subagent_mw.get_tools()

    # 3. Setup AGENTS.md memory sources
    memory_sources = _setup_agents_md()

    # 4. Determine working directory
    working_dir = cwd or Path.cwd()

    # 5. Create agent via DeepAgents CLI
    try:
        import inspect
        from deepagents_cli.agent import create_cli_agent

        # Build desired kwargs — we'll filter to only those the installed
        # version of create_cli_agent actually accepts, so we never blow up
        # on unexpected keyword arguments across different versions.
        desired_kwargs: dict[str, Any] = {
            "model": cfg.primary_model_string,
            "assistant_id": "coding-ai-agent",
            "tools": custom_tools,
            "system_prompt": SYSTEM_PROMPT,
            "interactive": False,
            "auto_approve": True,
            "enable_memory": True,
            "enable_skills": True,
            "enable_shell": True,
            "cwd": str(working_dir),
        }

        # Inspect the real signature and keep only supported params
        try:
            sig = inspect.signature(create_cli_agent)
            supported = set(sig.parameters.keys())
            # If **kwargs is present, pass everything
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not has_var_keyword:
                filtered = {k: v for k, v in desired_kwargs.items() if k in supported}
                skipped = set(desired_kwargs) - set(filtered)
                if skipped:
                    logger.info(
                        "create_cli_agent: skipped unsupported kwargs: %s",
                        skipped,
                    )
                desired_kwargs = filtered
        except (ValueError, TypeError):
            # If inspection fails, try with all kwargs
            pass

        logger.info(
            "Calling create_cli_agent with kwargs: %s",
            list(desired_kwargs.keys()),
        )
        agent, backend = create_cli_agent(**desired_kwargs)

        logger.info(
            "Agent created with DeepAgents CLI. Model: %s, Memory sources: %s",
            cfg.primary_model_string,
            memory_sources,
        )

        return {
            "agent": agent,
            "backend": backend,
            "fallback_middleware": fallback_mw,
            "memory_middleware": ltm_mw,
            "subagent_middleware": subagent_mw,
            "loop_guard": loop_guard,
        }

    except ImportError:
        # Fallback: if deepagents_cli is not installed, use langgraph directly
        logger.warning(
            "deepagents_cli not available, falling back to LangGraph create_react_agent"
        )
        return _create_agent_fallback(
            cfg, primary_model, custom_tools, fallback_mw, ltm_mw, subagent_mw, loop_guard,
        )
    except Exception as e:
        # If DeepAgents CLI fails for any other reason, fall back gracefully
        logger.warning(
            "create_cli_agent failed (%s: %s), falling back to LangGraph",
            type(e).__name__,
            e,
        )
        return _create_agent_fallback(
            cfg, primary_model, custom_tools, fallback_mw, ltm_mw, subagent_mw, loop_guard,
        )


def _create_agent_fallback(
    cfg: Settings,
    model,
    tools: list,
    fallback_mw: ModelFallbackMiddleware,
    ltm_mw: LongTermMemoryMiddleware,
    subagent_mw: SubAgentLifecycleMiddleware,
    loop_guard: AgentLoopGuard,
) -> dict[str, Any]:
    """Fallback agent creation using LangGraph directly.

    Used when deepagents_cli is not installed.
    Provides the same interface but without DeepAgents middleware stack.
    """
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    logger.info(
        "Agent created with LangGraph fallback. Model: %s",
        fallback_mw.current_model,
    )

    return {
        "agent": agent,
        "backend": None,
        "fallback_middleware": fallback_mw,
        "memory_middleware": ltm_mw,
        "subagent_middleware": subagent_mw,
        "loop_guard": loop_guard,
    }
