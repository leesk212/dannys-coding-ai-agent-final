"""Agent assembly - built on top of DeepAgents' create_deep_agent().

This module wraps the DeepAgents SDK to create a coding agent with:
1. ModelFallbackMiddleware - OpenRouter → Ollama with circuit breaker
2. LongTermMemoryMiddleware - ChromaDB vector memory
3. SubAgentLifecycleMiddleware - dynamic sub-agent management
4. Agentic loop defense - iteration guards, stuck detection

Architecture (mirrors deepagents_cli/agent.py create_cli_agent):
  create_deep_agent(
      model=...,
      tools=[memory_store, memory_search, spawn_subagent, list_subagents],
      middleware=[fallback, ltm, subagent, ...],   # injected AFTER base stack
      backend=CompositeBackend(LocalShellBackend),  # filesystem + shell
      memory=[AGENTS.md paths],                     # DeepAgents MemoryMiddleware
  )
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend

from coding_agent.config import Settings, settings
from coding_agent.middleware.long_term_memory import LongTermMemoryMiddleware
from coding_agent.middleware.model_fallback import ModelFallbackMiddleware, _create_model
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
    """Defense mechanisms for the agentic loop."""

    def __init__(self, max_iterations: int = 25, max_retries: int = 3) -> None:
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.iteration_count = 0
        self.empty_response_count = 0
        self.tool_call_history: list[tuple[str, str]] = []

    def check_iteration(self) -> str | None:
        self.iteration_count += 1
        if self.iteration_count >= self.max_iterations:
            return (
                f"Reached maximum iterations ({self.max_iterations}). "
                "Stopping to prevent infinite loop. Here's a summary of what was accomplished."
            )
        return None

    def check_empty_response(self, response: str) -> bool:
        if not response or not response.strip():
            self.empty_response_count += 1
            if self.empty_response_count >= self.max_retries:
                logger.warning("Max empty responses reached (%d)", self.max_retries)
                return False
            logger.warning(
                "Empty response detected (attempt %d/%d)",
                self.empty_response_count, self.max_retries,
            )
            return True
        self.empty_response_count = 0
        return False

    def check_stuck(self, tool_name: str, args: str) -> str | None:
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
        self.iteration_count = 0
        self.empty_response_count = 0
        self.tool_call_history.clear()


def _setup_agents_md(agent_id: str = "coding-agent") -> list[str]:
    """Setup AGENTS.md files for DeepAgents MemoryMiddleware.

    Returns list of paths for MemoryMiddleware sources.
    Mirrors deepagents_cli/agent.py lines 950-957.
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


def _setup_backend(cwd: Path | None = None) -> CompositeBackend:
    """Setup filesystem + shell backend.

    Mirrors deepagents_cli/agent.py lines 1102-1189.
    """
    root_dir = cwd or Path.cwd()
    shell_env = os.environ.copy()

    # LocalShellBackend provides both filesystem ops and shell execution
    # virtual_mode=True enforces root_dir as the working directory
    backend = LocalShellBackend(
        root_dir=root_dir,
        inherit_env=True,
        env=shell_env,
        virtual_mode=True,
    )

    # Route large results to temp dirs (same pattern as CLI)
    large_results_backend = FilesystemBackend(
        root_dir=tempfile.mkdtemp(prefix="coding_agent_large_results_"),
        virtual_mode=True,
    )
    conversation_history_backend = FilesystemBackend(
        root_dir=tempfile.mkdtemp(prefix="coding_agent_conversation_history_"),
        virtual_mode=True,
    )

    return CompositeBackend(
        default=backend,
        routes={
            "/large_tool_results/": large_results_backend,
            "/conversation_history/": conversation_history_backend,
        },
    )


def create_coding_agent(
    custom_settings: Settings | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Create the coding agent using DeepAgents' create_deep_agent().

    This mirrors deepagents_cli/agent.py create_cli_agent() but adds:
    - ModelFallbackMiddleware (OpenRouter → Ollama)
    - LongTermMemoryMiddleware (ChromaDB vector memory)
    - SubAgentLifecycleMiddleware (dynamic sub-agents)

    Returns a dict with:
    - agent: The compiled LangGraph Pregel graph
    - backend: CompositeBackend for file operations
    - fallback_middleware: For status monitoring
    - memory_middleware: For memory access
    - subagent_middleware: For sub-agent monitoring
    - loop_guard: For loop defense
    """
    cfg = custom_settings or settings

    # 1. Create primary model
    primary_model_spec = cfg.model_priority[0] if cfg.model_priority else cfg.local_fallback_model
    primary_model = _create_model(primary_model_spec)

    # 2. Setup backend (filesystem + shell, like CLI)
    composite_backend = _setup_backend(cwd)

    # 3. Setup AGENTS.md memory sources (DeepAgents native memory)
    memory_sources = _setup_agents_md()

    # 4. Initialize our custom middleware
    fallback_mw = ModelFallbackMiddleware(
        models=cfg.get_all_models(),
        timeout=cfg.model_timeout,
    )

    ltm_mw = LongTermMemoryMiddleware(
        memory_dir=str(cfg.memory_dir),
    )

    subagent_mw = SubAgentLifecycleMiddleware(
        model=primary_model,
        max_concurrent=cfg.max_subagents,
    )

    loop_guard = AgentLoopGuard(max_iterations=cfg.max_iterations)

    # 5. Collect custom tools from our middleware
    custom_tools = ltm_mw.get_tools() + subagent_mw.get_tools()

    # 6. Build our middleware stack
    #    These are injected AFTER DeepAgents' base stack
    #    (TodoList, Skills, Filesystem, SubAgent, Summarization, PatchToolCalls)
    #    but BEFORE AnthropicPromptCachingMiddleware and MemoryMiddleware
    #    See deepagents/graph.py lines 292-322
    custom_middleware = [
        fallback_mw,
        ltm_mw,
        subagent_mw,
    ]

    # 7. Create agent via DeepAgents SDK
    #    This gives us the full DeepAgents stack:
    #    - TodoListMiddleware
    #    - SkillsMiddleware (if skills provided)
    #    - FilesystemMiddleware (ls, read_file, write_file, edit_file, glob, grep)
    #    - SubAgentMiddleware (task tool with general-purpose subagent)
    #    - SummarizationMiddleware
    #    - PatchToolCallsMiddleware
    #    - [our custom middleware here]
    #    - AnthropicPromptCachingMiddleware
    #    - MemoryMiddleware (AGENTS.md)
    agent = create_deep_agent(
        model=primary_model,
        system_prompt=SYSTEM_PROMPT,
        tools=custom_tools,
        backend=composite_backend,
        middleware=custom_middleware,
        memory=memory_sources,
    )

    logger.info(
        "Agent created with DeepAgents SDK. Model: %s, Memory sources: %s",
        primary_model_spec.name, memory_sources,
    )

    return {
        "agent": agent,
        "backend": composite_backend,
        "fallback_middleware": fallback_mw,
        "memory_middleware": ltm_mw,
        "subagent_middleware": subagent_mw,
        "loop_guard": loop_guard,
    }
