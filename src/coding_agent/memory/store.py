"""ChromaDB-backed vector store for long-term memory."""

from __future__ import annotations

import logging
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from coding_agent.memory.categories import MemoryCategory

logger = logging.getLogger(__name__)


def _resolve_persist_path(persist_dir: str) -> str:
    """Resolve the persistence path, handling WSL/Windows filesystem issues.

    ChromaDB uses SQLite internally, which has file-locking problems on
    Windows filesystem mounts (/mnt/c/, /mnt/d/) in WSL. When such a path
    is detected, we redirect to a Linux-native path instead.
    """
    expanded = os.path.expanduser(persist_dir)

    # Detect WSL + Windows mount path
    is_wsl = "microsoft" in platform.uname().release.lower() or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterp")
    is_windows_mount = expanded.startswith("/mnt/") and len(expanded) > 5 and expanded[5].isalpha()

    if is_wsl and is_windows_mount:
        # Redirect to Linux-native home directory
        linux_path = os.path.expanduser("~/.coding_agent/memory")
        logger.warning(
            "WSL detected with Windows mount path (%s). "
            "Redirecting ChromaDB storage to Linux path: %s",
            expanded,
            linux_path,
        )
        return linux_path

    return expanded


class LongTermMemory:
    """ChromaDB-backed vector store for persistent memory.

    Uses ChromaDB's built-in embedding (all-MiniLM-L6-v2) for semantic search.
    One collection per memory category for clean separation.
    """

    def __init__(self, persist_dir: str = "~/.coding_agent/memory") -> None:
        import chromadb

        persist_path = _resolve_persist_path(persist_dir)
        os.makedirs(persist_path, exist_ok=True)

        self.client = None
        self._ephemeral = False

        # Try 1: PersistentClient at resolved path
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
        except Exception as e1:
            logger.warning(
                "Failed to create ChromaDB at %s (%s). Trying /tmp…",
                persist_path, e1,
            )
            # Try 2: PersistentClient at /tmp
            try:
                fallback_path = "/tmp/coding_agent_memory"
                os.makedirs(fallback_path, exist_ok=True)
                self.client = chromadb.PersistentClient(path=fallback_path)
                persist_path = fallback_path
            except Exception as e2:
                logger.warning(
                    "PersistentClient /tmp also failed (%s). "
                    "Falling back to EphemeralClient (in-memory, non-persistent).",
                    e2,
                )
                # Try 3: EphemeralClient (in-memory — data lost on restart)
                try:
                    self.client = chromadb.EphemeralClient()
                    self._ephemeral = True
                    persist_path = "(in-memory)"
                except Exception as e3:
                    # Try 4: bare Client() — oldest API
                    logger.warning("EphemeralClient failed (%s). Trying chromadb.Client().", e3)
                    self.client = chromadb.Client()
                    self._ephemeral = True
                    persist_path = "(in-memory)"

        self._persist_path = persist_path
        self.collections: dict[MemoryCategory, chromadb.Collection] = {}
        for cat in MemoryCategory:
            self.collections[cat] = self.client.get_or_create_collection(
                name=cat.value,
                metadata={"hnsw:space": "cosine"},
            )
        mode = "ephemeral" if self._ephemeral else "persistent"
        logger.info("LongTermMemory initialized at %s (%s)", persist_path, mode)

    def store(
        self,
        content: str,
        category: MemoryCategory,
        metadata: dict | None = None,
    ) -> str:
        """Store a memory entry with auto-generated embedding.

        Returns the document ID.
        """
        collection = self.collections[category]
        doc_id = f"{category.value}_{uuid.uuid4().hex[:12]}"
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        collection.add(
            documents=[content],
            metadatas=[meta],
            ids=[doc_id],
        )
        logger.debug("Stored memory [%s]: %s...", category.value, content[:80])
        return doc_id

    def search(
        self,
        query: str,
        category: MemoryCategory | None = None,
        n_results: int = 5,
    ) -> list[dict]:
        """Semantic search across one or all categories.

        Returns list of dicts with content, category, metadata, distance.
        """
        results: list[dict] = []
        targets = [category] if category else list(MemoryCategory)

        for cat in targets:
            collection = self.collections[cat]
            if collection.count() == 0:
                continue
            hits = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count()),
            )
            if not hits or not hits.get("documents"):
                continue
            docs = hits["documents"][0]
            metas = hits["metadatas"][0] if hits.get("metadatas") else [{}] * len(docs)
            dists = hits["distances"][0] if hits.get("distances") else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, dists):
                results.append({
                    "content": doc,
                    "category": cat.value,
                    "metadata": meta,
                    "distance": dist,
                })

        results.sort(key=lambda x: x["distance"])
        return results[:n_results]

    def get_all(self, category: MemoryCategory) -> list[dict]:
        """Retrieve all entries for a category."""
        collection = self.collections[category]
        result = collection.get()
        entries = []
        if result.get("documents"):
            ids = result.get("ids", [])
            metas = result.get("metadatas", [])
            for i, doc in enumerate(result["documents"]):
                entries.append({
                    "id": ids[i] if i < len(ids) else "",
                    "content": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                })
        return entries

    def delete(self, doc_id: str, category: MemoryCategory) -> bool:
        """Delete a memory entry by ID."""
        try:
            self.collections[category].delete(ids=[doc_id])
            return True
        except Exception:
            logger.exception("Failed to delete memory %s", doc_id)
            return False

    def get_stats(self) -> dict[str, int]:
        """Get count of entries per category."""
        return {
            cat.value: self.collections[cat].count()
            for cat in MemoryCategory
        }
