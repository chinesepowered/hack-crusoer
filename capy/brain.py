"""Shared brain: runbook entries written by the archivist after verified fixes.

Entries are only added after verification passed, so every lesson in the
brain is backed by a fix that demonstrably worked. The brain is rendered
into the lead's system prompt in a stable order, so successive incidents
reuse the same cached prompt prefix on Crusoe's inference fabric.
"""
from __future__ import annotations

import json
from pathlib import Path


class Brain:
    def __init__(self, path: str | Path | None = None, entries: list[dict] | None = None):
        self.path = Path(path) if path else None
        self.entries: list[dict] = list(entries or [])
        if self.path and self.path.exists():
            self.entries = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def add(self, entry: dict):
        sig = entry.get("signature", "").strip().lower()
        if any(e.get("signature", "").strip().lower() == sig for e in self.entries):
            return False
        self.entries.append(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        return True

    def render(self, limit: int = 12) -> str:
        if not self.entries:
            return "(empty: no verified incidents yet)"
        lines = []
        for i, e in enumerate(self.entries[:limit], 1):
            lines.append(f"{i}. [{e.get('root_cause_type', '?')}] {e.get('signature', '')}"
                         f" | confirm: {e.get('confirm_with', '')} | fix: {e.get('fix', '')}"
                         + (f" | pitfall: {e['pitfalls']}" if e.get("pitfalls") else ""))
        return "\n".join(lines)
