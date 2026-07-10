"""Durable, append-only trace persistence (audit / regulatory record).

Every orchestration run's :class:`~uil.agent.trace.ToolCallTrace` is written as one JSON
object per line (JSONL) to an append-only log. Records are immutable once written and each
carries a unique ``runId`` and a UTC ``recordedUtc`` timestamp, so the sequence of tool calls
that produced a diagnosis is auditable and reproducible after the fact.

Design notes for the audit requirement:
- **Append-only**: records are only ever appended; the store never rewrites prior lines.
- **Tamper-evident**: each record carries ``prevHash`` + ``recordHash`` forming a hash chain
  (SHA-256). Any retroactive edit, deletion, or reordering breaks the chain, detectable via
  :meth:`TraceStore.verify`.
- **Daily rotation**: records are written to ``tool_call_traces-YYYY-MM-DD.jsonl`` (UTC), so the
  log rotates per day; the hash chain is per file (per day).
- **On by default**: persistence happens automatically at the end of every run. It can be
  redirected with the ``UIL_TRACE_DIR`` environment variable and disabled only explicitly with
  ``UIL_TRACE_DISABLE=1`` (used by the test suite so runs don't litter the repo).
- **Handles only**: the trace itself already excludes raw spectra, so the audit log contains
  reference-surface handles + counts + decisions, never bulk RF payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from uil.agent.trace import ToolCallTrace

DEFAULT_TRACE_DIR = "traces"
TRACE_FILE_PREFIX = "tool_call_traces"
GENESIS_HASH = "0" * 64

_ENV_DIR = "UIL_TRACE_DIR"
_ENV_DISABLE = "UIL_TRACE_DISABLE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def persistence_disabled() -> bool:
    return os.getenv(_ENV_DISABLE, "").strip() not in ("", "0", "false", "False")


def default_trace_dir() -> Path:
    return Path(os.getenv(_ENV_DIR, DEFAULT_TRACE_DIR))


class TraceStore:
    """Append-only, hash-chained, daily-rotated JSONL sink for tool-call traces."""

    def __init__(self, directory: Optional[Path | str] = None,
                 filename: Optional[str] = None) -> None:
        self.directory = Path(directory) if directory is not None else default_trace_dir()
        # If a fixed filename is given, rotation is disabled (single file). Otherwise the
        # store rotates daily.
        self._fixed_filename = filename

    def _filename(self) -> str:
        if self._fixed_filename is not None:
            return self._fixed_filename
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"{TRACE_FILE_PREFIX}-{day}.jsonl"

    @property
    def path(self) -> Path:
        """Path to the current (today's) audit log file."""
        return self.directory / self._filename()

    def build_record(self, trace: ToolCallTrace,
                     extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "runId": str(uuid.uuid4()),
            "recordedUtc": _now(),
            "scenario": trace.scenario,
            "finalStatus": trace.final_status,
            "label": trace.label,
            "toolCallCount": len(trace.calls),
            "trace": json.loads(trace.to_jsonl()),
        }
        if extra:
            record.update(extra)
        return record

    @staticmethod
    def _hash(record: dict[str, Any]) -> str:
        """SHA-256 over the canonical record content, excluding the ``recordHash`` field itself."""
        payload = {k: v for k, v in record.items() if k != "recordHash"}
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _last_hash(self, path: Path) -> str:
        if not path.exists():
            return GENESIS_HASH
        last = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = line
        if last is None:
            return GENESIS_HASH
        return json.loads(last).get("recordHash", GENESIS_HASH)

    def record(self, trace: ToolCallTrace,
               extra: Optional[dict[str, Any]] = None) -> Optional[str]:
        """Append one tamper-evident audit record. Returns the ``runId`` (``None`` if disabled)."""
        if persistence_disabled():
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path
        rec = self.build_record(trace, extra)
        rec["prevHash"] = self._last_hash(path)
        rec["recordHash"] = self._hash(rec)
        line = json.dumps(rec, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return rec["runId"]

    def verify(self, path: Optional[Path | str] = None) -> tuple[bool, Optional[int]]:
        """Verify the hash chain of an audit log file.

        Returns ``(True, None)`` if intact (or empty/absent), else ``(False, i)`` where ``i`` is
        the 0-based index of the first record whose link or hash fails.
        """
        p = Path(path) if path is not None else self.path
        if not p.exists():
            return (True, None)
        prev = GENESIS_HASH
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return (False, i)
            if rec.get("prevHash") != prev:
                return (False, i)
            if rec.get("recordHash") != self._hash(rec):
                return (False, i)
            prev = rec["recordHash"]
        return (True, None)
