"""Backend handle store — keeps raw payloads OUT of the SLM-facing surface.

The reference tools return opaque string handles (``measurementRef.measurementId``,
``measurementSetRef``, ``ampListRef``, ``classificationSetRef``). The backend resolves them
to the raw payloads here. This is the mechanism that keeps raw spectra out of SLM context.
"""

from __future__ import annotations

import itertools
from typing import Any


class HandleStore:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._counters: dict[str, itertools.count] = {}

    def _new_id(self, prefix: str) -> str:
        c = self._counters.setdefault(prefix, itertools.count(1))
        return f"{prefix}-{next(c):04d}"

    def put(self, prefix: str, value: Any) -> str:
        handle = self._new_id(prefix)
        self._data[handle] = value
        return handle

    def get(self, handle: str) -> Any:
        if handle not in self._data:
            raise KeyError(handle)
        return self._data[handle]

    def has(self, handle: str) -> bool:
        return handle in self._data
