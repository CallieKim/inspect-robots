"""Helper packs used by the helper-hook tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def pack(get_obs: Callable[[], Mapping[str, Any]]) -> Mapping[str, Callable[..., Any]]:
    """Two helpers: one reads the live ``obs`` view, one is pure."""

    def object_count() -> int:
        return int(get_obs()["objects"])

    def double(value: float) -> float:
        return value * 2

    return {"object_count": object_count, "double": double}


pack.docs = "object_count() -> int\n    Number of objects in view.\ndouble(x) -> float"  # type: ignore[attr-defined]

not_callable = 3
