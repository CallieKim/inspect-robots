"""inspect-robots-leisaac: a LeIsaac SO-101 embodiment for Inspect Robots.

With a working Isaac Lab environment and the ``leisaac`` package installed, the
``leisaac`` embodiment drives a simulated SO-101 arm::

    inspect-robots "lift the cube" --policy agent --embodiment leisaac

The embodiment is discovered through the ``inspect_robots.embodiments`` entry point.
"""

from __future__ import annotations

from typing import Any

from inspect_robots_leisaac.embodiment import LeIsaacSO101Embodiment

__all__ = ["LeIsaacSO101Embodiment", "leisaac_embodiment"]

__version__ = "0.1.0"


def leisaac_embodiment(**kwargs: Any) -> LeIsaacSO101Embodiment:
    """Factory the Inspect Robots registry calls (entry point ``leisaac``).

    Accepts the keyword arguments of :class:`LeIsaacSO101Embodiment`; the CLI
    forwards ``-E key=value`` pairs here.
    """
    return LeIsaacSO101Embodiment(**kwargs)
