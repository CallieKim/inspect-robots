"""Extra code-as-policy helpers for an RGB-only tabletop camera (no depth).

Bind them into the ``capx`` policy with
``-P helpers=inspect_robots_leisaac.helpers:table_top_helpers``.
They read ``intrinsics``, ``extrinsics`` and ``table_z`` from the observation, which the
``leisaac`` embodiment provides, and locate objects by intersecting mask rays with the table
plane (see :mod:`inspect_robots_leisaac.geometry`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots_leisaac import geometry

DEFAULT_OBJECT_HEIGHT = 0.03
"""Assumed object height in metres when the caller gives none (a 3 cm cube)."""


def table_top_helpers(
    get_obs: Callable[[], Mapping[str, Any]],
) -> dict[str, Callable[..., Any]]:
    """Return the table-top helpers bound to the current turn's observation."""

    def calibration() -> tuple[npt.NDArray[Any], npt.NDArray[Any], float]:
        obs = get_obs()
        missing = [key for key in ("intrinsics", "extrinsics", "table_z") if key not in obs]
        if missing:
            raise KeyError(f"observation is missing {missing}; the embodiment must provide them")
        return np.asarray(obs["intrinsics"]), np.asarray(obs["extrinsics"]), float(obs["table_z"])

    def object_center(
        mask: npt.ArrayLike, height: float = DEFAULT_OBJECT_HEIGHT
    ) -> npt.NDArray[Any]:
        k, t, table_z = calibration()
        return geometry.object_center(mask, k, t, table_z, height)

    def box_yaw(
        mask: npt.ArrayLike, size: tuple[float, float, float] = (0.03, 0.03, 0.03)
    ) -> float:
        k, t, table_z = calibration()
        return float(geometry.fit_box_yaw(mask, k, t, table_z, size)["yaw"])

    def top_down_quat(closing_yaw: float = 0.0) -> npt.NDArray[Any]:
        return geometry.top_down_quaternion_wxyz(closing_yaw)

    def grasp_position(
        center: npt.ArrayLike, width: float, closing_yaw: float = 0.0, clearance: float = 0.01
    ) -> npt.NDArray[Any]:
        return geometry.grasp_position(center, width, closing_yaw, clearance)

    return {
        "object_center": object_center,
        "box_yaw": box_yaw,
        "top_down_quat": top_down_quat,
        "grasp_position": grasp_position,
    }


TABLE_TOP_DOCS = """\
object_center(mask: np.ndarray, height: float = 0.03) -> np.ndarray
    Position (3,) of an upright object's centre in the robot-base frame, from a
    `segment(...)` mask, assuming it rests on the table. `height` is the object's
    height in metres. There is no depth image; the mask's pixels are intersected
    with the table plane. Accurate to about a millimetre in simulation. The
    object's orientation cannot be recovered from an RGB mask, so none is given.
box_yaw(mask: np.ndarray, size=(0.03, 0.03, 0.03)) -> float
    Yaw in radians (from base +x, counter-clockwise from above) of an upright box of
    known (x, y, z) `size` in metres, fitted to the mask by comparing rendered
    silhouettes. Only meaningful for box-shaped objects of about that size. Pass it
    as `closing_yaw` so the jaws close squarely on a face. Accurate to roughly 6
    degrees (worst case about 15) in simulation.
top_down_quat(closing_yaw: float = 0.0) -> np.ndarray
    Quaternion (w, x, y, z) for a tool pointing straight down. The jaws close along
    tool x: the fixed jaw's inner surface is at the tool point and the moving jaw is
    on the tool -x side. `closing_yaw` is the angle of tool +x from base +x
    (counter-clockwise seen from above).
grasp_position(center, width: float, closing_yaw: float = 0.0,
               clearance: float = 0.01) -> np.ndarray
    Tool-point position for grasping an object of `width` (its extent along the
    closing direction, metres): center + (width/2 + clearance) * tool +x. The
    clearance keeps the fixed jaw off the object while descending; closing then
    slides the object against it. Use it for the grasp, and the same x, y with a
    larger z for the approach. Open the gripper before descending; the opening
    grows with the gripper value (about 16 mm at 0.0, 43 mm at 0.4, 73 mm at 0.8).
Observation entries `obs["intrinsics"]` (3x3), `obs["extrinsics"]` (4x4 camera to
robot base, optical convention) and `obs["table_z"]` (table height, metres) are
provided; `obs["depth"]` is not, so `plan_grasp` is unavailable."""

table_top_helpers.docs = TABLE_TOP_DOCS  # type: ignore[attr-defined]
