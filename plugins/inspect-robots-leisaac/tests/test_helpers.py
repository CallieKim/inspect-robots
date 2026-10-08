"""Tests for the table-top helper pack."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inspect_robots_leisaac.helpers import table_top_helpers

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _obs() -> dict[str, Any]:
    forward = np.array([0.6, 0.0, -0.8])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    t = np.eye(4)
    t[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    t[:3, 3] = [-0.5, 0.0, 0.6]
    return {"intrinsics": K, "extrinsics": t, "table_z": 0.03}


def _mask_at(point: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
    p = np.linalg.inv(obs["extrinsics"]) @ np.append(point, 1.0)
    u, v = (K @ p[:3])[:2] / p[2]
    mask = np.zeros((480, 640), dtype=bool)
    row, col = round(v), round(u)
    mask[row - 2 : row + 3, col - 2 : col + 3] = True
    return mask


def test_object_center_uses_the_observation_calibration() -> None:
    obs = _obs()
    helpers = table_top_helpers(lambda: obs)
    centre = np.array([0.25, 0.05, 0.03 + 0.015])
    estimate = helpers["object_center"](_mask_at(centre, obs))
    assert np.linalg.norm(estimate - centre) < 0.003


def test_box_yaw_is_exposed_and_returns_radians() -> None:
    obs = _obs()
    helpers = table_top_helpers(lambda: obs)
    yaw = helpers["box_yaw"](_mask_at(np.array([0.25, 0.05, 0.045]), obs))
    assert isinstance(yaw, float)
    assert -np.pi / 4 <= yaw <= np.pi / 4


def test_grasp_position_is_exposed() -> None:
    helpers = table_top_helpers(lambda: _obs())
    position = helpers["grasp_position"]([0.0, -0.3, 0.046], 0.03, 0.0)
    np.testing.assert_allclose(position, [0.025, -0.3, 0.046])


def test_top_down_quat_is_exposed() -> None:
    helpers = table_top_helpers(lambda: _obs())
    quat = helpers["top_down_quat"](0.5)
    assert quat.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0)


def test_missing_calibration_names_the_missing_keys() -> None:
    helpers = table_top_helpers(lambda: {"intrinsics": K})
    with pytest.raises(KeyError, match=r"extrinsics.*table_z"):
        helpers["object_center"](np.ones((4, 4), dtype=bool))


def test_pack_carries_prompt_docs_for_every_helper() -> None:
    docs = table_top_helpers.docs  # type: ignore[attr-defined]
    for name in ("object_center", "box_yaw", "top_down_quat", "grasp_position", "plan_grasp"):
        assert name in docs
