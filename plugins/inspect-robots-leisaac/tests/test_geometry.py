"""Tests for the table-plane geometry (pure NumPy)."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_leisaac.geometry import (
    backproject_to_plane,
    fit_box_yaw,
    grasp_position,
    mask_pixels,
    object_center,
    top_down_quaternion_wxyz,
)

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _camera() -> np.ndarray:
    """A camera 0.6 m above and 0.5 m behind the origin, looking forward and down (x forward)."""
    forward = np.array([0.6, 0.0, -0.8])  # optical z in base frame
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    t = np.eye(4)
    t[:3, :3] = np.stack([right, down, forward], axis=1)
    t[:3, 3] = [-0.5, 0.0, 0.6]
    return t


def _project(point: np.ndarray, t: np.ndarray) -> np.ndarray:
    p = np.linalg.inv(t) @ np.append(point, 1.0)
    uv = K @ p[:3]
    projected: np.ndarray = uv[:2] / uv[2]
    return projected


def test_backprojection_inverts_projection_on_the_plane() -> None:
    t = _camera()
    for xy in ([0.1, 0.05], [0.3, -0.1], [0.2, 0.2]):
        point = np.array([*xy, 0.04])
        recovered = backproject_to_plane(_project(point, t), K, t, 0.04)[0]
        np.testing.assert_allclose(recovered, point, atol=1e-9)


def test_backprojection_rejects_parallel_rays_and_planes_behind_the_camera() -> None:
    t = np.eye(4)
    t[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical z = +x
    with pytest.raises(ValueError, match="parallel"):
        backproject_to_plane([320.0, 240.0], K, t, 0.0)
    with pytest.raises(ValueError, match="behind"):
        backproject_to_plane([320.0, 240.0], K, _camera(), 2.0)


def test_mask_pixels_subsamples_and_rejects_empty_masks() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 10:20] = True
    assert mask_pixels(mask).shape == (100, 2)
    assert mask_pixels(mask, max_points=10).shape == (10, 2)
    with pytest.raises(ValueError, match="empty"):
        mask_pixels(np.zeros((4, 4), dtype=bool))


def _render_box(
    center: np.ndarray, size: np.ndarray, t: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    rng = np.random.default_rng(0)
    for p in center + (rng.uniform(-0.5, 0.5, size=(6000, 3)) * size):
        u, v = np.round(_project(p, t)).astype(int)
        if 0 <= v < shape[0] and 0 <= u < shape[1]:
            mask[v, u] = True
    return mask


def test_object_center_recovers_a_cube_within_a_few_millimetres() -> None:
    t = _camera()
    table_z, height = 0.03, 0.03
    center = np.array([0.25, 0.05, table_z + height / 2.0])
    mask = _render_box(center, np.array([0.03, 0.03, 0.03]), t, (480, 640))
    estimate = object_center(mask, K, t, table_z, height)
    assert np.linalg.norm(estimate - center) < 0.006


def _rotation(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.2, np.pi / 2, np.pi, -np.pi / 2])
def test_top_down_quaternion_points_down_with_jaws_closing_along_tool_x(yaw: float) -> None:
    quat = top_down_quaternion_wxyz(yaw)
    assert np.isclose(np.linalg.norm(quat), 1.0)
    rotation = _rotation(quat)
    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-12)  # approach axis down
    np.testing.assert_allclose(rotation[:, 0], [np.cos(yaw), np.sin(yaw), 0.0], atol=1e-12)  # jaws
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)


@pytest.mark.parametrize("yaw", [0.0, 0.9, -2.0])
def test_grasp_position_offsets_the_tool_point_along_tool_x(yaw: float) -> None:
    center = np.array([0.1, -0.3, 0.046])
    tool_x = _rotation(top_down_quaternion_wxyz(yaw))[:, 0]
    tight = grasp_position(center, width=0.03, closing_yaw=yaw, clearance=0.0)
    np.testing.assert_allclose(tight - center, 0.015 * tool_x, atol=1e-12)
    default = grasp_position(center, width=0.03, closing_yaw=yaw)
    np.testing.assert_allclose(default - center, 0.025 * tool_x, atol=1e-12)


def _box_mask(
    center: np.ndarray, yaw: float, size: tuple[float, float, float], t: np.ndarray
) -> np.ndarray:
    """Rasterise the silhouette of an upright box (the model the fit assumes)."""
    w, length, h = size
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    mask = np.zeros((480, 640), dtype=bool)
    rng = np.random.default_rng(3)
    for _ in range(40000):
        u, v, z = rng.uniform(-0.5, 0.5, 3)
        xy = center[:2] + rot @ np.array([u * w, v * length])
        col, row = np.round(_project(np.array([*xy, 0.03 + (z + 0.5) * h]), t)).astype(int)
        mask[row, col] = True
    return mask


@pytest.mark.parametrize("yaw_degrees", [-30.0, -10.0, 0.0, 15.0, 35.0])
def test_fit_box_yaw_recovers_a_squarely_rendered_cube(yaw_degrees: float) -> None:
    t = _camera()
    size = (0.03, 0.03, 0.03)
    mask = _box_mask(np.array([0.25, 0.04, 0.045]), np.radians(yaw_degrees), size, t)
    fit = fit_box_yaw(mask, K, t, 0.03, size)
    error = abs((np.degrees(fit["yaw"]) - yaw_degrees + 45) % 90 - 45)
    assert error < 4.0
    assert fit["iou"] > 0.8
    np.testing.assert_allclose(fit["center"][:2], [0.25, 0.04], atol=0.006)


def test_fit_box_yaw_uses_a_half_turn_for_non_square_footprints() -> None:
    t = _camera()
    size = (0.06, 0.02, 0.03)
    mask = _box_mask(np.array([0.25, 0.0, 0.045]), np.radians(60.0), size, t)
    fit = fit_box_yaw(mask, K, t, 0.03, size)
    error = abs((np.degrees(fit["yaw"]) - 60.0 + 90) % 180 - 90)
    assert error < 6.0
