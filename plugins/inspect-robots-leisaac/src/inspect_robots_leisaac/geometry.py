"""Table-plane geometry for an RGB camera with known calibration (NumPy only).

Everything is expressed in the robot base frame. Without a depth image, the 3D position of an
object resting on a table follows from its pixel, the camera calibration, and the height of the
plane the pixel's ray should hit: a pixel is a ray, and the object lies where the ray meets that
plane. For a fixed camera and a flat table this reaches millimetre accuracy in simulation.

Camera frames follow the OpenCV/ROS optical convention (x right, y down, z forward).
``extrinsics`` is the 4x4 camera-to-base transform.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def backproject_to_plane(
    pixels: npt.ArrayLike,
    intrinsics: npt.ArrayLike,
    extrinsics: npt.ArrayLike,
    plane_z: float,
) -> Array:
    """Intersect pixel rays with the horizontal plane ``z = plane_z`` (base frame).

    ``pixels`` is ``(N, 2)`` as ``(u, v)`` (a single ``(2,)`` pixel is accepted). Returns ``(N, 3)``
    points. Raises ``ValueError`` if a ray is parallel to the plane or the plane is behind the
    camera.
    """
    uv = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
    k = np.asarray(intrinsics, dtype=np.float64)
    t = np.asarray(extrinsics, dtype=np.float64)
    rays_cam = (np.linalg.inv(k) @ np.vstack([uv.T, np.ones(len(uv))])).T
    rays = rays_cam @ t[:3, :3].T
    origin = t[:3, 3]
    if np.any(np.abs(rays[:, 2]) < 1e-9):
        raise ValueError("a pixel ray is parallel to the table plane")
    scale = (plane_z - origin[2]) / rays[:, 2]
    if np.any(scale <= 0):
        raise ValueError("the table plane is behind the camera for at least one pixel")
    result: Array = origin + scale[:, None] * rays
    return result


def mask_pixels(mask: npt.ArrayLike, max_points: int | None = None) -> Array:
    """Return the ``(u, v)`` pixels of a boolean mask as ``(N, 2)``, optionally subsampled."""
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if len(rows) == 0:
        raise ValueError("mask is empty")
    pixels = np.stack([cols, rows], axis=1).astype(np.float64)
    if max_points is not None and len(pixels) > max_points:
        pixels = pixels[np.linspace(0, len(pixels) - 1, max_points).astype(int)]
    return pixels


def object_center(
    mask: npt.ArrayLike,
    intrinsics: npt.ArrayLike,
    extrinsics: npt.ArrayLike,
    table_z: float,
    object_height: float,
    max_points: int = 4000,
) -> Array:
    """Estimate an upright object's centre from its mask, assuming it rests on the table.

    The mask's pixels are back-projected onto the plane at half the object's height, where an
    upright object's centre lies, and averaged. Against ground truth on 12 rendered cube placements
    the error was 0.7 mm median and 1.0 mm worst. A wrong ``object_height`` shifts the result along
    the viewing ray by roughly ``error * cot(elevation)``. The object's orientation is not
    recoverable from an RGB silhouette, so none is returned.
    """
    plane_z = table_z + object_height / 2.0
    points = backproject_to_plane(mask_pixels(mask, max_points), intrinsics, extrinsics, plane_z)
    center: Array = points.mean(axis=0)
    return center


def _convex_hull(points: Array) -> Array:
    """Counter-clockwise convex hull of 2-D points (Andrew's monotone chain)."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def half(sequence: Array) -> list[Array]:
        chain: list[Array] = []
        for q in sequence:
            while len(chain) >= 2:
                (ax, ay), (bx, by) = chain[-2], chain[-1]
                if (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax) > 0:
                    break
                chain.pop()
            chain.append(q)
        return chain

    return np.array(half(pts)[:-1] + half(pts[::-1])[:-1])


def _inside_convex(polygon: Array, points: Array) -> npt.NDArray[np.bool_]:
    """Whether each point lies inside a counter-clockwise convex polygon."""
    inside = np.ones(len(points), dtype=bool)
    for i in range(len(polygon)):
        (ax, ay), (bx, by) = polygon[i], polygon[(i + 1) % len(polygon)]
        inside &= (bx - ax) * (points[:, 1] - ay) - (by - ay) * (points[:, 0] - ax) >= -1e-9
    return inside


def fit_box_yaw(
    mask: npt.ArrayLike,
    intrinsics: npt.ArrayLike,
    extrinsics: npt.ArrayLike,
    table_z: float,
    size: tuple[float, float, float],
    step_degrees: float = 1.0,
) -> dict[str, Any]:
    """Estimate the yaw of an upright box of known ``size`` from its mask.

    A silhouette alone does not reveal a box's yaw (its outline is a hexagon whose shape changes
    only slowly), so each candidate yaw is rendered: the box's eight corners are projected into the
    image at the mask-derived centre and the convex hull is compared with the mask by
    intersection-over-union. ``size`` is ``(x, y, z)`` extents in the box's own frame. Returns the
    best ``yaw`` in radians, wrapped to the box's symmetry (a half turn, or a quarter turn for a
    square footprint), its ``iou``, and the ``center`` used. On 12 rendered 3 cm cube placements the
    error was 6 degrees median and 15 degrees worst.
    """
    mask_array = np.asarray(mask, dtype=bool)
    k = np.asarray(intrinsics, dtype=np.float64)
    t = np.asarray(extrinsics, dtype=np.float64)
    width, length, height = size
    center = object_center(mask_array, k, t, table_z, height)
    rows, cols = np.nonzero(mask_array)
    margin = 6
    x0, x1 = cols.min() - margin, cols.max() + margin + 1
    y0, y1 = rows.min() - margin, rows.max() + margin + 1
    grid_x, grid_y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
    inside_mask = mask_array[max(y0, 0) : y1, max(x0, 0) : x1]
    padded = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    padded[
        max(y0, 0) - y0 : max(y0, 0) - y0 + inside_mask.shape[0],
        max(x0, 0) - x0 : max(x0, 0) - x0 + inside_mask.shape[1],
    ] = inside_mask
    target = padded.ravel()
    half_turn = np.pi / 2 if np.isclose(width, length) else np.pi
    footprint = np.array(
        [[sx * width / 2, sy * length / 2] for sx in (-1, 1) for sy in (-1, 1)], dtype=np.float64
    )
    best_iou, best_yaw = -1.0, 0.0
    for yaw in np.deg2rad(np.arange(-45.0, 135.0, step_degrees)):
        if yaw > -np.pi / 4 + half_turn:
            break
        c, s = np.cos(yaw), np.sin(yaw)
        xy = footprint @ np.array([[c, s], [-s, c]]) + center[:2]
        corners = np.array([[x, y, z] for x, y in xy for z in (table_z, table_z + height)])
        camera = (np.linalg.inv(t) @ np.vstack([corners.T, np.ones(len(corners))]))[:3]
        pixels = (k @ camera)[:2] / (k @ camera)[2]
        polygon = _convex_hull(pixels.T)
        silhouette = _inside_convex(polygon, grid)
        union = int(np.count_nonzero(silhouette | target))
        iou = np.count_nonzero(silhouette & target) / max(union, 1)
        if iou > best_iou:
            best_iou, best_yaw = float(iou), float(yaw)
    return {"yaw": best_yaw, "iou": best_iou, "center": center}


def quat_xyzw_to_matrix(q: npt.ArrayLike) -> Array:
    """Rotation matrix for a unit quaternion given as ``(x, y, z, w)`` (Isaac Lab 3.0 order)."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def top_down_quaternion_wxyz(closing_yaw: float = 0.0) -> Array:
    """Tool orientation pointing straight down, as a unit quaternion ``(w, x, y, z)``.

    The SO-101's approach axis is tool +z. Its jaws close along tool x: the fixed jaw's inner
    surface is at tool x = 0 (the tool point) and the moving jaw swings in on the -x side.
    ``closing_yaw`` is the angle from base +x, counter-clockwise seen from above, of tool +x, the
    direction from the moving jaw towards the fixed jaw.
    """
    c, s = np.cos(closing_yaw), np.sin(closing_yaw)
    tool_x = np.array([c, s, 0.0])
    tool_z = np.array([0.0, 0.0, -1.0])
    tool_y = np.cross(tool_z, tool_x)
    return _matrix_to_wxyz(np.stack([tool_x, tool_y, tool_z], axis=1))


def grasp_position(
    center: npt.ArrayLike, width: float, closing_yaw: float = 0.0, clearance: float = 0.01
) -> Array:
    """Tool-point position for grasping an object of ``width`` between the jaws.

    The tool point lies on the fixed jaw's inner surface, so the object's centre sits half its
    width away on the moving-jaw side: ``center + (width / 2 + clearance) * tool_x``. ``width`` is
    the object's extent along the closing direction, ``closing_yaw`` as in
    :func:`top_down_quaternion_wxyz`. The ``clearance`` keeps the fixed jaw off the object while
    descending (a corner turned by a small yaw error would otherwise be hit and pushed away);
    closing then slides the object across the gap against the fixed jaw.
    """
    direction = np.array([np.cos(closing_yaw), np.sin(closing_yaw), 0.0])
    position: Array = np.asarray(center, dtype=np.float64) + (0.5 * width + clearance) * direction
    return position


def _matrix_to_wxyz(r: Array) -> Array:
    trace = np.trace(r)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array(
            [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = np.zeros(4)
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    out: Array = q / np.linalg.norm(q)
    return out
