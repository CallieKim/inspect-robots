"""LeIsaac SO-101 embodiment: the Isaac Lab adapter specialised for the SO-101 arm.

The SO-101 has five arm joints and a continuous gripper joint (six in total).
LeIsaac maps an absolute 6-D joint-position action onto them when its task cfg is
set up for the ``so101leader`` device, so this embodiment asks for that mode and
exposes the action as ``joint_pos`` in radians, bounded by the robot's USD joint
limits. Proprioception is the six joint positions only, which keeps exactly one
state field with the action-vector shape (what code-as-policy planners expect).
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from inspect_robots import (
    ActionSemantics,
    Box,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    Scene,
    StateField,
    StateSpec,
)
from inspect_robots.spaces import CANONICAL_STATE_UNITS
from inspect_robots_isaacsim import IsaacSimEmbodiment
from inspect_robots_leisaac.geometry import quat_xyzw_to_matrix

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
"""Action dimension order; matches LeIsaac's ``SINGLE_ARM_JOINT_NAMES``."""

JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}
"""Joint limits written in the LeIsaac SO-101 USD, in degrees (the gripper's low end is raised to
:data:`GRIPPER_CLOSED_RAD` in the action space)."""

_ROBOT_USD = Path("robots") / "so101_follower.usd"

DEFAULT_TABLE_Z = 0.031
"""Table-top height (base frame) of LeIsaac's table scenes, measured from a resting cube."""

_DOCS = """\
Simulated SO-101 arm (5 joints plus a gripper) on a table. Joint positions are in radians in the
order shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper. At all zeros the
arm is stretched out horizontally away from its base along base -y. The gripper is open when its
value is high and closed when it is low.

All poses are in the robot base frame: z is up and the table surface is at z = {table_z:.3f} m.
Objects start on the table in front of the arm, around |x| < 0.1 m and -0.35 m < y < -0.2 m. Targets
for solve_ik are poses of the tool point on the fixed jaw's inner surface: the approach axis is
tool +z, the jaws close along tool x, and the moving jaw is on the tool -x side. The arm has five
joints, so it cannot reach arbitrary tool orientations; a straight-down tool at any yaw is
reachable, so prefer top-down grasps.

The camera '{camera}' is fixed relative to the base and the table is flat. Lifting the cube means
raising it about 15 cm clear of the table; the episode ends when it is lifted. A grasp that closes
on nothing leaves the gripper at 0; a grasp that holds an object stops at its width."""


GRIPPER_CLOSED_RAD = 0.0
"""Lowest gripper target. The USD allows -10 degrees, but the real motor's calibrated range
starts at 0 (closed). Commanding past 0 only squeezes harder: with a cube between the jaws, a
target of -0.17 rad ejected it far more often than a target of 0.0 in simulation."""


def _action_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return per-dimension ``(low, high)`` joint targets in radians."""
    limits = np.radians(np.array([JOINT_LIMITS_DEG[name] for name in JOINT_NAMES]))
    low, high = limits[:, 0].copy(), limits[:, 1]
    low[JOINT_NAMES.index("gripper")] = GRIPPER_CLOSED_RAD
    return low, high


class LeIsaacSO101Embodiment(IsaacSimEmbodiment):
    """A simulated SO-101 arm in a LeIsaac task, driven by absolute joint targets.

    Parameters
    ----------
    task_id:
        A registered LeIsaac gym id. The default lifts a cube.
    assets_root:
        Directory holding LeIsaac's ``robots/`` and ``scenes/`` USD assets. Sets
        ``LEISAAC_ASSETS_ROOT`` unless that variable is already set. Without
        either, LeIsaac looks for ``assets/`` under the current directory's git
        root, which is usually wrong when running outside the leisaac checkout.
    teleop_device:
        Device name passed to the task cfg's ``use_teleop_device``. ``so101leader``
        selects absolute joint-position actions for all six joints.
    table_z:
        Height of the table surface in the robot base frame, published as ``obs["table_z"]``
        for planners that locate objects on the table plane. The default is measured from a
        cube resting in LeIsaac's table scene.
    settle_steps:
        Physics steps with the arm held in place right after a reset, so camera frames show the
        randomized scene. The frame straight after ``env.reset`` lags it and shows the previous
        placement of the objects. Set 0 to report the raw post-reset observation.
    cameras:
        Camera streams as ``(name, height, width[, channels])``. LeIsaac's
        LiftCube task renders only ``front``.
    headless:
        When False, Isaac Lab opens a Kit viewport. Isaac Lab 3.0 runs headless
        unless a Kit visualizer is requested, so this also requests one.
    """

    def __init__(
        self,
        task_id: str = "LeIsaac-SO101-LiftCube-v0",
        *,
        assets_root: str | None = None,
        teleop_device: str = "so101leader",
        table_z: float = DEFAULT_TABLE_Z,
        settle_steps: int = 2,
        cameras: Sequence[tuple[str, int, int] | tuple[str, int, int, int]] = (
            ("front", 480, 640, 3),
        ),
        control_hz: float = 60.0,
        headless: bool = True,
        device: str = "cuda:0",
        terminated_implies_success: bool = True,
        name: str = "leisaac-so101",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task_id,
            num_arm_joints=len(JOINT_NAMES) - 1,
            cameras=cameras,
            control_hz=control_hz,
            headless=headless,
            device=device,
            terminated_implies_success=terminated_implies_success,
            name=name,
            **kwargs,
        )
        self.assets_root = assets_root
        self.teleop_device = teleop_device
        self.table_z = table_z
        self.settle_steps = settle_steps
        self._calibration: dict[str, Any] | None = None

        low, high = _action_bounds()
        self.info = EmbodimentInfo(
            name=name,
            action_space=Box(
                shape=(len(JOINT_NAMES),),
                low=low,
                high=high,
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    rotation_repr="none",
                    gripper="continuous",
                    frame="base",
                    dim_labels=JOINT_NAMES,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=self.info.observation_space.cameras,
                state=StateSpec(
                    fields=(
                        StateField(
                            "joint_pos",
                            (len(JOINT_NAMES),),
                            CANONICAL_STATE_UNITS.get("joint_pos", ""),
                        ),
                    )
                ),
            ),
            control_hz=control_hz,
            is_simulated=True,
            capabilities=self.info.capabilities,
            supported_setups=self.info.supported_setups,
            supported_target_kinds=self.info.supported_target_kinds,
            docs=_DOCS.format(table_z=table_z, camera=self.info.observation_space.cameras[0].name),
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset the task; calibration is re-read because the task may jitter the camera pose."""
        self._calibration = None
        return super().reset(scene, seed=seed)

    def _post_reset(self, env: Any, obs: Any) -> Any:
        for _ in range(self.settle_steps):
            hold = obs["policy"]["joint_pos"].reshape(1, -1).clone()
            obs, *_ = env.step(hold)
        return obs

    def _observation_extra(self) -> dict[str, Any]:
        """Camera ``intrinsics`` (3x3), ``extrinsics`` (4x4 camera to base) and ``table_z``.

        Read from the simulator once per episode and cached, so recorded trials hold plain arrays
        rather than handles into a live simulation. The camera is the first declared camera.
        """
        if self._calibration is None:
            self._calibration = {**self._read_calibration(), "table_z": self.table_z}
        return self._calibration

    def _read_calibration(self) -> dict[str, Any]:
        assert self._env is not None  # observations only exist after the env is built
        scene = self._env.unwrapped.scene
        camera = scene[self.info.observation_space.cameras[0].name]
        robot = scene["robot"]
        k = _to_numpy(camera.data.intrinsic_matrices)[0]
        cam_rotation = quat_xyzw_to_matrix(_to_numpy(camera.data.quat_w_ros)[0])
        cam_position = _to_numpy(camera.data.pos_w)[0]
        root_rotation = quat_xyzw_to_matrix(_to_numpy(robot.data.root_quat_w)[0])
        root_position = _to_numpy(robot.data.root_pos_w)[0]
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = root_rotation.T @ cam_rotation
        extrinsics[:3, 3] = root_rotation.T @ (cam_position - root_position)
        return {"intrinsics": k, "extrinsics": extrinsics}

    def _launcher_kwargs(self) -> dict[str, Any]:
        kwargs = super()._launcher_kwargs()
        if not self.headless:
            kwargs["visualizer"] = ["kit"]
        return kwargs

    def _register_tasks(self) -> None:
        if self.assets_root is not None:
            os.environ.setdefault("LEISAAC_ASSETS_ROOT", str(Path(self.assets_root).expanduser()))
        try:
            importlib.import_module("leisaac.tasks")
            constant = importlib.import_module("leisaac.utils.constant")
            gym = importlib.import_module("gymnasium")
        except ImportError as exc:
            raise RuntimeError(
                f"leisaac is not importable ({exc}). Install it into the Isaac Lab "
                "environment, e.g. `pip install -e source/leisaac` from the leisaac checkout."
            ) from exc
        # leisaac's package init swallows ImportErrors, so a broken task import
        # surfaces here as a missing gym id instead of an exception.
        if self.task_id not in gym.registry:
            raise RuntimeError(
                f"gym id {self.task_id!r} is not registered after importing leisaac.tasks. "
                "Import leisaac.tasks in a Python shell to see the import error."
            )
        robot_usd = Path(constant.ASSETS_ROOT) / _ROBOT_USD
        if not robot_usd.is_file():
            raise RuntimeError(
                f"LeIsaac assets not found: {robot_usd} does not exist. Pass "
                "assets_root=/path/to/leisaac/assets or set LEISAAC_ASSETS_ROOT."
            )

    def _close_app(self, app: Any) -> bool:
        # Kit's fast shutdown ends the process from inside close(), which would cut off
        # eval()'s own log writing and reporting. Isaac Lab 3.0 registers an atexit
        # shutdown for the app, so leave it running and let the process exit close it.
        return False

    def _prepare_env_cfg(self, env_cfg: Any) -> None:
        env_cfg.use_teleop_device(self.teleop_device)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a Warp-backed ``ProxyArray`` or torch tensor to a float64 NumPy array."""
    tensor = value.torch if hasattr(value, "torch") else value
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)
