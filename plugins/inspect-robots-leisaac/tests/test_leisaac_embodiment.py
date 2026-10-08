"""Unit tests for the LeIsaac SO-101 embodiment (no Isaac Sim needed)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import Embodiment
from inspect_robots_leisaac import LeIsaacSO101Embodiment, leisaac_embodiment
from inspect_robots_leisaac.embodiment import JOINT_NAMES


def test_action_space_is_six_joint_targets_in_radians() -> None:
    space = LeIsaacSO101Embodiment().info.action_space
    assert space.shape == (6,)
    assert space.semantics is not None
    assert space.semantics.control_mode == "joint_pos"
    assert space.semantics.gripper == "continuous"
    assert space.semantics.dim_labels == JOINT_NAMES
    assert space.semantics.dim_labels[-1] == "gripper"
    assert space.low is not None and space.high is not None
    np.testing.assert_allclose(space.low[0], np.radians(-110.0))
    np.testing.assert_allclose(space.high[2], np.radians(90.0))
    np.testing.assert_allclose(space.low[5], 0.0)  # closed; the USD's -10 deg only over-squeezes
    np.testing.assert_allclose(space.high[5], np.radians(100.0))


def test_state_has_exactly_one_action_shaped_field() -> None:
    info = LeIsaacSO101Embodiment().info
    assert info.observation_space.state is not None
    fields = info.observation_space.state.fields
    assert [f.key for f in fields] == ["joint_pos"]
    assert [f.shape for f in fields] == [info.action_space.shape]


def test_default_camera_matches_leisaac_front_camera() -> None:
    (camera,) = LeIsaacSO101Embodiment().info.observation_space.cameras
    assert (camera.name, camera.height, camera.width) == ("front", 480, 640)


def test_satisfies_embodiment_protocol_and_defaults() -> None:
    emb = LeIsaacSO101Embodiment()
    assert isinstance(emb, Embodiment)
    assert emb.info.is_simulated
    assert emb.info.control_hz == 60.0
    assert emb.terminated_implies_success is True


def test_factory_forwards_kwargs() -> None:
    emb = leisaac_embodiment(task_id="LeIsaac-SO101-PickOrange-v0", headless=False)
    assert emb.task_id == "LeIsaac-SO101-PickOrange-v0"
    assert emb.headless is False


def test_launcher_enables_cameras_and_only_requests_viewer_when_not_headless() -> None:
    headless = LeIsaacSO101Embodiment()._launcher_kwargs()
    assert headless["enable_cameras"] is True
    assert headless["headless"] is True
    assert "visualizer" not in headless
    assert LeIsaacSO101Embodiment(headless=False)._launcher_kwargs()["visualizer"] == ["kit"]


def test_prepare_env_cfg_selects_joint_position_actions() -> None:
    calls: list[str] = []
    cfg = types.SimpleNamespace(use_teleop_device=calls.append)
    LeIsaacSO101Embodiment()._prepare_env_cfg(cfg)
    assert calls == ["so101leader"]


class _HoldTensor:
    """Minimal tensor: supports ``reshape`` and ``clone`` and remembers its values."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def reshape(self, *shape: int) -> _HoldTensor:
        return self

    def clone(self) -> _HoldTensor:
        return _HoldTensor(list(self.values))


def test_post_reset_holds_the_arm_for_the_settle_steps() -> None:
    held: list[list[float]] = []

    class _Env:
        def step(self, action: _HoldTensor) -> tuple[Any, ...]:
            held.append(action.values)
            return ({"policy": {"joint_pos": _HoldTensor([0.5, 0.5])}}, 0, False, False, {})

    emb = LeIsaacSO101Embodiment(settle_steps=2)
    start = {"policy": {"joint_pos": _HoldTensor([0.1, 0.2])}}

    result = emb._post_reset(_Env(), start)

    assert held == [[0.1, 0.2], [0.5, 0.5]]  # each step holds the latest joint positions
    assert result["policy"]["joint_pos"].values == [0.5, 0.5]
    assert emb._post_reset(_Env(), start) is not start  # still steps again on the next call
    assert LeIsaacSO101Embodiment(settle_steps=0)._post_reset(_Env(), start) is start


def _install_fake_leisaac(
    monkeypatch: pytest.MonkeyPatch, assets: Path, registry: dict[str, Any]
) -> None:
    constant = types.ModuleType("leisaac.utils.constant")
    constant.ASSETS_ROOT = str(assets)  # type: ignore[attr-defined]
    gym = types.ModuleType("gymnasium")
    gym.registry = registry  # type: ignore[attr-defined]
    for name, module in (
        ("leisaac.tasks", types.ModuleType("leisaac.tasks")),
        ("leisaac.utils.constant", constant),
        ("gymnasium", gym),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _make_assets(root: Path) -> Path:
    usd = root / "robots" / "so101_follower.usd"
    usd.parent.mkdir(parents=True)
    usd.write_bytes(b"")
    return root


def test_register_tasks_succeeds_and_exports_assets_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assets = _make_assets(tmp_path / "assets")
    _install_fake_leisaac(monkeypatch, assets, {"LeIsaac-SO101-LiftCube-v0": object()})
    monkeypatch.delenv("LEISAAC_ASSETS_ROOT", raising=False)
    LeIsaacSO101Embodiment(assets_root=str(assets))._register_tasks()
    import os

    assert os.environ["LEISAAC_ASSETS_ROOT"] == str(assets)
    monkeypatch.delenv("LEISAAC_ASSETS_ROOT")


def test_register_tasks_keeps_existing_assets_root_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assets = _make_assets(tmp_path / "assets")
    _install_fake_leisaac(monkeypatch, assets, {"LeIsaac-SO101-LiftCube-v0": object()})
    monkeypatch.setenv("LEISAAC_ASSETS_ROOT", "/already/set")
    LeIsaacSO101Embodiment(assets_root=str(assets))._register_tasks()
    import os

    assert os.environ["LEISAAC_ASSETS_ROOT"] == "/already/set"


def test_register_tasks_reports_unregistered_gym_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_leisaac(monkeypatch, _make_assets(tmp_path / "assets"), {})
    with pytest.raises(RuntimeError, match="not registered"):
        LeIsaacSO101Embodiment()._register_tasks()


def test_register_tasks_reports_missing_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_leisaac(monkeypatch, tmp_path / "empty", {"LeIsaac-SO101-LiftCube-v0": object()})
    with pytest.raises(RuntimeError, match="assets not found"):
        LeIsaacSO101Embodiment()._register_tasks()


def test_register_tasks_reports_missing_leisaac(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "leisaac.tasks", None)
    with pytest.raises(RuntimeError, match="leisaac is not importable"):
        LeIsaacSO101Embodiment()._register_tasks()


def test_close_leaves_the_app_to_isaac_labs_exit_hook() -> None:
    closed: list[bool] = []

    class _App:
        def close(self) -> None:
            closed.append(True)

    emb = LeIsaacSO101Embodiment()
    emb._app = _App()
    emb.close()
    assert closed == []
    assert emb._app is None


class _Tensor:
    """Stands in for a torch tensor or Warp ProxyArray (``.torch`` returns a ``detach``-able)."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array
        self.torch = self

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def _fake_env() -> Any:
    camera = types.SimpleNamespace(
        data=types.SimpleNamespace(
            intrinsic_matrices=_Tensor(
                np.array([[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]])
            ),
            # camera 1 m above and 0.5 m in front of the world origin, identity orientation
            quat_w_ros=_Tensor(np.array([[0.0, 0.0, 0.0, 1.0]])),
            pos_w=_Tensor(np.array([[0.5, 0.0, 1.0]])),
        )
    )
    # robot base at (1, 0, 0), rotated 180 degrees about z (xyzw)
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_quat_w=_Tensor(np.array([[0.0, 0.0, 1.0, 0.0]])),
            root_pos_w=_Tensor(np.array([[1.0, 0.0, 0.0]])),
        )
    )
    scene = {"front": camera, "robot": robot}
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(scene=scene))


def test_observation_extra_reports_camera_calibration_in_the_base_frame() -> None:
    emb = LeIsaacSO101Embodiment(table_z=0.04)
    emb._env = _fake_env()

    extra = emb._observation_extra()

    np.testing.assert_allclose(extra["intrinsics"], [[600, 0, 320], [0, 600, 240], [0, 0, 1]])
    # world camera position (0.5, 0, 1) relative to a base at (1, 0, 0) turned 180 deg about z
    np.testing.assert_allclose(extra["extrinsics"][:3, 3], [0.5, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(extra["extrinsics"][:3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-12)
    assert extra["table_z"] == 0.04
    assert emb._observation_extra() is extra  # cached for the episode


def test_reset_clears_the_cached_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    emb = LeIsaacSO101Embodiment()
    emb._calibration = {"stale": True}
    seen: list[Any] = []

    def fake_reset(self: Any, scene: Any, *, seed: int | None = None) -> str:
        seen.append(self._calibration)
        return "obs"

    from inspect_robots_isaacsim import IsaacSimEmbodiment

    monkeypatch.setattr(IsaacSimEmbodiment, "reset", fake_reset)
    from inspect_robots import Scene

    assert emb.reset(Scene(id="s", instruction="go")) == "obs"  # type: ignore[comparison-overlap]
    assert seen == [None]


def test_embodiment_docs_describe_frames_and_tool_axes() -> None:
    docs = LeIsaacSO101Embodiment(table_z=0.031).info.docs
    assert docs is not None
    assert "z = 0.031 m" in docs
    assert "tool +z" in docs and "tool x" in docs
    assert "'front'" in docs
