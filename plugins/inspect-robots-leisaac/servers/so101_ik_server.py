"""Pyroki inverse-kinematics server for the SO-101, speaking the CaP-X ``/ik`` protocol.

Run it in an environment with ``pyroki`` installed (not the Isaac Lab environment)::

    python so101_ik_server.py --port 8116
    python so101_ik_server.py --selftest

``POST /ik`` takes ``{"target_pose_wxyz_xyz": [qw, qx, qy, qz, x, y, z], "prev_cfg": [...] | null}``
and returns ``{"joint_positions": [...6 joints...], ...}`` in the embodiment's action order
(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper). The optional request
keys ``pos_weight`` and ``ori_weight`` override the cost weights (``ori_weight=0`` gives a
position-only solve) and ``relax=false`` disables the orientation relaxation described below.
The response also carries ``position_error_mm`` and ``orientation_error_deg``.

The SO-101 has five arm joints, so a pose is only reachable when its orientation is consistent
with the arm. Position is weighted far above orientation, which makes top-down grasps (tool
pointing down, any yaw) solvable up to about 0.30 m from the base. Farther targets are solved with
the smallest tilt that still reaches the position, and the response reports the orientation error.

Poses are expressed in the simulated robot's base frame. ``--frame urdf`` skips the conversion
from the LeIsaac USD base frame to the URDF base frame, for use with a real robot described
directly by this URDF.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

EMBODIMENT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
TCP_LINK = "gripper_frame_link"
"""Tool centre point: the URDF frame between the fingertips."""

# LeIsaac USD base frame = R @ URDF base frame + T. Fitted over every link and six joint
# configurations: mean residual 0.33 mm, max 1.3 mm.
SIM_FROM_URDF_R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
SIM_FROM_URDF_T = np.array([0.02084, 0.01579, 0.03236])

READY_CFG = np.array([0.0, -0.5, 0.8, 0.5, 0.0, 0.3])
"""A comfortable pose above the table, used as one of the solver's starting points."""

RELAX_THRESHOLD_MM = 1.5
"""Position error above which the orientation weight is lowered to reach the target."""

RELAXED_ORI_WEIGHTS = (3.0, 1.0, 0.3, 0.1)

RANDOM_RESTARTS = 12
"""Extra random starting configurations tried when the named starts end in a local minimum."""

DEFAULT_POS_WEIGHT = 100.0
DEFAULT_ORI_WEIGHT = 10.0

log = logging.getLogger("so101_ik")


@jdc.jit
def _solve(
    robot: pk.Robot,
    link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    init_cfg: jax.Array,
    pos_weight: jax.Array,
    ori_weight: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
            link_index,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(init_cfg)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return solution[joint_var]


class So101Solver:
    """Solve poses for the SO-101 TCP; owns the Pyroki robot and the joint-order maps."""

    def __init__(self, urdf_path: Path, *, frame: str = "sim") -> None:
        urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False)
        self.robot = pk.Robot.from_urdf(urdf)
        self.link_index = self.robot.links.names.index(TCP_LINK)
        actuated = list(self.robot.joints.actuated_names)
        # to_pyroki[i] = index in the embodiment order of Pyroki's i-th joint
        self.to_pyroki = np.array([EMBODIMENT_ORDER.index(name) for name in actuated])
        self.from_pyroki = np.array([actuated.index(name) for name in EMBODIMENT_ORDER])
        self.frame = frame
        self.lower = np.array(self.robot.joints.lower_limits)[self.from_pyroki]
        self.upper = np.array(self.robot.joints.upper_limits)[self.from_pyroki]
        self._rng = np.random.default_rng(0)

    def _to_urdf_frame(self, wxyz: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.frame == "urdf":
            return wxyz, xyz
        rotation = jaxlie.SO3(jnp.asarray(wxyz)).as_matrix()
        urdf_rotation = SIM_FROM_URDF_R.T @ np.asarray(rotation)
        return np.asarray(
            jaxlie.SO3.from_matrix(jnp.asarray(urdf_rotation)).wxyz
        ), SIM_FROM_URDF_R.T @ (xyz - SIM_FROM_URDF_T)

    def tcp_pose(self, cfg_embodiment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: TCP pose ``(wxyz, xyz)`` in the server frame for a 6-joint config."""
        poses = np.asarray(
            self.robot.forward_kinematics(jnp.asarray(cfg_embodiment[self.to_pyroki]))
        )
        wxyz, xyz = poses[self.link_index][:4], poses[self.link_index][4:]
        if self.frame == "urdf":
            return wxyz, xyz
        rotation = SIM_FROM_URDF_R @ np.asarray(jaxlie.SO3(jnp.asarray(wxyz)).as_matrix())
        return np.asarray(
            jaxlie.SO3.from_matrix(jnp.asarray(rotation)).wxyz
        ), SIM_FROM_URDF_R @ xyz + SIM_FROM_URDF_T

    def solve(
        self,
        wxyz: np.ndarray,
        xyz: np.ndarray,
        prev_cfg: np.ndarray | None = None,
        *,
        pos_weight: float = DEFAULT_POS_WEIGHT,
        ori_weight: float = DEFAULT_ORI_WEIGHT,
        relax: bool = True,
    ) -> dict[str, Any]:
        """Solve a pose, relaxing the orientation just enough to reach the position.

        A five-joint arm cannot point straight down at distant targets. When the requested
        orientation leaves the position more than ``RELAX_THRESHOLD_MM`` off, the orientation
        weight is lowered step by step until the position is met, so the tool tilts as little as
        needed. The response says so through ``relaxed_ori_weight``.
        """
        best = self._solve_weights(
            wxyz, xyz, prev_cfg, pos_weight=pos_weight, ori_weight=ori_weight
        )
        if not relax or ori_weight <= 0 or best["position_error_mm"] <= RELAX_THRESHOLD_MM:
            return best
        for weight in RELAXED_ORI_WEIGHTS:
            if weight >= ori_weight:
                continue
            attempt = self._solve_weights(
                wxyz, xyz, prev_cfg, pos_weight=pos_weight, ori_weight=weight
            )
            attempt["relaxed_ori_weight"] = weight
            if attempt["position_error_mm"] < best["position_error_mm"]:
                best = attempt
            if best["position_error_mm"] <= RELAX_THRESHOLD_MM:
                break
        return best

    def _solve_weights(
        self,
        wxyz: np.ndarray,
        xyz: np.ndarray,
        prev_cfg: np.ndarray | None = None,
        *,
        pos_weight: float,
        ori_weight: float,
    ) -> dict[str, Any]:
        """Return the best of several starting points, in the embodiment's joint order."""
        urdf_wxyz, urdf_xyz = self._to_urdf_frame(wxyz, xyz)
        starts = [READY_CFG, np.zeros(6)]
        if prev_cfg is not None:
            starts.insert(0, np.clip(prev_cfg, self.lower, self.upper))
        starts += [self._rng.uniform(self.lower, self.upper) for _ in range(RANDOM_RESTARTS)]
        best: dict[str, Any] | None = None
        for start in starts:
            cfg_pyroki = _solve(
                self.robot,
                jnp.asarray(self.link_index),
                jnp.asarray(urdf_wxyz),
                jnp.asarray(urdf_xyz),
                jnp.asarray(start[self.to_pyroki]),
                jnp.asarray(pos_weight),
                jnp.asarray(ori_weight),
            )
            cfg = np.asarray(cfg_pyroki)[self.from_pyroki]
            result = self._score(cfg, wxyz, xyz)
            if best is None or result["_rank"] < best["_rank"]:
                best = result
            if result["position_error_mm"] < 1.0 and result["orientation_error_deg"] < 2.0:
                break
        assert best is not None
        best.pop("_rank")
        return best

    def _score(self, cfg: np.ndarray, wxyz: np.ndarray, xyz: np.ndarray) -> dict[str, Any]:
        got_wxyz, got_xyz = self.tcp_pose(cfg)
        position_error = float(np.linalg.norm(got_xyz - xyz) * 1000.0)
        relative = jaxlie.SO3(jnp.asarray(got_wxyz)).inverse() @ jaxlie.SO3(jnp.asarray(wxyz))
        orientation_error = float(np.degrees(np.linalg.norm(np.asarray(relative.log()))))
        return {
            "joint_positions": cfg.tolist(),
            "position_error_mm": position_error,
            "orientation_error_deg": orientation_error,
            "_rank": position_error + 0.5 * orientation_error,
        }


def selftest(solver: So101Solver) -> int:
    """Round-trip random reachable configs through FK and IK; return a process exit code."""
    rng = np.random.default_rng(0)
    pos_errors, ori_errors, times = [], [], []
    for _ in range(40):
        cfg = rng.uniform(solver.lower, solver.upper) * 0.8
        cfg[5] = 0.3
        wxyz, xyz = solver.tcp_pose(cfg)
        start = time.perf_counter()
        result = solver.solve(wxyz, xyz, prev_cfg=None)
        times.append(time.perf_counter() - start)
        pos_errors.append(result["position_error_mm"])
        ori_errors.append(result["orientation_error_deg"])
    failures = sum(e > 2.0 for e in pos_errors)
    print(
        f"reachable round trips: {failures}/{len(pos_errors)} above 2 mm; "
        f"position error median {np.median(pos_errors):.3f} mm, max {max(pos_errors):.3f} mm; "
        f"orientation max {max(ori_errors):.2f} deg; "
        f"solve time median {np.median(times) * 1000:.0f} ms"
    )
    # A top-down pose with the tool pointing straight down at a reachable point.
    down = jaxlie.SO3.from_rpy_radians(np.pi, 0.0, 0.0).wxyz
    top = solver.solve(np.asarray(down), np.array([0.0, -0.22, 0.05]))
    print(
        f"top-down at (0, -0.22, 0.05): position error {top['position_error_mm']:.2f} mm, "
        f"orientation error {top['orientation_error_deg']:.2f} deg"
    )
    ok = failures == 0 and max(ori_errors) < 5.0
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def make_handler(solver: So101Solver) -> type[BaseHTTPRequestHandler]:
    """Build the request handler class that serves ``solver`` over HTTP."""

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._reply(200, {"status": "ok", "tcp_link": TCP_LINK, "frame": solver.frame})

        def do_POST(self) -> None:
            if self.path != "/ik":
                self._reply(404, {"error": f"unknown path {self.path}"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                pose = np.asarray(body["target_pose_wxyz_xyz"], dtype=float)
                if pose.shape != (7,):
                    raise ValueError("target_pose_wxyz_xyz must have 7 numbers: qw qx qy qz x y z")
                prev = body.get("prev_cfg")
                result = solver.solve(
                    pose[:4] / np.linalg.norm(pose[:4]),
                    pose[4:],
                    None if prev is None else np.asarray(prev, dtype=float)[:6],
                    pos_weight=float(body.get("pos_weight", DEFAULT_POS_WEIGHT)),
                    ori_weight=float(body.get("ori_weight", DEFAULT_ORI_WEIGHT)),
                    relax=bool(body.get("relax", True)),
                )
                self._reply(200, result)
            except (KeyError, ValueError, TypeError) as exc:
                self._reply(400, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            log.info(fmt, *args)

    return Handler


def main() -> int:
    """Parse arguments, then run the self-test or serve forever."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--urdf", type=Path, default=Path(__file__).with_name("so101_leisaac.urdf"))
    parser.add_argument("--frame", choices=("sim", "urdf"), default="sim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8116)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    solver = So101Solver(args.urdf, frame=args.frame)
    wxyz, xyz = solver.tcp_pose(READY_CFG)  # warm up the JIT before serving
    solver.solve(wxyz, xyz)
    if args.selftest:
        return selftest(solver)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(solver))
    log.info(
        "SO-101 IK server on http://%s:%d (frame=%s, tcp=%s)",
        args.host,
        args.port,
        args.frame,
        TCP_LINK,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
