#!/usr/bin/env python3
"""High-rate robot_state + UDP mocap logging with camera-time alignment.

Captures every /robot_state message (plus OptiTrack body + flipper poses)
at the native publish rate, then aligns those samples to RGB-D frame 
timestamps (from camera 0) after each trial. Output includes:
  - robot_state_raw: full-rate samples
  - robot_state: nearest-neighbor aligned to camera_time_0
  - mocap_raw: full-rate UDP mocap samples grouped by rigid body ID
  - mocap: nearest-neighbor aligned to camera_time_0
  - per-trial metadata and session-level metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import signal
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray

try:
    from .trajectory import (
        ADDUCTION_DISPLACEMENTS_DEG,
        DEFAULT_TRAJECTORY_NAME,
        SWEEP_DISPLACEMENTS_DEG,
        TRAJ_SPEED_RAD_S,
        TrajectorySpec,
        parse_trajectory_name,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from trajectory import (
        ADDUCTION_DISPLACEMENTS_DEG,
        DEFAULT_TRAJECTORY_NAME,
        SWEEP_DISPLACEMENTS_DEG,
        TRAJ_SPEED_RAD_S,
        TrajectorySpec,
        parse_trajectory_name,
    )

try:
    import pyrealsense2 as rs
except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
    raise SystemExit(
        "pyrealsense2 is not installed.\n"
        "Install the Intel RealSense SDK Python bindings before running this program."
    ) from exc

SESSION_ROOT = Path(__file__).resolve().parents[2] / "data"
REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "output" / "references"
DEFAULT_TIMEZONE = os.environ.get("TERRAIN_TIMEZONE", "Etc/GMT+8")

STREAM_WIDTH = 848
STREAM_HEIGHT = 480
STREAM_FPS = 30
DEPTH_MIN_M = None
DEPTH_MAX_M = None
DEPTH_SCHEME = "jet"
DEPTH_HIST_EQ = False
DEPTH_POSTPROCESS = False
TRIAL_COUNT = 1
HEIGHT_CM = 4
# Dwell duration after /trajectory_complete before ending the trial record.
DWELL_TIME_S = 3.0
SAVE_RGB_MP4 = False
MOCAP_ENABLED = True
MOCAP_UDP_IP = "0.0.0.0"
MOCAP_UDP_PORT = 8000
MOCAP_PACKET_SIZE_BYTES = 4096
MOCAP_RB_NAMES = {
    2: "Empty Half Sphere",
    3: "Lead Half Sphere",
    6: "Steel Half Sphere",
    5: "Resin Half Sphere",
    8: "Sand Half Sphere",
}
# Orientation reference mode for roll/pitch/yaw:
#   "trial"   -> re-zero each trial (recommended if object is re-placed each trial)
#   "session" -> keep one reference for the whole run
MOCAP_REFERENCE_MODE = "session"
MOCAP_INCLINE_DEG = 0.0

DEFAULT_TRAJECTORY_SEQUENCE = (f"{DEFAULT_TRAJECTORY_NAME}:1",)
SCHEDULE_VERSION = "terrain_manipulation_v1"
DEFAULT_SCHEDULE_SEED = 20260713
SCHEDULE_DIRECTION = "front"
SCHEDULE_DIRECTIONS = ("front", "back")
SCHEDULE_SWEEPS_PER_TRIAL = 5
EXPERIMENT_COMBINATIONS = tuple(
    (adduction_deg, sweep_deg)
    for adduction_deg in ADDUCTION_DISPLACEMENTS_DEG
    for sweep_deg in SWEEP_DISPLACEMENTS_DEG
    if (adduction_deg, sweep_deg) != (90, 90)
)
EXPERIMENT_ACTIONS = tuple(
    (adduction_deg, sweep_deg, direction)
    for adduction_deg, sweep_deg in EXPERIMENT_COMBINATIONS
    for direction in SCHEDULE_DIRECTIONS
)


@dataclass(frozen=True)
class TrajectorySequenceBlock:
    spec: TrajectorySpec
    count: int

    def as_metadata(self) -> Dict[str, object]:
        payload = self.spec.as_metadata()
        payload["count"] = int(self.count)
        return payload


def parse_trajectory_sequence(items: List[str]) -> List[TrajectorySequenceBlock]:
    blocks: List[TrajectorySequenceBlock] = []
    for item in items:
        if ":" in item:
            name, count_text = item.split(":", maxsplit=1)
        else:
            name, count_text = item, "1"
        name = name.strip()
        if not name:
            raise ValueError(f"empty trajectory name in {item!r}")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"trajectory repeat count must be an integer in {item!r}") from exc
        if count <= 0:
            raise ValueError(f"trajectory repeat count must be positive in {item!r}")
        blocks.append(TrajectorySequenceBlock(parse_trajectory_name(name), count))
    if not blocks:
        raise ValueError("trajectory sequence must contain at least one trajectory")
    return blocks


def trajectory_sequence_metadata(blocks: List[TrajectorySequenceBlock]) -> List[Dict[str, object]]:
    return [block.as_metadata() for block in blocks]


def first_trajectory_spec(blocks: List[TrajectorySequenceBlock]) -> TrajectorySpec:
    return blocks[0].spec


def _schedule_trajectory_name(adduction_deg: int, sweep_deg: int, direction: str) -> str:
    speed_token = f"{float(TRAJ_SPEED_RAD_S):g}".replace(".", "p")
    return f"{int(adduction_deg)}_{int(sweep_deg)}_{speed_token}_{direction}"


def generate_experiment_schedule(seed: int) -> List[List[Tuple[int, int, str]]]:
    rng = random.Random(int(seed))
    schedule: List[List[Tuple[int, int, str]]] = []
    for adduction_deg, sweep_deg in EXPERIMENT_COMBINATIONS:
        trial = [(adduction_deg, sweep_deg, SCHEDULE_DIRECTION)]
        trial.extend(rng.choice(EXPERIMENT_ACTIONS) for _ in range(SCHEDULE_SWEEPS_PER_TRIAL - 1))
        schedule.append(trial)
    return schedule


def _jsonable_schedule(schedule: List[List[Tuple[int, int, str]]]) -> List[List[List[object]]]:
    return [[[int(a), int(s), str(direction)] for a, s, direction in trial] for trial in schedule]


def schedule_json(schedule: List[List[Tuple[int, int, str]]]) -> str:
    return json.dumps(_jsonable_schedule(schedule), separators=(",", ":"), sort_keys=True)


def schedule_hash(schedule: List[List[Tuple[int, int, str]]]) -> str:
    return hashlib.sha256(schedule_json(schedule).encode("utf-8")).hexdigest()


def build_scheduled_trajectory_sequence(
    seed: int,
    trial_index: int,
    height_cm: float,
) -> Tuple[List[str], Dict[str, object]]:
    schedule = generate_experiment_schedule(seed)
    if trial_index < 0 or trial_index >= len(schedule):
        raise ValueError(f"schedule trial index must be 0 through {len(schedule) - 1}, got {trial_index}")

    selected_trial = schedule[trial_index]
    selected_names = [_schedule_trajectory_name(a, s, direction) for a, s, direction in selected_trial]
    for name in selected_names:
        parse_trajectory_name(name)

    metadata = {
        "schedule_version": SCHEDULE_VERSION,
        "schedule_seed": int(seed),
        "schedule_trial_index": int(trial_index),
        "complete_schedule": _jsonable_schedule(schedule),
        "complete_schedule_hash": schedule_hash(schedule),
        "selected_sequence": [[int(a), int(s), str(direction)] for a, s, direction in selected_trial],
        "selected_trajectory_names": selected_names,
        "starting_combination": [int(selected_trial[0][0]), int(selected_trial[0][1])],
        "starting_direction": str(selected_trial[0][2]),
        "valid_combinations": [[int(a), int(s)] for a, s in EXPERIMENT_COMBINATIONS],
        "valid_actions": [[int(a), int(s), str(direction)] for a, s, direction in EXPERIMENT_ACTIONS],
        "height_cm": float(height_cm),
        "starting_direction_policy": SCHEDULE_DIRECTION,
        "random_tail_directions": list(SCHEDULE_DIRECTIONS),
        "speed_rad_s": float(TRAJ_SPEED_RAD_S),
    }
    return [f"{name}:1" for name in selected_names], metadata


def _resolve_now(timezone_name: Optional[str]) -> datetime:
    """Return an aware datetime using the requested timezone, falling back to local time."""
    if timezone_name:
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
        except ModuleNotFoundError:  # pragma: no cover - Python 3.8 fallback
            try:
                from backports.zoneinfo import ZoneInfo  # type: ignore
            except ModuleNotFoundError:
                ZoneInfo = None  # type: ignore
        if ZoneInfo is not None:
            try:
                return datetime.now(ZoneInfo(timezone_name))
            except Exception:
                pass  # fall through to local time
    now = datetime.now()
    if now.tzinfo is None:
        return now.astimezone()
    return now


def _format_height_cm_label(height_cm: float) -> str:
    label = f"{float(height_cm):g}"
    return label.replace("-", "minus").replace(".", "p")


def ensure_session_dir(run_time: datetime, height_cm: float) -> Path:
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = run_time.strftime("%Y%m%d_%H%M%S")
    height_label = _format_height_cm_label(height_cm)
    session_dir = SESSION_ROOT / f"session_{timestamp}_height_{height_label}cm"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def ensure_reference_session_dir(run_time: datetime, height_cm: float, reference_session_dir: Path) -> Path:
    reference_root = SESSION_ROOT / reference_session_dir.name
    sessions_root = reference_root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    reference_copy = reference_root / "reference"
    if not reference_copy.exists():
        shutil.copytree(reference_session_dir, reference_copy)

    timestamp = run_time.strftime("%Y%m%d_%H%M%S")
    height_label = _format_height_cm_label(height_cm)
    session_dir = sessions_root / f"session_{timestamp}_height_{height_label}cm"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def save_session_metadata(
    session_dir: Path,
    start_time: datetime,
    stop_time: datetime,
    trials_planned: int,
    trials_completed: int,
    trial_duration_sec: float,
    dwell_time_sec: float,
    depth_scale_0: float,
    depth_scale_1: float,
    camera_serial_0: str,
    camera_serial_1: str,
    reference_session_dir: Path,
    reference_camera_dir_0: Path,
    reference_camera_dir_1: Path,
    mocap_incline_deg: float,
    height_cm: float,
    trajectory_sequence_request: List[Dict[str, object]],
    schedule_metadata: Optional[Dict[str, object]] = None,
) -> None:
    first_trajectory = trajectory_sequence_request[0] if trajectory_sequence_request else {}
    payload = {
        "start_time": start_time.isoformat(),
        "stop_time": stop_time.isoformat(),
        "duration_sec": (stop_time - start_time).total_seconds(),
        "trials_planned": int(trials_planned),
        "trials_completed": int(trials_completed),
        "trial_duration_sec": float(trial_duration_sec),
        "dwell_time_sec": float(dwell_time_sec),
        "slope": 0,
        "initial_compaction": -1,
        "height_cm": float(height_cm),
        "trajectory_name": first_trajectory.get("name", ""),
        "trajectory_sequence_request": trajectory_sequence_request,
        "image_resolution": [int(STREAM_WIDTH), int(STREAM_HEIGHT)],
        "fps": int(STREAM_FPS),
        "histogram_equalization": bool(DEPTH_HIST_EQ),
        "depth_scale_0": float(depth_scale_0),
        "depth_scale_1": float(depth_scale_1),
        "camera_serial_0": str(camera_serial_0),
        "camera_serial_1": str(camera_serial_1),
        "depth_units": "plane_corrected_height_mm",
        "reference_session_dir": str(reference_session_dir),
        "reference_camera_dir_0": str(reference_camera_dir_0),
        "reference_camera_dir_1": str(reference_camera_dir_1),
        "mocap_enabled": bool(MOCAP_ENABLED),
        "mocap_udp_ip": str(MOCAP_UDP_IP),
        "mocap_udp_port": int(MOCAP_UDP_PORT),
        "mocap_reference_mode": str(MOCAP_REFERENCE_MODE),
        "mocap_incline_deg": float(mocap_incline_deg),
        "mocap_rigid_body_names": {str(key): value for key, value in sorted(MOCAP_RB_NAMES.items())},
    }
    if schedule_metadata is not None:
        payload.update(schedule_metadata)
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def build_metadata(
    start_time: datetime,
    stop_time: datetime,
    dwell_time_sec: float,
    mocap_incline_deg: float,
    height_cm: float,
    trajectory_sequence_request: List[Dict[str, object]],
    traj_complete_time_sec: Optional[float] = None,
    mocap_summary: Optional[Dict[str, object]] = None,
    schedule_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    first_trajectory = trajectory_sequence_request[0] if trajectory_sequence_request else {}
    metadata = {
        "start_time": start_time.isoformat(),
        "stop_time": stop_time.isoformat(),
        "duration_sec": (stop_time - start_time).total_seconds(),
        "dwell_time_sec": float(dwell_time_sec),
        "height_cm": float(height_cm),
        "trajectory_name": first_trajectory.get("name", ""),
        "trajectory_sequence_request": trajectory_sequence_request,
        "mocap_enabled": bool(MOCAP_ENABLED),
        "mocap_reference_mode": str(MOCAP_REFERENCE_MODE),
        "mocap_incline_deg": float(mocap_incline_deg),
    }
    if traj_complete_time_sec is not None:
        metadata["traj_complete_time_sec"] = float(traj_complete_time_sec)
    if mocap_summary is not None:
        metadata["mocap_summary"] = mocap_summary
    if schedule_metadata is not None:
        metadata.update(schedule_metadata)
    return metadata


def bind_signal(sig: int, handler) -> None:
    try:
        signal.signal(sig, handler)
    except ValueError:
        # Signals are not available on all platforms (e.g., Windows), so ignore failure.
        pass


def _build_gui_message(start_flag: float) -> Float64MultiArray:
    msg = Float64MultiArray()
    msg.data = [float(start_flag), 0.0]
    return msg


def _try_set(opt_owner, option, value) -> None:
    try:
        opt_owner.set_option(option, value)
    except Exception:
        pass


def _make_colorizer() -> rs.colorizer:
    scheme_map = {
        "jet": 0,
        "classic": 1,
        "white_to_black": 2,
        "black_to_white": 3,
        "bio": 4,
        "cold": 5,
        "warm": 6,
        "quantized": 7,
        "pattern": 8,
        "turbo": 9,
    }
    cz = rs.colorizer()
    scheme = scheme_map.get(DEPTH_SCHEME, 0)
    _try_set(cz, rs.option.color_scheme, float(scheme))
    if DEPTH_MIN_M is not None:
        _try_set(cz, rs.option.min_distance, float(DEPTH_MIN_M))
    if DEPTH_MAX_M is not None:
        _try_set(cz, rs.option.max_distance, float(DEPTH_MAX_M))
    _try_set(cz, rs.option.histogram_equalization_enabled, 1.0 if DEPTH_HIST_EQ else 0.0)
    return cz


def _open_rgb_writer(path: Path, fps: int, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def _write_rgb_frame(writer: Optional[cv2.VideoWriter], frame: np.ndarray, size: Tuple[int, int]) -> None:
    if writer is None:
        return
    if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    writer.write(frame)


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass
class CameraReference:
    serial: str
    reference_dir: Path
    depth_scale: float
    intrinsics: CameraIntrinsics
    plane: np.ndarray


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _latest_reference_session_path() -> Path:
    pointer = REFERENCE_ROOT / "latest_reference_session.json"
    if not pointer.exists():
        raise SystemExit(
            f"No latest reference session found at {pointer}. "
            "Run src/data_collectors/record_realsense_reference.py first or pass --reference-session."
        )
    with open(pointer, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    session_dir = payload.get("session_dir")
    if not session_dir:
        raise SystemExit(f"Latest reference pointer does not contain session_dir: {pointer}")
    return Path(session_dir)


def _resolve_reference_session(path: Optional[Path]) -> Path:
    reference_session = _latest_reference_session_path() if path is None else path
    if reference_session.is_file():
        with open(reference_session, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if "session_dir" in payload:
            reference_session = Path(payload["session_dir"])
        elif "latest_reference" in payload:
            reference_session = Path(payload["latest_reference"])
    reference_session = reference_session.expanduser().resolve()
    if not reference_session.exists():
        raise SystemExit(f"Reference session does not exist: {reference_session}")
    return reference_session


def _load_camera_reference(reference_session: Path, serial: str) -> CameraReference:
    camera_dir = reference_session / _safe_name(serial)
    metadata_path = camera_dir / "metadata.json"
    if not metadata_path.exists():
        available = sorted(path.name for path in reference_session.iterdir() if path.is_dir())
        raise SystemExit(
            f"No reference for camera serial {serial} in {reference_session}. "
            f"Available reference folders: {available}"
        )

    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    intr = metadata["intrinsics"]
    reference_serial = str(metadata["device_serial"])
    if reference_serial != serial:
        raise SystemExit(f"Reference serial mismatch: expected {serial}, got {reference_serial}")
    plane = np.load(camera_dir / metadata["files"]["sand_plane_abcd"])
    return CameraReference(
        serial=serial,
        reference_dir=camera_dir,
        depth_scale=float(metadata["depth_scale_m_per_unit"]),
        intrinsics=CameraIntrinsics(
            width=int(intr["width"]),
            height=int(intr["height"]),
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            ppx=float(intr["ppx"]),
            ppy=float(intr["ppy"]),
        ),
        plane=plane,
    )


def _pixel_rays(intr: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray]:
    v, u = np.indices((intr.height, intr.width), dtype=np.float32)
    x_coeff = (u - intr.ppx) / intr.fx
    y_coeff = (v - intr.ppy) / intr.fy
    return x_coeff, y_coeff


def _signed_plane_distance_mm(
    depth_raw: np.ndarray,
    depth_scale: float,
    x_coeff: np.ndarray,
    y_coeff: np.ndarray,
    plane: np.ndarray,
) -> np.ndarray:
    z = depth_raw.astype(np.float32) * depth_scale
    x = x_coeff * z
    y = y_coeff * z
    a, b, c, d = plane
    height_mm = ((a * x + b * y + c * z + d) / np.linalg.norm(plane[:3])) * 1000.0
    height_mm[depth_raw <= 0] = np.nan
    return height_mm.astype(np.float32)


class RGBDRecorder:
    def __init__(self) -> None:
        self.color_raw_frames: List[np.ndarray] = []
        self.depth_raw_frames: List[np.ndarray] = []
        self.timestamps: List[float] = []

    def write(self, color_image: np.ndarray, depth_raw: np.ndarray, timestamp: float) -> None:
        self.color_raw_frames.append(color_image.copy())
        self.depth_raw_frames.append(depth_raw.copy())
        self.timestamps.append(float(timestamp))

    def finalize(self) -> Dict[str, np.ndarray]:
        rgb = np.stack(self.color_raw_frames) if self.color_raw_frames else np.empty((0,))
        depth = np.stack(self.depth_raw_frames) if self.depth_raw_frames else np.empty((0,))
        timestamps = np.asarray(self.timestamps, dtype=float)
        self.color_raw_frames.clear()
        self.depth_raw_frames.clear()
        self.timestamps.clear()
        return {
            "rgb": rgb,
            "depth": depth,
            "timestamps": timestamps,
        }


class RealSenseSession:
    def __init__(self, serial: Optional[str] = None, reference: Optional[CameraReference] = None) -> None:
        self.serial = serial
        self.reference = reference
        self.x_coeff: Optional[np.ndarray] = None
        self.y_coeff: Optional[np.ndarray] = None
        if reference is not None:
            self.x_coeff, self.y_coeff = _pixel_rays(reference.intrinsics)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16, STREAM_FPS)
        self.config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, STREAM_FPS)
        self.colorizer = _make_colorizer()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole = rs.hole_filling_filter()
        self.align = rs.align(rs.stream.color)

    def start(self) -> None:
        self.pipeline.start(self.config)

    def stop(self) -> None:
        self.pipeline.stop()

    def poll(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            raise RuntimeError("Incomplete RGB-D frame received from RealSense pipeline.")

        if DEPTH_POSTPROCESS:
            depth = self.spatial.process(depth)
            depth = self.temporal.process(depth)
            depth = self.hole.process(depth)

        depth_color = self.colorizer.colorize(depth)
        depth_img = np.asanyarray(depth_color.get_data())
        depth_bgr = cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR)
        color_img = np.asanyarray(color.get_data())
        depth_raw = np.asanyarray(depth.get_data())
        if self.reference is not None:
            if depth_raw.shape != (self.reference.intrinsics.height, self.reference.intrinsics.width):
                raise RuntimeError(
                    "Depth frame shape does not match reference intrinsics: "
                    f"{depth_raw.shape} vs {(self.reference.intrinsics.height, self.reference.intrinsics.width)}"
                )
            assert self.x_coeff is not None and self.y_coeff is not None
            depth_out = _signed_plane_distance_mm(
                depth_raw,
                self.reference.depth_scale,
                self.x_coeff,
                self.y_coeff,
                self.reference.plane,
            )
        else:
            depth_out = depth_raw
        return color_img, depth_out, depth_bgr


def _get_realsense_serials() -> List[str]:
    ctx = rs.context()
    devices = ctx.query_devices()
    serials: List[str] = []
    for dev in devices:
        try:
            serials.append(dev.get_info(rs.camera_info.serial_number))
        except Exception:
            continue
    return sorted(serials)


@dataclass
class RobotStateSample:
    time_s: float
    turtle_state: float
    leftadduction_pos: float
    leftsweeping_pos: float
    rightadduction_pos: float
    rightsweeping_pos: float
    leftadduction_curr: float
    leftsweeping_curr: float
    rightadduction_curr: float
    rightsweeping_curr: float
    optitrack_position_x: float
    optitrack_position_y: float
    optitrack_position_z: float
    optitrack_orientation_x: float
    optitrack_orientation_y: float
    optitrack_orientation_z: float
    optitrack_orientation_w: float
    left_flipper_position_x: float
    left_flipper_position_y: float
    left_flipper_position_z: float
    left_flipper_orientation_x: float
    left_flipper_orientation_y: float
    left_flipper_orientation_z: float
    left_flipper_orientation_w: float
    right_flipper_position_x: float
    right_flipper_position_y: float
    right_flipper_position_z: float
    right_flipper_orientation_x: float
    right_flipper_orientation_y: float
    right_flipper_orientation_z: float
    right_flipper_orientation_w: float


@dataclass
class MocapSample:
    time_s: float
    rigid_body_id: int
    position_x: float
    position_y: float
    position_z: float
    zeroed_position_x: float
    zeroed_position_y: float
    zeroed_position_z: float
    rotated_position_x: float
    rotated_position_y: float
    rotated_position_z: float
    rotated_zeroed_position_x: float
    rotated_zeroed_position_y: float
    rotated_zeroed_position_z: float
    orientation_x: float
    orientation_y: float
    orientation_z: float
    orientation_w: float
    zeroed_orientation_x: float
    zeroed_orientation_y: float
    zeroed_orientation_z: float
    zeroed_orientation_w: float
    rotated_orientation_x: float
    rotated_orientation_y: float
    rotated_orientation_z: float
    rotated_orientation_w: float
    rotated_zeroed_orientation_x: float
    rotated_zeroed_orientation_y: float
    rotated_zeroed_orientation_z: float
    rotated_zeroed_orientation_w: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    rotated_roll_deg: float
    rotated_pitch_deg: float
    rotated_yaw_deg: float


class MocapUDPReceiver:
    def __init__(self, udp_ip: str, udp_port: int, incline_deg: float = MOCAP_INCLINE_DEG) -> None:
        self.udp_ip = udp_ip
        self.udp_port = int(udp_port)
        self.incline_deg = float(incline_deg)
        # Clockwise y-z plane rotation equals a -x right-handed rotation.
        self._frame_rot = R.from_euler("x", -self.incline_deg, degrees=True)
        self._lock = threading.Lock()
        self._run_start = time.time()
        self._samples_by_rigid_body: Dict[int, List[MocapSample]] = {}
        self._initial_rot: Dict[int, R] = {}
        self._initial_pos: Dict[int, np.ndarray] = {}
        self._packets_received = 0
        self._decode_errors = 0
        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.udp_ip, self.udp_port))
        self._sock.settimeout(0.2)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested.set()
        try:
            self._sock.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def reset(self, run_start: float, reset_reference: bool = True) -> None:
        with self._lock:
            self._run_start = run_start
            self._samples_by_rigid_body.clear()
            if reset_reference:
                self._initial_rot.clear()
                self._initial_pos.clear()
            self._packets_received = 0
            self._decode_errors = 0

    def snapshot(self) -> Dict[int, List[MocapSample]]:
        with self._lock:
            return {rigid_body_id: list(samples) for rigid_body_id, samples in self._samples_by_rigid_body.items()}

    def summary(self) -> Dict[str, object]:
        with self._lock:
            return {
                "packets_received": int(self._packets_received),
                "decode_errors": int(self._decode_errors),
                "rigid_body_ids": [int(rigid_body_id) for rigid_body_id in sorted(self._samples_by_rigid_body.keys())],
                "samples_per_rigid_body": {
                    str(rigid_body_id): int(len(samples))
                    for rigid_body_id, samples in sorted(self._samples_by_rigid_body.items())
                },
                "rigid_body_names": {str(key): value for key, value in sorted(MOCAP_RB_NAMES.items())},
                "udp_ip": self.udp_ip,
                "udp_port": self.udp_port,
                "incline_deg": float(self.incline_deg),
            }

    def _recv_loop(self) -> None:
        while not self._stop_requested.is_set():
            try:
                packet, _addr = self._sock.recvfrom(MOCAP_PACKET_SIZE_BYTES)
            except socket.timeout:
                continue
            except OSError:
                if self._stop_requested.is_set():
                    break
                continue
            self._handle_packet(packet)

    def _record_decode_error(self) -> None:
        with self._lock:
            self._decode_errors += 1

    def _handle_packet(self, packet: bytes) -> None:
        with self._lock:
            self._packets_received += 1

        try:
            rb = pickle.loads(packet)
        except Exception:
            self._record_decode_error()
            return

        if not hasattr(rb, "__len__") or len(rb) < 8:
            self._record_decode_error()
            return

        try:
            rigid_body_id = int(rb[0])
            quat = np.asarray(rb[1:5], dtype=float)
            pos = np.asarray(rb[5:8], dtype=float)
        except Exception:
            self._record_decode_error()
            return

        if quat.shape != (4,) or pos.shape != (3,):
            self._record_decode_error()
            return
        if not np.all(np.isfinite(quat)) or not np.all(np.isfinite(pos)):
            self._record_decode_error()
            return

        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-9:
            self._record_decode_error()
            return
        quat = quat / quat_norm

        try:
            current_rot = R.from_quat(quat)
        except Exception:
            self._record_decode_error()
            return

        with self._lock:
            if rigid_body_id not in self._initial_rot:
                self._initial_rot[rigid_body_id] = current_rot
                relative_rot = R.identity()
            else:
                # World-relative zeroed rotation: initial pose is treated as the
                # session reference frame, and deltas are expressed in fixed axes.
                relative_rot = current_rot * self._initial_rot[rigid_body_id].inv()
            if rigid_body_id not in self._initial_pos:
                self._initial_pos[rigid_body_id] = pos.copy()
            zeroed_pos = pos - self._initial_pos[rigid_body_id]
            zeroed_quat = relative_rot.as_quat()
            # Report RPY in fixed axes of the zeroed trial frame (extrinsic xyz).
            roll_deg, pitch_deg, yaw_deg = relative_rot.as_euler("xyz", degrees=True)
            rotated_pos = self._frame_rot.apply(pos)
            rotated_zeroed_pos = self._frame_rot.apply(zeroed_pos)
            # Express orientation in the rotated frame via basis change:
            # R' = F * R * F^{-1}
            rotated_rot = self._frame_rot * current_rot * self._frame_rot.inv()
            rotated_zeroed_rot = self._frame_rot * relative_rot * self._frame_rot.inv()
            rotated_quat = rotated_rot.as_quat()
            rotated_zeroed_quat = rotated_zeroed_rot.as_quat()
            rotated_roll_deg, rotated_pitch_deg, rotated_yaw_deg = rotated_zeroed_rot.as_euler("xyz", degrees=True)

            sample = MocapSample(
                time_s=time.time() - self._run_start,
                rigid_body_id=rigid_body_id,
                position_x=float(pos[0]),
                position_y=float(pos[1]),
                position_z=float(pos[2]),
                zeroed_position_x=float(zeroed_pos[0]),
                zeroed_position_y=float(zeroed_pos[1]),
                zeroed_position_z=float(zeroed_pos[2]),
                rotated_position_x=float(rotated_pos[0]),
                rotated_position_y=float(rotated_pos[1]),
                rotated_position_z=float(rotated_pos[2]),
                rotated_zeroed_position_x=float(rotated_zeroed_pos[0]),
                rotated_zeroed_position_y=float(rotated_zeroed_pos[1]),
                rotated_zeroed_position_z=float(rotated_zeroed_pos[2]),
                orientation_x=float(quat[0]),
                orientation_y=float(quat[1]),
                orientation_z=float(quat[2]),
                orientation_w=float(quat[3]),
                zeroed_orientation_x=float(zeroed_quat[0]),
                zeroed_orientation_y=float(zeroed_quat[1]),
                zeroed_orientation_z=float(zeroed_quat[2]),
                zeroed_orientation_w=float(zeroed_quat[3]),
                rotated_orientation_x=float(rotated_quat[0]),
                rotated_orientation_y=float(rotated_quat[1]),
                rotated_orientation_z=float(rotated_quat[2]),
                rotated_orientation_w=float(rotated_quat[3]),
                rotated_zeroed_orientation_x=float(rotated_zeroed_quat[0]),
                rotated_zeroed_orientation_y=float(rotated_zeroed_quat[1]),
                rotated_zeroed_orientation_z=float(rotated_zeroed_quat[2]),
                rotated_zeroed_orientation_w=float(rotated_zeroed_quat[3]),
                roll_deg=float(roll_deg),
                pitch_deg=float(pitch_deg),
                yaw_deg=float(yaw_deg),
                rotated_roll_deg=float(rotated_roll_deg),
                rotated_pitch_deg=float(rotated_pitch_deg),
                rotated_yaw_deg=float(rotated_yaw_deg),
            )
            self._samples_by_rigid_body.setdefault(rigid_body_id, []).append(sample)


class ControlNodeHighRate(Node):
    def __init__(self) -> None:
        super().__init__("control_node_highrate")
        self.publisher_ = self.create_publisher(Float64MultiArray, "/Gui_information", 10)
        self.create_subscription(Float64MultiArray, "/robot_state", self._robot_state_cb, 10)
        self.create_subscription(Bool, "/trajectory_complete", self._traj_complete_cb, 10)
        self.create_subscription(Pose, "/optitrack_body", self._optitrack_cb, 10)
        self.create_subscription(Pose, "/optitrack_left_flipper", self._left_flipper_cb, 10)
        self.create_subscription(Pose, "/optitrack_right_flipper", self._right_flipper_cb, 10)
        self.run_start = time.time()
        self._lock = threading.Lock()
        self._samples: List[RobotStateSample] = []
        self._traj_complete = False
        self._optitrack_pose: Optional[Pose] = None
        self._left_flipper_pose: Optional[Pose] = None
        self._right_flipper_pose: Optional[Pose] = None

    def publish_gui_information(self, msg: Float64MultiArray) -> None:
        self.publisher_.publish(msg)

    def reset(self, run_start: float) -> None:
        with self._lock:
            self.run_start = run_start
            self._samples.clear()
            self._optitrack_pose = None
            self._left_flipper_pose = None
            self._right_flipper_pose = None
            self._traj_complete = False

    def clear_traj_complete(self) -> None:
        with self._lock:
            self._traj_complete = False

    def _traj_complete_cb(self, msg: Bool) -> None:
        with self._lock:
            self._traj_complete = bool(msg.data)

    def _optitrack_cb(self, msg: Pose) -> None:
        with self._lock:
            self._optitrack_pose = msg

    def _left_flipper_cb(self, msg: Pose) -> None:
        with self._lock:
            self._left_flipper_pose = msg

    def _right_flipper_cb(self, msg: Pose) -> None:
        with self._lock:
            self._right_flipper_pose = msg

    def _robot_state_cb(self, msg: Float64MultiArray) -> None:
        now_s = time.time() - self.run_start
        data = msg.data
        if len(data) < 9:
            return
        with self._lock:
            optitrack_pose = self._optitrack_pose
            left_flipper_pose = self._left_flipper_pose
            right_flipper_pose = self._right_flipper_pose

        sample = RobotStateSample(
            time_s=now_s,
            turtle_state=data[0],
            leftadduction_pos=data[1],
            leftsweeping_pos=data[2],
            rightadduction_pos=data[3],
            rightsweeping_pos=data[4],
            leftadduction_curr=data[5],
            leftsweeping_curr=data[6],
            rightadduction_curr=data[7],
            rightsweeping_curr=data[8],
            optitrack_position_x=0.0 if optitrack_pose is None else optitrack_pose.position.x,
            optitrack_position_y=0.0 if optitrack_pose is None else optitrack_pose.position.y,
            optitrack_position_z=0.0 if optitrack_pose is None else optitrack_pose.position.z,
            optitrack_orientation_x=0.0 if optitrack_pose is None else optitrack_pose.orientation.x,
            optitrack_orientation_y=0.0 if optitrack_pose is None else optitrack_pose.orientation.y,
            optitrack_orientation_z=0.0 if optitrack_pose is None else optitrack_pose.orientation.z,
            optitrack_orientation_w=1.0 if optitrack_pose is None else optitrack_pose.orientation.w,
            left_flipper_position_x=0.0 if left_flipper_pose is None else left_flipper_pose.position.x,
            left_flipper_position_y=0.0 if left_flipper_pose is None else left_flipper_pose.position.y,
            left_flipper_position_z=0.0 if left_flipper_pose is None else left_flipper_pose.position.z,
            left_flipper_orientation_x=0.0 if left_flipper_pose is None else left_flipper_pose.orientation.x,
            left_flipper_orientation_y=0.0 if left_flipper_pose is None else left_flipper_pose.orientation.y,
            left_flipper_orientation_z=0.0 if left_flipper_pose is None else left_flipper_pose.orientation.z,
            left_flipper_orientation_w=1.0 if left_flipper_pose is None else left_flipper_pose.orientation.w,
            right_flipper_position_x=0.0 if right_flipper_pose is None else right_flipper_pose.position.x,
            right_flipper_position_y=0.0 if right_flipper_pose is None else right_flipper_pose.position.y,
            right_flipper_position_z=0.0 if right_flipper_pose is None else right_flipper_pose.position.z,
            right_flipper_orientation_x=0.0 if right_flipper_pose is None else right_flipper_pose.orientation.x,
            right_flipper_orientation_y=0.0 if right_flipper_pose is None else right_flipper_pose.orientation.y,
            right_flipper_orientation_z=0.0 if right_flipper_pose is None else right_flipper_pose.orientation.z,
            right_flipper_orientation_w=1.0 if right_flipper_pose is None else right_flipper_pose.orientation.w,
        )
        with self._lock:
            self._samples.append(sample)

    def snapshot(self) -> List[RobotStateSample]:
        with self._lock:
            return list(self._samples)

    def is_traj_complete(self) -> bool:
        with self._lock:
            return self._traj_complete


def _build_robot_state(samples: List[RobotStateSample]) -> Dict[str, np.ndarray]:
    return {
        "time": np.asarray([s.time_s for s in samples], dtype=float),
        "turtle_state": np.asarray([s.turtle_state for s in samples], dtype=float),
        "leftadduction_pos": np.asarray([s.leftadduction_pos for s in samples], dtype=float),
        "leftsweeping_pos": np.asarray([s.leftsweeping_pos for s in samples], dtype=float),
        "rightadduction_pos": np.asarray([s.rightadduction_pos for s in samples], dtype=float),
        "rightsweeping_pos": np.asarray([s.rightsweeping_pos for s in samples], dtype=float),
        "leftadduction_curr": np.asarray([s.leftadduction_curr for s in samples], dtype=float),
        "leftsweeping_curr": np.asarray([s.leftsweeping_curr for s in samples], dtype=float),
        "rightadduction_curr": np.asarray([s.rightadduction_curr for s in samples], dtype=float),
        "rightsweeping_curr": np.asarray([s.rightsweeping_curr for s in samples], dtype=float),
        "OptitrackPosition_x": np.asarray([s.optitrack_position_x for s in samples], dtype=float),
        "OptitrackPosition_y": np.asarray([s.optitrack_position_y for s in samples], dtype=float),
        "OptitrackPosition_z": np.asarray([s.optitrack_position_z for s in samples], dtype=float),
        "OptitrackOrientation_x": np.asarray([s.optitrack_orientation_x for s in samples], dtype=float),
        "OptitrackOrientation_y": np.asarray([s.optitrack_orientation_y for s in samples], dtype=float),
        "OptitrackOrientation_z": np.asarray([s.optitrack_orientation_z for s in samples], dtype=float),
        "OptitrackOrientation_w": np.asarray([s.optitrack_orientation_w for s in samples], dtype=float),
        "LeftFlipperPosition_x": np.asarray([s.left_flipper_position_x for s in samples], dtype=float),
        "LeftFlipperPosition_y": np.asarray([s.left_flipper_position_y for s in samples], dtype=float),
        "LeftFlipperPosition_z": np.asarray([s.left_flipper_position_z for s in samples], dtype=float),
        "LeftFlipperOrientation_x": np.asarray([s.left_flipper_orientation_x for s in samples], dtype=float),
        "LeftFlipperOrientation_y": np.asarray([s.left_flipper_orientation_y for s in samples], dtype=float),
        "LeftFlipperOrientation_z": np.asarray([s.left_flipper_orientation_z for s in samples], dtype=float),
        "LeftFlipperOrientation_w": np.asarray([s.left_flipper_orientation_w for s in samples], dtype=float),
        "RightFlipperPosition_x": np.asarray([s.right_flipper_position_x for s in samples], dtype=float),
        "RightFlipperPosition_y": np.asarray([s.right_flipper_position_y for s in samples], dtype=float),
        "RightFlipperPosition_z": np.asarray([s.right_flipper_position_z for s in samples], dtype=float),
        "RightFlipperOrientation_x": np.asarray([s.right_flipper_orientation_x for s in samples], dtype=float),
        "RightFlipperOrientation_y": np.asarray([s.right_flipper_orientation_y for s in samples], dtype=float),
        "RightFlipperOrientation_z": np.asarray([s.right_flipper_orientation_z for s in samples], dtype=float),
        "RightFlipperOrientation_w": np.asarray([s.right_flipper_orientation_w for s in samples], dtype=float),
    }


def _build_mocap_state(samples_by_rigid_body: Dict[int, List[MocapSample]]) -> Dict[str, Dict[str, np.ndarray]]:
    mocap_state: Dict[str, Dict[str, np.ndarray]] = {}
    for rigid_body_id, samples in sorted(samples_by_rigid_body.items()):
        key = str(rigid_body_id)
        mocap_state[key] = {
            "time": np.asarray([s.time_s for s in samples], dtype=float),
            "rigid_body_id": np.asarray([s.rigid_body_id for s in samples], dtype=float),
            "position_x": np.asarray([s.position_x for s in samples], dtype=float),
            "position_y": np.asarray([s.position_y for s in samples], dtype=float),
            "position_z": np.asarray([s.position_z for s in samples], dtype=float),
            "zeroed_position_x": np.asarray([s.zeroed_position_x for s in samples], dtype=float),
            "zeroed_position_y": np.asarray([s.zeroed_position_y for s in samples], dtype=float),
            "zeroed_position_z": np.asarray([s.zeroed_position_z for s in samples], dtype=float),
            "rotated_position_x": np.asarray([s.rotated_position_x for s in samples], dtype=float),
            "rotated_position_y": np.asarray([s.rotated_position_y for s in samples], dtype=float),
            "rotated_position_z": np.asarray([s.rotated_position_z for s in samples], dtype=float),
            "rotated_zeroed_position_x": np.asarray([s.rotated_zeroed_position_x for s in samples], dtype=float),
            "rotated_zeroed_position_y": np.asarray([s.rotated_zeroed_position_y for s in samples], dtype=float),
            "rotated_zeroed_position_z": np.asarray([s.rotated_zeroed_position_z for s in samples], dtype=float),
            "orientation_x": np.asarray([s.orientation_x for s in samples], dtype=float),
            "orientation_y": np.asarray([s.orientation_y for s in samples], dtype=float),
            "orientation_z": np.asarray([s.orientation_z for s in samples], dtype=float),
            "orientation_w": np.asarray([s.orientation_w for s in samples], dtype=float),
            # Zeroed orientation (initial pose treated as identity/world-aligned).
            "zeroed_orientation_x": np.asarray([s.zeroed_orientation_x for s in samples], dtype=float),
            "zeroed_orientation_y": np.asarray([s.zeroed_orientation_y for s in samples], dtype=float),
            "zeroed_orientation_z": np.asarray([s.zeroed_orientation_z for s in samples], dtype=float),
            "zeroed_orientation_w": np.asarray([s.zeroed_orientation_w for s in samples], dtype=float),
            "rotated_orientation_x": np.asarray([s.rotated_orientation_x for s in samples], dtype=float),
            "rotated_orientation_y": np.asarray([s.rotated_orientation_y for s in samples], dtype=float),
            "rotated_orientation_z": np.asarray([s.rotated_orientation_z for s in samples], dtype=float),
            "rotated_orientation_w": np.asarray([s.rotated_orientation_w for s in samples], dtype=float),
            "rotated_zeroed_orientation_x": np.asarray([s.rotated_zeroed_orientation_x for s in samples], dtype=float),
            "rotated_zeroed_orientation_y": np.asarray([s.rotated_zeroed_orientation_y for s in samples], dtype=float),
            "rotated_zeroed_orientation_z": np.asarray([s.rotated_zeroed_orientation_z for s in samples], dtype=float),
            "rotated_zeroed_orientation_w": np.asarray([s.rotated_zeroed_orientation_w for s in samples], dtype=float),
            "roll_deg": np.asarray([s.roll_deg for s in samples], dtype=float),
            "pitch_deg": np.asarray([s.pitch_deg for s in samples], dtype=float),
            "yaw_deg": np.asarray([s.yaw_deg for s in samples], dtype=float),
            "rotated_roll_deg": np.asarray([s.rotated_roll_deg for s in samples], dtype=float),
            "rotated_pitch_deg": np.asarray([s.rotated_pitch_deg for s in samples], dtype=float),
            "rotated_yaw_deg": np.asarray([s.rotated_yaw_deg for s in samples], dtype=float),
        }
    return mocap_state


def _nearest_indices(sample_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    if sample_times.size == 0:
        return np.zeros_like(query_times, dtype=int)
    idx = np.searchsorted(sample_times, query_times, side="left")
    idx = np.clip(idx, 0, sample_times.size - 1)
    prev_idx = np.clip(idx - 1, 0, sample_times.size - 1)
    next_idx = idx
    prev_diff = np.abs(query_times - sample_times[prev_idx])
    next_diff = np.abs(query_times - sample_times[next_idx])
    choose_prev = prev_diff <= next_diff
    return np.where(choose_prev, prev_idx, next_idx)


def _align_robot_state(robot_state: Dict[str, np.ndarray], camera_times: np.ndarray) -> Dict[str, np.ndarray]:
    sample_times = robot_state["time"]
    indices = _nearest_indices(sample_times, camera_times)
    aligned: Dict[str, np.ndarray] = {"time": sample_times[indices]}
    for key, values in robot_state.items():
        if key == "time":
            continue
        aligned[key] = values[indices]
    return aligned


def _align_mocap_state(
    mocap_state: Dict[str, Dict[str, np.ndarray]],
    camera_times: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    aligned_mocap: Dict[str, Dict[str, np.ndarray]] = {}
    for rigid_body_id, rigid_body_state in mocap_state.items():
        sample_times = rigid_body_state.get("time")
        if not isinstance(sample_times, np.ndarray) or sample_times.size == 0:
            continue
        indices = _nearest_indices(sample_times, camera_times)
        aligned_state: Dict[str, np.ndarray] = {"time": sample_times[indices]}
        for key, values in rigid_body_state.items():
            if key == "time":
                continue
            aligned_state[key] = values[indices]
        aligned_mocap[rigid_body_id] = aligned_state
    return aligned_mocap


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="High-rate robot_state logging with camera alignment.")
    ap.add_argument("--trials", type=int, default=TRIAL_COUNT, help="Number of trials to record.")
    ap.add_argument(
        "--reference-session",
        type=Path,
        default=None,
        help="Reference session folder or metadata JSON. Default: output/references/latest_reference_session.json.",
    )
    ap.add_argument(
        "--height-cm",
        type=float,
        default=HEIGHT_CM,
        help="Physically set experiment height in cm; used for run naming and metadata only.",
    )
    ap.add_argument(
        "--trajectory-sequence",
        nargs="+",
        default=list(DEFAULT_TRAJECTORY_SEQUENCE),
        metavar="NAME:COUNT",
        help=(
            "Ordered trajectory blocks to run inside each trial, e.g. "
            "30_30_2_front:5 30_90_1p5_back:3. COUNT defaults to 1 if omitted."
        ),
    )
    ap.add_argument(
        "--schedule-trial-index",
        type=int,
        choices=range(len(EXPERIMENT_COMBINATIONS)),
        default=None,
        metavar="0..7",
        help="Run one deterministic five-sweep schedule trial by canonical index.",
    )
    ap.add_argument(
        "--schedule-seed",
        type=int,
        default=DEFAULT_SCHEDULE_SEED,
        help=f"Fixed deterministic schedule seed. Default: {DEFAULT_SCHEDULE_SEED}.",
    )
    ap.add_argument(
        "--print-schedule",
        action="store_true",
        help="Print the complete eight-trial deterministic schedule and exit before hardware initialization.",
    )
    ap.add_argument(
        "--incline-deg",
        type=float,
        default=MOCAP_INCLINE_DEG,
        help="Clockwise y-z plane rotation angle (deg) used to save rotated mocap position/orientation.",
    )
    ap.add_argument(
        "--save-rgb-mp4",
        action="store_true",
        default=SAVE_RGB_MP4,
        help="Save RGB MP4 videos for both cameras.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    schedule_metadata: Optional[Dict[str, object]] = None

    if args.print_schedule:
        schedule = generate_experiment_schedule(int(args.schedule_seed))
        print(f"Schedule version: {SCHEDULE_VERSION}")
        print(f"Schedule seed: {int(args.schedule_seed)}")
        print(f"Schedule hash: {schedule_hash(schedule)}")
        print(schedule_json(schedule))
        return 0

    if args.schedule_trial_index is not None:
        if list(args.trajectory_sequence) != list(DEFAULT_TRAJECTORY_SEQUENCE):
            raise SystemExit("--schedule-trial-index cannot be used with custom --trajectory-sequence.")
        if int(args.trials) != 1:
            print(
                f"--schedule-trial-index executes one selected five-sweep trial; "
                f"overriding --trials {int(args.trials)} to 1."
            )
            args.trials = 1
        try:
            scheduled_sequence, schedule_metadata = build_scheduled_trajectory_sequence(
                seed=int(args.schedule_seed),
                trial_index=int(args.schedule_trial_index),
                height_cm=float(args.height_cm),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        args.trajectory_sequence = scheduled_sequence
        print("Deterministic schedule selection:")
        print(f"  seed: {int(args.schedule_seed)}")
        print(f"  height_cm: {float(args.height_cm):g}")
        print(f"  selected_trial_index: {int(args.schedule_trial_index)}")
        print("  selected_trajectory_names:")
        for name in schedule_metadata["selected_trajectory_names"]:
            print(f"    {name}")
        print(f"  complete_schedule_hash: {schedule_metadata['complete_schedule_hash']}")
        print(f"  complete_schedule: {schedule_json(generate_experiment_schedule(int(args.schedule_seed)))}")

    try:
        trajectory_sequence = parse_trajectory_sequence(list(args.trajectory_sequence))
    except ValueError as exc:
        raise SystemExit(f"Invalid --trajectory-sequence: {exc}") from exc
    trajectory_sequence_request = trajectory_sequence_metadata(trajectory_sequence)
    first_trajectory = first_trajectory_spec(trajectory_sequence)

    reference_session_dir = _resolve_reference_session(args.reference_session)
    session_dir = ensure_reference_session_dir(
        _resolve_now(DEFAULT_TIMEZONE),
        height_cm=float(args.height_cm),
        reference_session_dir=reference_session_dir,
    )
    print(f"Session directory: {session_dir}")
    print(f"Reference session: {reference_session_dir}")
    print(f"Experiment height: {float(args.height_cm):g} cm")
    print("Trajectory sequence:")
    for block in trajectory_sequence:
        print(f"  {block.spec.name} x{block.count}")

    rclpy.init()
    node = ControlNodeHighRate()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    stop_requested = threading.Event()

    def executor_thread() -> None:
        while not stop_requested.is_set():
            executor.spin_once(timeout_sec=0.1)

    spin_thread = threading.Thread(target=executor_thread, daemon=True)
    spin_thread.start()

    serials = _get_realsense_serials()
    if len(serials) < 2:
        raise SystemExit("Need two RealSense devices connected for high-rate collection.")
    reference_0 = _load_camera_reference(reference_session_dir, serials[0])
    reference_1 = _load_camera_reference(reference_session_dir, serials[1])
    realsense_primary = RealSenseSession(serials[0], reference=reference_0)
    realsense_secondary = RealSenseSession(serials[1], reference=reference_1)
    realsense_primary.start()
    realsense_secondary.start()
    depth_scale_0 = (
        realsense_primary.pipeline.get_active_profile()
        .get_device()
        .first_depth_sensor()
        .get_depth_scale()
    )
    depth_scale_1 = (
        realsense_secondary.pipeline.get_active_profile()
        .get_device()
        .first_depth_sensor()
        .get_depth_scale()
    )

    trajectory_publisher = node.create_publisher(Float64MultiArray, "/trajectory_points", 10)
    mocap_receiver: Optional[MocapUDPReceiver] = None
    if MOCAP_ENABLED:
        try:
            mocap_receiver = MocapUDPReceiver(MOCAP_UDP_IP, MOCAP_UDP_PORT, incline_deg=float(args.incline_deg))
            mocap_receiver.start()
            print(
                f"Mocap UDP listener active on {MOCAP_UDP_IP}:{MOCAP_UDP_PORT} "
                f"(incline={float(args.incline_deg):.3f} deg)"
            )
        except OSError as exc:
            raise SystemExit(
                f"Failed to bind mocap UDP listener on {MOCAP_UDP_IP}:{MOCAP_UDP_PORT}: {exc}"
            ) from exc

    print(
        "Robot command issued. Recording RGB-D + high-rate telemetry...\n"
        "Depth arrays are plane-corrected height maps in millimeters.\n"
        "Each trial runs until /trajectory_complete is received (CTRL+C to abort)."
    )

    session_start_time = _resolve_now(DEFAULT_TIMEZONE)
    trials_completed = 0
    trial_durations: List[float] = []

    def request_stop(*_args) -> None:
        stop_requested.set()

    bind_signal(signal.SIGINT, request_stop)
    bind_signal(signal.SIGTERM, request_stop)

    input("Press Enter to start the trajectory sequence run...")

    for trial_idx in range(args.trials):
        if stop_requested.is_set():
            break
        recorder = RGBDRecorder()
        recorder_2 = RGBDRecorder()
        rgb_size = (STREAM_WIDTH, STREAM_HEIGHT)
        rgb_writer_0: Optional[object] = None
        rgb_writer_1: Optional[object] = None
        if args.save_rgb_mp4:
            rgb_path_0 = session_dir / f"trial_{trial_idx + 1}_rgb_0.mp4"
            rgb_path_1 = session_dir / f"trial_{trial_idx + 1}_rgb_1.mp4"
            rgb_writer_0 = _open_rgb_writer(rgb_path_0, STREAM_FPS, rgb_size)
            rgb_writer_1 = _open_rgb_writer(rgb_path_1, STREAM_FPS, rgb_size)

        run_start = time.time()
        node.reset(run_start)
        if mocap_receiver is not None:
            reset_reference = (MOCAP_REFERENCE_MODE == "trial") or (trial_idx == 0)
            mocap_receiver.reset(run_start, reset_reference=reset_reference)
        start_time = _resolve_now(DEFAULT_TIMEZONE)
        print(f"Starting trial {trial_idx + 1}/{args.trials}...")
        traj_complete_time_s: Optional[float] = None
        trajectory_runs: List[Dict[str, object]] = []

        try:
            sequence_index = 0
            for block in trajectory_sequence:
                for repeat_index in range(block.count):
                    if stop_requested.is_set():
                        break

                    node.clear_traj_complete()
                    trajectory_msg = Float64MultiArray()
                    trajectory_msg.data = list(block.spec.points)
                    command_time_s = time.time() - run_start
                    run_log: Dict[str, object] = {
                        "run_index": int(sequence_index),
                        "block_repeat_index": int(repeat_index),
                        "name": block.spec.name,
                        "adduction_deg": int(block.spec.adduction_deg),
                        "sweep_deg": int(block.spec.sweep_deg),
                        "speed_rad_s": float(block.spec.speed_rad_s),
                        "direction": block.spec.direction,
                        "points": list(block.spec.points),
                        "command_time_sec": float(command_time_s),
                        "complete_time_sec": None,
                        "status": "running",
                    }
                    trajectory_runs.append(run_log)
                    print(
                        f"  Trajectory {sequence_index + 1}: "
                        f"{block.spec.name} ({repeat_index + 1}/{block.count})"
                    )
                    trajectory_publisher.publish(trajectory_msg)

                    while not stop_requested.is_set():
                        color_img, depth_raw, _depth_bgr = realsense_primary.poll()
                        color_img_2, depth_raw_2, _depth_bgr_2 = realsense_secondary.poll()
                        frame_time = time.time() - run_start
                        recorder.write(color_img, depth_raw, frame_time)
                        recorder_2.write(color_img_2, depth_raw_2, frame_time)
                        _write_rgb_frame(rgb_writer_0, color_img, rgb_size)
                        _write_rgb_frame(rgb_writer_1, color_img_2, rgb_size)
                        if node.is_traj_complete():
                            traj_complete_time_s = frame_time
                            run_log["complete_time_sec"] = float(frame_time)
                            run_log["status"] = "complete"
                            break

                    sequence_index += 1
                if stop_requested.is_set():
                    break

            for run_log in trajectory_runs:
                if run_log["status"] == "running":
                    run_log["status"] = "interrupted"

            if not stop_requested.is_set():
                if DWELL_TIME_S > 0.0:
                    print(
                        "Trajectory sequence complete; "
                        f"recording dwell for {DWELL_TIME_S:.2f} second(s)..."
                    )
                dwell_start_time_s = time.time() - run_start
                while not stop_requested.is_set():
                    color_img, depth_raw, _depth_bgr = realsense_primary.poll()
                    color_img_2, depth_raw_2, _depth_bgr_2 = realsense_secondary.poll()
                    frame_time = time.time() - run_start
                    recorder.write(color_img, depth_raw, frame_time)
                    recorder_2.write(color_img_2, depth_raw_2, frame_time)
                    _write_rgb_frame(rgb_writer_0, color_img, rgb_size)
                    _write_rgb_frame(rgb_writer_1, color_img_2, rgb_size)
                    if frame_time - dwell_start_time_s >= DWELL_TIME_S:
                        break
        except RuntimeError as exc:
            for run_log in trajectory_runs:
                if run_log["status"] == "running":
                    run_log["status"] = "stream_error"
            print(f"RealSense stream error: {exc}")
            stop_requested.set()
        finally:
            if rgb_writer_0 is not None:
                rgb_writer_0.release()
            if rgb_writer_1 is not None:
                rgb_writer_1.release()

        stop_time = _resolve_now(DEFAULT_TIMEZONE)
        trial_duration_sec = (stop_time - start_time).total_seconds()

        rgbd_payload = recorder.finalize()
        rgbd_payload_2 = recorder_2.finalize()
        robot_state_raw = _build_robot_state(node.snapshot())
        robot_state_aligned = _align_robot_state(robot_state_raw, rgbd_payload["timestamps"])
        if mocap_receiver is not None:
            mocap_raw = _build_mocap_state(mocap_receiver.snapshot())
            mocap_aligned = _align_mocap_state(mocap_raw, rgbd_payload["timestamps"])
            mocap_summary = mocap_receiver.summary()
        else:
            mocap_raw = {}
            mocap_aligned = {}
            mocap_summary = {
                "packets_received": 0,
                "decode_errors": 0,
                "rigid_body_ids": [],
                "samples_per_rigid_body": {},
                "rigid_body_names": {},
                "udp_ip": "",
                "udp_port": 0,
            }
        metadata = build_metadata(
            start_time,
            stop_time,
            DWELL_TIME_S,
            mocap_incline_deg=float(args.incline_deg),
            height_cm=float(args.height_cm),
            trajectory_sequence_request=trajectory_sequence_request,
            traj_complete_time_sec=traj_complete_time_s,
            mocap_summary=mocap_summary,
            schedule_metadata=schedule_metadata,
        )
        payload = {
            "rgb_0": rgbd_payload["rgb"],
            "depth_0": rgbd_payload["depth"],
            "camera_time_0": rgbd_payload["timestamps"],
            "rgb_1": rgbd_payload_2["rgb"],
            "depth_1": rgbd_payload_2["depth"],
            "camera_time_1": rgbd_payload_2["timestamps"],
            "depth_units": "plane_corrected_height_mm",
            "camera_serial_0": serials[0],
            "camera_serial_1": serials[1],
            "reference_session_dir": str(reference_session_dir),
            "reference_camera_dir_0": str(reference_0.reference_dir),
            "reference_camera_dir_1": str(reference_1.reference_dir),
            "trajectory_name": first_trajectory.name,
            "trajectory_points": np.asarray(first_trajectory.points, dtype=float),
            "trajectory_sequence_request": trajectory_sequence_request,
            "trajectory_runs": trajectory_runs,
            "schedule_metadata": schedule_metadata,
            "robot_state_raw": robot_state_raw,
            "robot_state": robot_state_aligned,
            "mocap_raw": mocap_raw,
            "mocap": mocap_aligned,
            "metadata": metadata,
        }

        trial_path = session_dir / f"trial_{trial_idx + 1}.npy"
        np.save(trial_path, payload, allow_pickle=True)
        print(f"Saved trial data to {trial_path}")
        trials_completed += 1
        trial_durations.append(trial_duration_sec)

    # Send one final stop command when exiting so motor control is released
    node.publish_gui_information(_build_gui_message(start_flag=0.0))
    
    realsense_primary.stop()
    realsense_secondary.stop()
    if mocap_receiver is not None:
        mocap_receiver.stop()

    stop_requested.set()
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

    session_stop_time = _resolve_now(DEFAULT_TIMEZONE)
    duration_sec = (session_stop_time - session_start_time).total_seconds()
    avg_trial_duration = float(sum(trial_durations) / len(trial_durations)) if trial_durations else 0.0
    save_session_metadata(
        session_dir,
        session_start_time,
        session_stop_time,
        args.trials,
        trials_completed,
        avg_trial_duration,
        DWELL_TIME_S,
        depth_scale_0,
        depth_scale_1,
        serials[0],
        serials[1],
        reference_session_dir,
        reference_0.reference_dir,
        reference_1.reference_dir,
        mocap_incline_deg=float(args.incline_deg),
        height_cm=float(args.height_cm),
        trajectory_sequence_request=trajectory_sequence_request,
        schedule_metadata=schedule_metadata,
    )
    print(f"Completed {trials_completed} trial(s) in {duration_sec:.1f} seconds.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# there is a small jump between trials to save stuff and to send the trajectory command again. if we need uninterrupted recording, i can chnage the whole thing so we save only one file.
# also make sure copy mocap scripts from pc
## I need to think about RPY. I think the orientatins are coupled. 
# i think ppl changed the location of cameras. need to readjust. having issue with mocap
# to expediate data collection, maybe I should make a plotting script so I dont have to wait for the full render to check the accuracy of mocap etc.
# why is the delay increasing?


# the plots for sand vs resin doesn't make sense. recalculate resin with correct COM or redo (same with steel) and observe.
# also check sand videos for validation.


# check rendered vbideos for resin and steel to see what changed vs previous time.
