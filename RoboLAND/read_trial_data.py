#!/usr/bin/env python3
"""Load RGB-D trials and torque arrays for inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


TORQUE_SCALE = 0.072
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _load_trial(path: Path) -> Dict[str, object]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected npy payload format in {path}")


def _describe_value(value: object) -> str:
    if isinstance(value, np.ndarray):
        return f"shape={value.shape} dtype={value.dtype}"
    if isinstance(value, dict):
        return f"dict keys={list(value.keys())}"
    return f"type={type(value).__name__}"


def _load_required_arrays(
    payload: Dict[str, object],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    rgb = payload.get("rgb_0") if "rgb_0" in payload else payload.get("rgb")
    depth = payload.get("depth_0") if "depth_0" in payload else payload.get("depth")
    timestamps = payload.get("camera_time_0") if "camera_time_0" in payload else payload.get("camera_time")
    rgb_2 = payload.get("rgb_1") if "rgb_1" in payload else payload.get("rgb_2")
    depth_2 = payload.get("depth_1") if "depth_1" in payload else payload.get("depth_2")
    timestamps_2 = payload.get("camera_time_1") if "camera_time_1" in payload else payload.get("timestamps_2")
    trajectory_points = payload.get("trajectory_points")
    robot_state = payload.get("robot_state")

    if not isinstance(rgb, np.ndarray):
        raise ValueError("missing or invalid 'rgb' array")
    if not isinstance(depth, np.ndarray):
        raise ValueError("missing or invalid 'depth' array")
    if not isinstance(timestamps, np.ndarray):
        raise ValueError("missing or invalid 'timestamps' array")
    if rgb_2 is not None and not isinstance(rgb_2, np.ndarray):
        raise ValueError("invalid 'rgb_2' array")
    if depth_2 is not None and not isinstance(depth_2, np.ndarray):
        raise ValueError("invalid 'depth_2' array")
    if timestamps_2 is not None and not isinstance(timestamps_2, np.ndarray):
        raise ValueError("invalid 'timestamps_2' array")
    if trajectory_points is not None and not isinstance(trajectory_points, np.ndarray):
        raise ValueError("invalid 'trajectory_points' array")
    if not isinstance(robot_state, dict):
        raise ValueError("missing or invalid 'robot_state' dict")

    robot_times = robot_state.get("time")
    right_adduction_curr = robot_state.get("rightadduction_curr")
    right_sweeping_curr = robot_state.get("rightsweeping_curr")
    if not isinstance(robot_times, np.ndarray):
        raise ValueError("robot_state missing 'time' array")
    if not isinstance(right_adduction_curr, np.ndarray):
        raise ValueError("robot_state missing 'rightadduction_curr' array")
    if not isinstance(right_sweeping_curr, np.ndarray):
        raise ValueError("robot_state missing 'rightsweeping_curr' array")

    right_adduction_pos = robot_state.get("rightadduction_pos")
    right_sweeping_pos = robot_state.get("rightsweeping_pos")
    if not isinstance(right_adduction_pos, np.ndarray):
        raise ValueError("robot_state missing 'rightadduction_pos' array")
    if not isinstance(right_sweeping_pos, np.ndarray):
        raise ValueError("robot_state missing 'rightsweeping_pos' array")

    return (
        rgb,
        depth,
        timestamps,
        robot_times,
        right_adduction_curr,
        right_sweeping_curr,
        rgb_2,
        depth_2,
        timestamps_2,
        trajectory_points,
        right_adduction_pos,
        right_sweeping_pos,
    )


def load_trial_arrays(path: Path) -> Dict[str, np.ndarray]:
    try:
        payload = _load_trial(path)
        (
            rgb,
            depth,
            timestamps,
            robot_times,
            right_adduction_curr,
            right_sweeping_curr,
            rgb_2,
            depth_2,
            timestamps_2,
            trajectory_points,
            right_adduction_pos,
            right_sweeping_pos,
        ) = _load_required_arrays(payload)
    except Exception as exc:
        raise ValueError(f"{path.name}: {exc}") from exc

    right_adduction_torque = right_adduction_curr * TORQUE_SCALE
    right_sweeping_torque = right_sweeping_curr * TORQUE_SCALE

    result = {
        "rgb_0": rgb,
        "depth_0": depth,
        "camera_time_0": timestamps,
        "robot_time": robot_times,
        "rightadduction_torque": right_adduction_torque,
        "rightsweeping_torque": right_sweeping_torque,
        "rightadduction_pos": right_adduction_pos,
        "rightsweeping_pos": right_sweeping_pos,
    }
    if rgb_2 is not None:
        result["rgb_1"] = rgb_2
    if depth_2 is not None:
        result["depth_1"] = depth_2
    if timestamps_2 is not None:
        result["camera_time_1"] = timestamps_2
    if trajectory_points is not None:
        result["trajectory_points"] = trajectory_points
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        result["trial_metadata"] = metadata
    return result


def _gather_trials(session_dir: Path) -> List[Path]:
    if session_dir.is_file():
        return [session_dir]
    return sorted(session_dir.glob("*.npy"))


def _latest_session_dir(data_root: Path) -> Path:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    sessions = [path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("session_")]
    if not sessions:
        raise FileNotFoundError(f"No session folders found under {data_root}")
    return max(sessions, key=lambda path: path.name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Load RGB-D trials and return torque arrays.")
    ap.add_argument(
        "session_dir",
        nargs="?",
        default=None,
        type=Path,
        help="Session folder containing trial npy files.",
    )
    args = ap.parse_args()

    if args.session_dir is None:
        session_dir = _latest_session_dir(DATA_ROOT)
    else:
        session_dir = Path(str(args.session_dir).strip())
    trials = _gather_trials(session_dir)
    if not trials:
        print(f"No npy trials found under {session_dir}")
        return 1

    exit_code = 0
    for trial in trials:
        try:
            arrays = load_trial_arrays(trial)
        except ValueError as exc:
            print(f"{trial.name}: ERROR loading trial - {exc}")
            exit_code = 1
            continue
        print(f"{trial.name}: loaded")
        for key, value in arrays.items():
            print(f"  {key}: {_describe_value(value)}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
