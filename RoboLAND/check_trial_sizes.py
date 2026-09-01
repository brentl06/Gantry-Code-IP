#!/usr/bin/env python3
"""Check trial npy payload sizes for RGB-D data and robot telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480


def _load_trial(path: Path) -> Dict[str, object]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected npy payload format in {path}")


def _describe_array(arr: np.ndarray) -> str:
    return f"shape={arr.shape} dtype={arr.dtype}"


def _print_overview(path: Path, payload: Dict[str, object]) -> None:
    print(f"{path.name}: overview")
    keys = sorted(payload.keys())
    print(f"  keys: {', '.join(keys)}")

    rgb = payload.get("rgb")
    depth = payload.get("depth")
    timestamps = payload.get("timestamps")
    if isinstance(rgb, np.ndarray):
        print(f"  rgb: {_describe_array(rgb)}")
    if isinstance(depth, np.ndarray):
        print(f"  depth: {_describe_array(depth)}")
    if isinstance(timestamps, np.ndarray):
        print(f"  timestamps: {_describe_array(timestamps)}")

    robot_state = payload.get("robot_state")
    if isinstance(robot_state, dict):
        rs_keys = sorted(robot_state.keys())
        print(f"  robot_state keys: {len(rs_keys)}")
        for sample_key in ("time", "turtle_state", "rightadduction_curr", "rightsweeping_curr"):
            value = robot_state.get(sample_key)
            if isinstance(value, np.ndarray):
                print(f"  robot_state['{sample_key}']: {_describe_array(value)}")

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        meta_keys = sorted(metadata.keys())
        print(f"  metadata keys: {', '.join(meta_keys)}")


def _check_rgbd(payload: Dict[str, object], height: int, width: int) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    rgb = payload.get("rgb")
    depth = payload.get("depth")
    timestamps = payload.get("timestamps")

    if not isinstance(rgb, np.ndarray):
        errors.append("missing or invalid 'rgb' array")
    if not isinstance(depth, np.ndarray):
        errors.append("missing or invalid 'depth' array")
    if not isinstance(timestamps, np.ndarray):
        errors.append("missing or invalid 'timestamps' array")

    if errors:
        return errors, warnings

    if rgb.ndim != 4 or rgb.shape[1] != height or rgb.shape[2] != width or rgb.shape[3] != 3:
        warnings.append(f"rgb unexpected size: {_describe_array(rgb)} (expected N,{height},{width},3)")

    if depth.ndim != 3 or depth.shape[1] != height or depth.shape[2] != width:
        warnings.append(f"depth unexpected size: {_describe_array(depth)} (expected N,{height},{width})")

    if timestamps.ndim != 1:
        warnings.append(f"timestamps unexpected size: {_describe_array(timestamps)} (expected N,)")

    n_rgb = rgb.shape[0]
    n_depth = depth.shape[0]
    n_ts = len(timestamps)
    if n_rgb != n_depth or n_rgb != n_ts:
        warnings.append(f"frame count mismatch: rgb={n_rgb}, depth={n_depth}, timestamps={n_ts}")

    return errors, warnings


def _check_robot_state(payload: Dict[str, object]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    robot_state = payload.get("robot_state")
    if not isinstance(robot_state, dict):
        errors.append("missing or invalid 'robot_state' dict")
        return errors, warnings

    lengths: Dict[int, List[str]] = {}
    for key, value in robot_state.items():
        if not isinstance(value, np.ndarray):
            warnings.append(f"robot_state['{key}'] is not a numpy array")
            continue
        length = value.shape[0] if value.ndim >= 1 else 0
        lengths.setdefault(length, []).append(key)

    if not lengths:
        warnings.append("robot_state arrays are empty or missing")
        return errors, warnings

    if len(lengths) > 1:
        parts = [f"{length}: {', '.join(sorted(keys))}" for length, keys in sorted(lengths.items())]
        warnings.append("robot_state length mismatch across keys: " + " | ".join(parts))

    if "time" not in robot_state:
        warnings.append("robot_state missing 'time' array")

    return errors, warnings


def _check_metadata(payload: Dict[str, object]) -> List[str]:
    warnings: List[str] = []
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        warnings.append("missing or invalid 'metadata' dict")
        return warnings

    for key in ("trajectory_points", "start_time", "stop_time", "duration_sec"):
        if key not in metadata:
            warnings.append(f"metadata missing '{key}'")

    return warnings


def check_trial(path: Path, height: int, width: int) -> int:
    try:
        payload = _load_trial(path)
    except Exception as exc:
        print(f"{path.name}: ERROR loading npy - {exc}")
        return 1

    _print_overview(path, payload)

    errors: List[str] = []
    warnings: List[str] = []

    err, warn = _check_rgbd(payload, height, width)
    errors.extend(err)
    warnings.extend(warn)

    err, warn = _check_robot_state(payload)
    errors.extend(err)
    warnings.extend(warn)

    warnings.extend(_check_metadata(payload))

    status = "OK" if not errors and not warnings else "CHECK"
    print(f"{path.name}: {status}")

    for msg in errors:
        print(f"  ERROR: {msg}")
    for msg in warnings:
        print(f"  WARN: {msg}")

    return 1 if errors else 0


def _gather_trials(session_dir: Path) -> List[Path]:
    if session_dir.is_file():
        return [session_dir]
    return sorted(session_dir.glob("*.npy"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Check trial npy payload sizes.")
    ap.add_argument(
        "session_dir",
        nargs="?",
        default=Path(__file__).resolve().parents[2] / "data" / "session_preliminary",
        type=Path,
        help="Session folder containing trial npy files (default: data/session_preliminary)",
    )
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Expected RGB-D frame width")
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Expected RGB-D frame height")
    args = ap.parse_args()

    session_dir = Path(str(args.session_dir).strip())
    trials = _gather_trials(session_dir)
    if not trials:
        print(f"No npy trials found under {session_dir}")
        return 1

    exit_code = 0
    for trial in trials:
        exit_code |= check_trial(trial, args.height, args.width)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
