#!/usr/bin/env python3
"""Render a sync-review video with two RGB streams, two depth streams, and motor forces.

Creates an MP4 that stacks:
  - RGB camera 0 and RGB camera 1 (top row)
  - Depth camera 0 and Depth camera 1 (bottom row)
  - Force plot (right column)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required for plotting.") from exc

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _load_payload(path: Path) -> Dict[str, object]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected npy payload format in {path}")


def _trial_number(path: Path) -> int:
    match = re.match(r"trial_(\d+)\.npy$", path.name)
    if match is None:
        return 10**9
    return int(match.group(1))


def _list_trials(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    trials = [trial for trial in path.glob("trial_*.npy") if trial.is_file()]
    return sorted(trials, key=lambda trial: (_trial_number(trial), trial.name))


def _pick_trial(path: Path, trial_index: Optional[int]) -> Path:
    trials = _list_trials(path)
    if not trials:
        raise FileNotFoundError(f"No trial_*.npy files found in {path}")
    if trial_index is None:
        return trials[-1]
    if trial_index < 1 or trial_index > len(trials):
        raise ValueError(f"trial index {trial_index} out of range (1..{len(trials)})")
    return trials[trial_index - 1]


def _latest_session_dir(data_root: Path) -> Path:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    sessions = [path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("session_")]
    if not sessions:
        raise FileNotFoundError(f"No session folders found under {data_root}")
    return max(sessions, key=lambda path: path.name)


def _load_trial_arrays(
    trial_path: Path,
    force_a_key: str,
    force_b_key: str,
    torque_scale: float,
) -> Dict[str, np.ndarray]:
    payload = _load_payload(trial_path)

    rgb0 = payload.get("rgb_0") if "rgb_0" in payload else payload.get("rgb")
    depth0 = payload.get("depth_0") if "depth_0" in payload else payload.get("depth")
    t0 = payload.get("camera_time_0") if "camera_time_0" in payload else payload.get("camera_time")
    rgb1 = payload.get("rgb_1") if "rgb_1" in payload else payload.get("rgb_2")
    depth1 = payload.get("depth_1") if "depth_1" in payload else payload.get("depth_2")
    t1 = payload.get("camera_time_1") if "camera_time_1" in payload else payload.get("timestamps_2")
    robot_state = payload.get("robot_state")

    if not isinstance(rgb0, np.ndarray) or not isinstance(depth0, np.ndarray) or not isinstance(t0, np.ndarray):
        raise ValueError("missing rgb_0/depth_0/camera_time_0")
    if not isinstance(rgb1, np.ndarray) or not isinstance(depth1, np.ndarray) or not isinstance(t1, np.ndarray):
        raise ValueError("missing rgb_1/depth_1/camera_time_1")
    if not isinstance(robot_state, dict):
        raise ValueError("missing robot_state")

    force_a = robot_state.get(force_a_key)
    force_b = robot_state.get(force_b_key)
    robot_time = robot_state.get("time")
    if not isinstance(force_a, np.ndarray) or not isinstance(force_b, np.ndarray) or not isinstance(robot_time, np.ndarray):
        raise ValueError("robot_state missing force keys or time array")

    if len(rgb0) != len(t0):
        raise ValueError("rgb_0 and camera_time_0 length mismatch")
    if len(rgb1) != len(t1):
        raise ValueError("rgb_1 and camera_time_1 length mismatch")
    if len(robot_time) != len(t0):
        raise ValueError("robot_time and camera_time_0 length mismatch")
    if len(force_a) != len(robot_time) or len(force_b) != len(robot_time):
        raise ValueError("force arrays and robot_time length mismatch")

    return {
        "rgb0": rgb0,
        "depth0": depth0,
        "t0": np.asarray(t0, dtype=float),
        "rgb1": rgb1,
        "depth1": depth1,
        "t1": np.asarray(t1, dtype=float),
        "force_a": np.asarray(force_a, dtype=float) * float(torque_scale),
        "force_b": np.asarray(force_b, dtype=float) * float(torque_scale),
        "robot_time": np.asarray(robot_time, dtype=float),
    }


def _closest_indices(sample_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
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


def _depth_values(depth: np.ndarray, max_frames: int = 50) -> np.ndarray:
    if depth.size == 0:
        return np.empty((0,), dtype=depth.dtype)
    frames = depth
    if depth.ndim == 3 and depth.shape[0] > max_frames:
        idx = np.linspace(0, depth.shape[0] - 1, max_frames, dtype=int)
        frames = depth[idx]
    values = frames.reshape(-1)
    return values[values > 0]


def _estimate_shared_depth_range(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    max_frames: int = 50,
) -> Tuple[float, float]:
    values_a = _depth_values(depth_a, max_frames)
    values_b = _depth_values(depth_b, max_frames)
    if values_a.size == 0 and values_b.size == 0:
        return 0.0, 1.0
    if values_a.size == 0:
        values = values_b
    elif values_b.size == 0:
        values = values_a
    else:
        values = np.concatenate([values_a, values_b])
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _estimate_depth_range_for_trials(trials: List[Dict[str, np.ndarray]]) -> Tuple[float, float]:
    if not trials:
        return 0.0, 1.0
    lows: List[float] = []
    highs: List[float] = []
    for trial in trials:
        low, high = _estimate_shared_depth_range(trial["depth0"], trial["depth1"])
        lows.append(low)
        highs.append(high)
    low = min(lows)
    high = max(highs)
    if high <= low:
        high = low + 1.0
    return low, high


def _estimate_fps_from_trials(trials: List[Dict[str, np.ndarray]]) -> float:
    deltas: List[np.ndarray] = []
    for trial in trials:
        t0 = trial["t0"]
        if len(t0) < 2:
            continue
        dt = np.diff(t0)
        dt = dt[dt > 0]
        if dt.size > 0:
            deltas.append(dt)
    if not deltas:
        return 30.0
    median_dt = float(np.median(np.concatenate(deltas)))
    if median_dt <= 0.0:
        return 30.0
    return float(np.clip(1.0 / median_dt, 1.0, 240.0))


def _median_positive_step(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    dt = np.diff(times)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    return float(np.median(dt))


def _build_experiment_force_timeline(
    trials: List[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    all_times: List[np.ndarray] = []
    all_force_a: List[np.ndarray] = []
    all_force_b: List[np.ndarray] = []
    all_time_diffs: List[np.ndarray] = []
    trial_idx_offsets: List[int] = []

    running_time_offset = 0.0
    running_idx_offset = 0

    for trial in trials:
        robot_time_local = trial["robot_time"]
        frame_time_local = trial["t0"]
        force_a_local = trial["force_a"]
        force_b_local = trial["force_b"]

        if len(robot_time_local) == 0:
            trial_idx_offsets.append(running_idx_offset)
            continue

        robot_time_rel = robot_time_local - float(robot_time_local[0])
        frame_time_rel = frame_time_local - float(frame_time_local[0]) if len(frame_time_local) else frame_time_local.copy()
        robot_time_global = robot_time_rel + running_time_offset
        frame_time_global = frame_time_rel + running_time_offset

        robot_idx_for_frame = _closest_indices(robot_time_global, frame_time_global)
        time_diffs_global = robot_time_global[robot_idx_for_frame] - frame_time_global

        trial_idx_offsets.append(running_idx_offset)
        running_idx_offset += len(robot_time_global)

        all_times.append(robot_time_global)
        all_force_a.append(force_a_local)
        all_force_b.append(force_b_local)
        all_time_diffs.append(time_diffs_global)

        trial_span = max(float(frame_time_rel[-1]), float(robot_time_rel[-1])) if len(frame_time_rel) else float(robot_time_rel[-1])
        gap = max(_median_positive_step(frame_time_rel), _median_positive_step(robot_time_rel))
        running_time_offset += trial_span + gap

    if not all_times:
        return (
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            trial_idx_offsets,
        )

    return (
        np.concatenate(all_times),
        np.concatenate(all_force_a),
        np.concatenate(all_force_b),
        np.concatenate(all_time_diffs),
        trial_idx_offsets,
    )


def _colorize_depth(depth_raw: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    depth = depth_raw.astype(np.float32)
    depth = np.clip((depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
    depth_u8 = (depth * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def _torque_label_from_key(signal_key: str) -> str:
    if "curr" in signal_key:
        return signal_key.replace("curr", "torque")
    return signal_key


def _plot_forces(
    times: np.ndarray,
    force_a: np.ndarray,
    force_b: np.ndarray,
    idx: int,
    width: int,
    height: int,
    label_a: str,
    label_b: str,
    time_diffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    if time_diffs is None:
        fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
        ax.plot(times[: idx + 1], force_a[: idx + 1], color="tab:red", linewidth=1.5, label=label_a)
        ax.plot(times[: idx + 1], force_b[: idx + 1], color="tab:blue", linewidth=1.5, label=label_b)
        ax.set_title("Motor Torque")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("torque (scaled)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    else:
        fig, (ax_force, ax_diff) = plt.subplots(
            2,
            1,
            figsize=(width / 100.0, height / 100.0),
            dpi=100,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        ax_force.plot(times[: idx + 1], force_a[: idx + 1], color="tab:red", linewidth=1.5, label=label_a)
        ax_force.plot(times[: idx + 1], force_b[: idx + 1], color="tab:blue", linewidth=1.5, label=label_b)
        ax_force.set_title("Motor Torque")
        ax_force.set_ylabel("torque (scaled)")
        ax_force.legend(loc="upper right")
        ax_force.grid(True, alpha=0.3)

        diffs = np.abs(time_diffs[: idx + 1])
        ax_diff.plot(times[: idx + 1], diffs, color="tab:green", linewidth=1.2, label="|t_torque - t_frame|")
        ax_diff.set_ylabel("Δt (s)")
        ax_diff.set_xlabel("time (s)")
        ax_diff.grid(True, alpha=0.3)
        ax_diff.legend(loc="upper right")
    fig.tight_layout()
    fig.canvas.draw()
    try:
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8)[..., :3]
    except Exception:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def _render_trial_frames(
    writer: cv2.VideoWriter,
    trial: Dict[str, np.ndarray],
    depth_min: float,
    depth_max: float,
    plot_w: int,
    plot_h: int,
    force_a_label: str,
    force_b_label: str,
    trial_caption: Optional[str] = None,
    plot_times: Optional[np.ndarray] = None,
    plot_force_a: Optional[np.ndarray] = None,
    plot_force_b: Optional[np.ndarray] = None,
    plot_time_diffs: Optional[np.ndarray] = None,
    plot_idx_offset: int = 0,
) -> float:
    rgb0 = trial["rgb0"]
    depth0 = trial["depth0"]
    t0 = trial["t0"]
    rgb1 = trial["rgb1"]
    depth1 = trial["depth1"]
    t1 = trial["t1"]
    force_a = trial["force_a"]
    force_b = trial["force_b"]
    robot_time = trial["robot_time"]

    idx1_for_t0 = _closest_indices(t1, t0)
    robot_idx_for_t0 = _closest_indices(robot_time, t0)
    time_diffs = robot_time[robot_idx_for_t0] - t0
    plot_times_final = robot_time if plot_times is None else plot_times
    plot_force_a_final = force_a if plot_force_a is None else plot_force_a
    plot_force_b_final = force_b if plot_force_b is None else plot_force_b
    plot_time_diffs_final = time_diffs if plot_time_diffs is None else plot_time_diffs
    max_diff = float(np.max(np.abs(time_diffs))) if time_diffs.size else 0.0

    for i in range(len(t0)):
        idx1 = int(idx1_for_t0[i])
        idx_robot = int(robot_idx_for_t0[i])
        idx_plot = int(plot_idx_offset + idx_robot)

        rgb0_img = rgb0[i]
        rgb1_img = rgb1[idx1]
        depth0_img = _colorize_depth(depth0[i], depth_min, depth_max)
        depth1_img = _colorize_depth(depth1[idx1], depth_min, depth_max)

        top = np.hstack([rgb0_img, rgb1_img])
        bottom = np.hstack([depth0_img, depth1_img])
        grid = np.vstack([top, bottom])

        plot_img = _plot_forces(
            plot_times_final,
            plot_force_a_final,
            plot_force_b_final,
            idx_plot,
            plot_w,
            plot_h,
            force_a_label,
            force_b_label,
            time_diffs=plot_time_diffs_final,
        )
        plot_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

        canvas = np.hstack([grid, plot_bgr])
        if trial_caption:
            cv2.putText(canvas, trial_caption, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, trial_caption, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)

    return max_diff


def main() -> None:
    ap = argparse.ArgumentParser(description="Render sync-review MP4 for RGB-D + forces.")
    ap.add_argument(
        "session_or_trial",
        nargs="?",
        default=None,
        type=Path,
        help="Session directory or trial .npy file (default: latest session).",
    )
    ap.add_argument(
        "--mode",
        choices=["trial", "experiment"],
        default="trial",
        help="Render one trial or concatenate all trials in a session folder.",
    )
    ap.add_argument("--trial", type=int, default=None, help="Trial index (1-based) when a session dir is provided")
    ap.add_argument("--output", type=Path, default=None, help="Output mp4 path")
    ap.add_argument("--force-a", default="rightadduction_curr", help="robot_state key for force A")
    ap.add_argument("--force-b", default="rightsweeping_curr", help="robot_state key for force B")
    ap.add_argument("--torque-scale", type=float, default=0.072, help="Scale forces by this factor")
    args = ap.parse_args()

    source_path = _latest_session_dir(DATA_ROOT) if args.session_or_trial is None else Path(str(args.session_or_trial).strip())
    if args.mode == "trial":
        trial_paths = [_pick_trial(source_path, args.trial)]
    else:
        if args.trial is not None:
            raise SystemExit("--trial can only be used with --mode trial.")
        session_dir = source_path.parent if source_path.is_file() else source_path
        trial_paths = _list_trials(session_dir)
        if not trial_paths:
            raise SystemExit(f"No trial_*.npy files found in {session_dir}")

    trials: List[Dict[str, np.ndarray]] = []
    for trial_path in trial_paths:
        try:
            trials.append(_load_trial_arrays(trial_path, args.force_a, args.force_b, args.torque_scale))
        except ValueError as exc:
            raise SystemExit(f"{trial_path.name}: {exc}") from exc

    if not trials:
        raise SystemExit("No trials to render.")

    first_rgb = trials[0]["rgb0"]
    if first_rgb.ndim != 4 or first_rgb.shape[-1] != 3:
        raise SystemExit(f"{trial_paths[0].name}: rgb_0 expected shape (N,H,W,3), got {first_rgb.shape}")
    h, w = int(first_rgb.shape[1]), int(first_rgb.shape[2])

    for trial_path, trial in zip(trial_paths, trials):
        rgb0 = trial["rgb0"]
        rgb1 = trial["rgb1"]
        depth0 = trial["depth0"]
        depth1 = trial["depth1"]
        if rgb0.ndim != 4 or rgb0.shape[-1] != 3:
            raise SystemExit(f"{trial_path.name}: rgb_0 expected shape (N,H,W,3), got {rgb0.shape}")
        if rgb1.ndim != 4 or rgb1.shape[-1] != 3:
            raise SystemExit(f"{trial_path.name}: rgb_1 expected shape (N,H,W,3), got {rgb1.shape}")
        if depth0.ndim != 3:
            raise SystemExit(f"{trial_path.name}: depth_0 expected shape (N,H,W), got {depth0.shape}")
        if depth1.ndim != 3:
            raise SystemExit(f"{trial_path.name}: depth_1 expected shape (N,H,W), got {depth1.shape}")
        if rgb0.shape[1:3] != (h, w) or rgb1.shape[1:3] != (h, w):
            raise SystemExit(f"{trial_path.name}: RGB resolution mismatch, expected ({h},{w})")
        if depth0.shape[1:3] != (h, w) or depth1.shape[1:3] != (h, w):
            raise SystemExit(f"{trial_path.name}: depth resolution mismatch, expected ({h},{w})")

    plot_w, plot_h = w, h * 2
    canvas_w, canvas_h = w * 3, h * 2

    if args.output is not None:
        output_path = args.output
    elif args.mode == "trial":
        output_path = trial_paths[0].parent / f"{trial_paths[0].stem}_sync.mp4"
    else:
        output_path = trial_paths[0].parent / f"{trial_paths[0].parent.name}_sync.mp4"

    depth_min, depth_max = _estimate_depth_range_for_trials(trials)
    fps = _estimate_fps_from_trials(trials)
    torque_label_a = _torque_label_from_key(args.force_a)
    torque_label_b = _torque_label_from_key(args.force_b)
    experiment_plot_times: Optional[np.ndarray] = None
    experiment_plot_force_a: Optional[np.ndarray] = None
    experiment_plot_force_b: Optional[np.ndarray] = None
    experiment_plot_time_diffs: Optional[np.ndarray] = None
    experiment_plot_idx_offsets: List[int] = []
    if args.mode == "experiment":
        (
            experiment_plot_times,
            experiment_plot_force_a,
            experiment_plot_force_b,
            experiment_plot_time_diffs,
            experiment_plot_idx_offsets,
        ) = _build_experiment_force_timeline(trials)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise SystemExit("Failed to open output video writer.")

    max_diff = 0.0
    try:
        for trial_idx, trial in enumerate(trials, start=1):
            caption = None
            if len(trials) > 1:
                caption = f"Trial {trial_idx}/{len(trials)} ({trial_paths[trial_idx - 1].name})"
            plot_idx_offset = 0
            if args.mode == "experiment":
                plot_idx_offset = experiment_plot_idx_offsets[trial_idx - 1]
            trial_max_diff = _render_trial_frames(
                writer,
                trial,
                depth_min,
                depth_max,
                plot_w,
                plot_h,
                torque_label_a,
                torque_label_b,
                trial_caption=caption,
                plot_times=experiment_plot_times,
                plot_force_a=experiment_plot_force_a,
                plot_force_b=experiment_plot_force_b,
                plot_time_diffs=experiment_plot_time_diffs,
                plot_idx_offset=plot_idx_offset,
            )
            max_diff = max(max_diff, trial_max_diff)
    finally:
        writer.release()

    print(f"Mode: {args.mode}")
    print(f"Trials rendered: {len(trials)}")
    print(f"Saved {output_path}")
    print(f"Max |t_force - t_frame|: {max_diff:.6f} s")


if __name__ == "__main__":
    main()
