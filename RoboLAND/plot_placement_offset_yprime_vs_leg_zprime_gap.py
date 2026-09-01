#!/usr/bin/env python3
"""Plot obstacle y' vs leg-relative z' gap.

Definitions used by this script:
  - obstacle_y_prime = y'(t)
  - leg_z_prime_gap = |z'_obstacle(t) - z'_leg|
"""
# usage
# python3 highlevel/terrain_manipulation/src/utils/plot_placement_offset_yprime_vs_leg_zprime_gap.py   --fixed-y-range-mm 150 250   --x-axis-label "Absolute vertical separation from leg, |z' - z'_leg| (mm)"   --y-axis-label "Vertical obstacle position y' (mm)"   --output-dir highlevel/terrain_manipulation/data/obstacle_yprime_vs_leg_zprime_gap_both_plots


from __future__ import annotations

import argparse
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Avoid matplotlib cache permission issues when default cache path is read-only.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
M_TO_MM = 1000.0

MOCAP_RB_IDS_BY_KIND: Dict[str, int] = {
    "empty": 2,
    "lead": 3,
    "resin": 5,
    "steel": 6,
    "sand": 8,
}
DEFAULT_MOCAP_KIND = "steel"

DEFAULT_LEG_X_MM = 147.16
DEFAULT_LEG_Y_MM = -98.43
DEFAULT_LEG_Z_MM = 809.53
DEFAULT_INCLINE_DEG = 23.0
DEFAULT_Y_AXIS_LABEL = "Obstacle position y' (mm)"
DEFAULT_X_AXIS_LABEL = "Absolute vertical gap to leg, |z' - z'_leg| (mm)"
DEFAULT_DELTA_Y_AXIS_LABEL = "Trial vertical displacement, Δy' (mm)"
DEFAULT_PLOT_MODE = "default"
PAPER_FONT_SCALE = 3

# Hardcoded experiment groups for per-object plotting (same sessions used by
# displacement-vs-density plotting scripts).
DEFAULT_SESSIONS_BY_KIND: Mapping[str, Sequence[str]] = OrderedDict(
    [
        (
            "empty",
            (
                "session_20260313_111621",
                "session_20260313_112741",
                "session_20260313_125836",
                "session_20260313_121725",
                # "session_20260313_123622",
            ),
        ),
        (
            "lead",
            (
                "session_20260314_114731",
                "session_20260314_115216",
                "session_20260314_115730",
                "session_20260314_120728",
                "session_20260314_121349",
                "session_20260314_121907",
            ),
        ),
        (
            "steel",
            (
                "session_20260319_134621",
                "session_20260319_135104",
                "session_20260319_140118",
                "session_20260319_140705",
                "session_20260319_141232",
                "session_20260319_141641",
            ),
        ),
        (
            "resin",
            (
                "session_20260319_142653",
                "session_20260319_145057",
                "session_20260319_145447",
                "session_20260319_145955",
                # "session_20260319_151418",
                # "session_20260319_151845",
                "session_20260319_152329",
            ),
        ),
        (
            "sand",
            (
                "session_20260317_153944",
                "session_20260317_154742",
                "session_20260317_155128",
                "session_20260317_155550",
                "session_20260317_160026",
                "session_20260317_160410",
            ),
        ),
    ]
)


def _load_payload(path: Path) -> Dict[str, object]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        payload = data.item()
        if isinstance(payload, dict):
            return payload
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected npy payload format in {path}")


def _trial_number(path: Path) -> Optional[int]:
    match = re.match(r"trial_(\d+)\.npy$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _list_trials(session_dir: Path) -> List[Path]:
    trials = [p for p in session_dir.glob("trial_*.npy") if p.is_file()]
    return sorted(trials, key=lambda p: (_trial_number(p) or 10**9, p.name))


def _latest_session_dir(data_root: Path) -> Path:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    sessions = [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("session_")]
    if not sessions:
        raise FileNotFoundError(f"No session folders found under {data_root}")
    return max(sessions, key=lambda p: p.name)


def _as_float_array(value: object) -> Optional[np.ndarray]:
    if not isinstance(value, np.ndarray):
        return None
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _available_mocap_ids(payload: Dict[str, object]) -> List[int]:
    ids = set()
    for container_key in ("mocap_raw", "mocap"):
        mocap = payload.get(container_key)
        if not isinstance(mocap, dict):
            continue
        for key in mocap.keys():
            try:
                ids.add(int(key))
            except (TypeError, ValueError):
                continue
    return sorted(ids)


def _select_mocap_rb_id(payload: Dict[str, object], requested_rb_id: Optional[int], trial_path: Path) -> int:
    ids = _available_mocap_ids(payload)
    if requested_rb_id is not None:
        if requested_rb_id not in ids:
            raise ValueError(
                f"{trial_path.name}: requested RB ID {requested_rb_id} not found; available IDs: {ids}"
            )
        return requested_rb_id
    if len(ids) == 1:
        return ids[0]
    raise ValueError(
        f"{trial_path.name}: multiple mocap RB IDs present ({ids}); "
        "provide --mocap-rb-id or --mocap-kind"
    )


def _get_mocap_state(payload: Dict[str, object], rb_id: int) -> Optional[Dict[str, object]]:
    for container_key in ("mocap_raw", "mocap"):
        mocap = payload.get(container_key)
        if not isinstance(mocap, dict):
            continue
        state = mocap.get(str(rb_id), mocap.get(rb_id))
        if isinstance(state, dict):
            return state
    return None


def _require_series(mocap_state: Dict[str, object], key: str) -> np.ndarray:
    arr = _as_float_array(mocap_state.get(key))
    if arr is None:
        raise ValueError(f"missing required mocap array: {key}")
    return arr


def _rotate_xyz_by_incline_about_x(
    x: np.ndarray | float,
    y: np.ndarray | float,
    z: np.ndarray | float,
    incline_deg: float,
) -> Tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    # Matches data collection frame rotation:
    # clockwise rotation in y-z plane == right-handed rotation around x by -incline.
    theta = math.radians(float(incline_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    x_rot = x
    y_rot = c * y + s * z
    z_rot = -s * y + c * z
    return x_rot, y_rot, z_rot


def _binned_mean_std(x: np.ndarray, y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0 or bins < 2:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    edges = np.linspace(x_min, x_max, int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full((bins,), np.nan, dtype=float)
    stds = np.full((bins,), np.nan, dtype=float)

    for i in range(bins):
        if i < bins - 1:
            m = (x >= edges[i]) & (x < edges[i + 1])
        else:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        if np.count_nonzero(m) < 2:
            continue
        vals = y[m]
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue
        means[i] = float(np.mean(vals))
        stds[i] = float(np.std(vals))

    valid = np.isfinite(means) & np.isfinite(stds) & np.isfinite(centers)
    return centers[valid], means[valid], stds[valid]


def _percentile_axis_limits(
    values: np.ndarray,
    low_pct: float,
    high_pct: float,
    pad_frac: float = 0.08,
) -> Optional[Tuple[float, float]]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        center = 0.5 * (hi + lo)
        return center - 1.0, center + 1.0
    pad = (hi - lo) * float(pad_frac)
    return lo - pad, hi + pad


def _collect_xy_mm(
    session_dirs: Sequence[Path],
    requested_rb_id: int,
    leg_z_prime_m: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    Dict[int, Dict[str, float]],
    Dict[str, Dict[int, Dict[str, float]]],
]:
    x_all_mm: List[np.ndarray] = []
    y_all_mm: List[np.ndarray] = []
    total_trials = 0
    total_points = 0
    per_trial_x_mm: Dict[int, List[np.ndarray]] = {}
    per_trial_y_mm: Dict[int, List[np.ndarray]] = {}
    per_session_trial_x_mm: Dict[str, Dict[int, List[np.ndarray]]] = {}
    per_session_trial_y_mm: Dict[str, Dict[int, List[np.ndarray]]] = {}

    for session_dir in session_dirs:
        session_key = session_dir.name
        trials = _list_trials(session_dir)
        if not trials:
            print(f"{session_dir.name}: no trial_*.npy files found; skipping.")
            continue

        for trial_path in trials:
            trial_num = _trial_number(trial_path)
            if trial_num is None:
                continue
            payload = _load_payload(trial_path)
            rb_id = _select_mocap_rb_id(payload, requested_rb_id=requested_rb_id, trial_path=trial_path)
            mocap_state = _get_mocap_state(payload, rb_id)
            if mocap_state is None:
                continue

            y_prime = _require_series(mocap_state, "rotated_position_y")
            z_prime = _require_series(mocap_state, "rotated_position_z")
            if len(y_prime) != len(z_prime):
                raise ValueError(
                    f"{session_dir.name}/{trial_path.name}: rotated y'/z' length mismatch "
                    f"(len(y')={len(y_prime)}, len(z')={len(z_prime)})"
                )

            obstacle_y_prime = y_prime
            leg_z_prime_gap = np.abs(z_prime - float(leg_z_prime_m))
            finite = np.isfinite(obstacle_y_prime) & np.isfinite(leg_z_prime_gap)
            if not np.any(finite):
                continue

            x_vals_mm = leg_z_prime_gap[finite] * M_TO_MM
            y_vals_mm = obstacle_y_prime[finite] * M_TO_MM
            x_all_mm.append(x_vals_mm)
            y_all_mm.append(y_vals_mm)
            per_trial_x_mm.setdefault(trial_num, []).append(x_vals_mm)
            per_trial_y_mm.setdefault(trial_num, []).append(y_vals_mm)
            per_session_trial_x_mm.setdefault(session_key, {}).setdefault(trial_num, []).append(x_vals_mm)
            per_session_trial_y_mm.setdefault(session_key, {}).setdefault(trial_num, []).append(y_vals_mm)
            total_trials += 1
            total_points += int(np.count_nonzero(finite))

    trial_summary: Dict[int, Dict[str, float]] = {}
    for trial_num in sorted(per_trial_x_mm.keys()):
        x_vals_mm = np.concatenate(per_trial_x_mm[trial_num]) if per_trial_x_mm[trial_num] else np.empty((0,), dtype=float)
        y_vals_mm = np.concatenate(per_trial_y_mm.get(trial_num, [])) if per_trial_y_mm.get(trial_num) else np.empty((0,), dtype=float)
        finite = np.isfinite(x_vals_mm) & np.isfinite(y_vals_mm)
        x_vals_mm = x_vals_mm[finite]
        y_vals_mm = y_vals_mm[finite]
        if x_vals_mm.size < 5 or y_vals_mm.size < 5:
            continue
        trial_summary[trial_num] = {
            "x_med_mm": float(np.median(x_vals_mm)),
            "y_med_mm": float(np.median(y_vals_mm)),
            "x_min_mm": float(np.min(x_vals_mm)),
            "x_max_mm": float(np.max(x_vals_mm)),
            "n_points": float(x_vals_mm.size),
        }

    session_trial_summary: Dict[str, Dict[int, Dict[str, float]]] = {}
    for session_key, trials_x in per_session_trial_x_mm.items():
        out_trials: Dict[int, Dict[str, float]] = {}
        for trial_num, x_chunks in trials_x.items():
            y_chunks = per_session_trial_y_mm.get(session_key, {}).get(trial_num, [])
            if not x_chunks or not y_chunks:
                continue
            x_vals_mm = np.concatenate(x_chunks)
            y_vals_mm = np.concatenate(y_chunks)
            finite = np.isfinite(x_vals_mm) & np.isfinite(y_vals_mm)
            x_vals_mm = x_vals_mm[finite]
            y_vals_mm = y_vals_mm[finite]
            if x_vals_mm.size < 5 or y_vals_mm.size < 5:
                continue
            out_trials[trial_num] = {
                "x_med_mm": float(np.median(x_vals_mm)),
                "y_med_mm": float(np.median(y_vals_mm)),
                "x_first_mm": float(x_vals_mm[0]),
                "x_last_mm": float(x_vals_mm[-1]),
                "y_first_mm": float(y_vals_mm[0]),
                "y_last_mm": float(y_vals_mm[-1]),
                "n_points": float(x_vals_mm.size),
            }
        if out_trials:
            session_trial_summary[session_key] = out_trials

    if not x_all_mm:
        return (
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            total_trials,
            total_points,
            trial_summary,
            session_trial_summary,
        )
    return (
        np.concatenate(x_all_mm),
        np.concatenate(y_all_mm),
        total_trials,
        total_points,
        trial_summary,
        session_trial_summary,
    )


def _compute_trial_segments(
    trial_summary: Dict[int, Dict[str, float]],
    x_data_min: float,
    x_data_max: float,
) -> Tuple[List[int], Dict[int, Tuple[float, float]], np.ndarray]:
    ordered_trials = sorted(trial_summary.keys())
    if not ordered_trials:
        return [], {}, np.empty((0,), dtype=float)
    x_meds = np.asarray([float(trial_summary[t]["x_med_mm"]) for t in ordered_trials], dtype=float)
    if x_meds.size == 1:
        return ordered_trials, {ordered_trials[0]: (x_data_min, x_data_max)}, np.empty((0,), dtype=float)

    boundaries = 0.5 * (x_meds[:-1] + x_meds[1:])
    increasing = bool(x_meds[-1] >= x_meds[0])
    if increasing:
        for i in range(1, boundaries.size):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] + 1e-6
    else:
        for i in range(1, boundaries.size):
            if boundaries[i] >= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] - 1e-6

    segments: Dict[int, Tuple[float, float]] = {}
    for idx, trial_num in enumerate(ordered_trials):
        if increasing:
            x_lo = x_data_min if idx == 0 else float(boundaries[idx - 1])
            x_hi = x_data_max if idx == len(ordered_trials) - 1 else float(boundaries[idx])
        else:
            x_hi = x_data_max if idx == 0 else float(boundaries[idx - 1])
            x_lo = x_data_min if idx == len(ordered_trials) - 1 else float(boundaries[idx])
        if x_hi < x_lo:
            x_lo, x_hi = x_hi, x_lo
        segments[trial_num] = (x_lo, x_hi)
    return ordered_trials, segments, np.asarray(boundaries, dtype=float)


def _compute_trial_delta_y_stats(
    trial_summary: Dict[int, Dict[str, float]],
    session_trial_summary: Dict[str, Dict[int, Dict[str, float]]],
    x_data_min: float,
    x_data_max: float,
) -> List[Dict[str, float]]:
    ordered_trials, segments, boundaries = _compute_trial_segments(trial_summary, x_data_min, x_data_max)
    if not ordered_trials:
        return []

    stats: List[Dict[str, float]] = []
    for idx, trial_num in enumerate(ordered_trials):
        deltas: List[float] = []
        for _session_key, per_trial in session_trial_summary.items():
            cur_stats = per_trial.get(trial_num)
            if cur_stats is None:
                continue
            y_first = float(cur_stats.get("y_first_mm", math.nan))
            y_last = float(cur_stats.get("y_last_mm", math.nan))
            if np.isfinite(y_first) and np.isfinite(y_last):
                deltas.append(y_last - y_first)
        if not deltas:
            continue
        delta_arr = np.asarray(deltas, dtype=float)
        delta_arr = delta_arr[np.isfinite(delta_arr)]
        if delta_arr.size == 0:
            continue
        seg = segments.get(trial_num)
        if seg is not None:
            x_start, x_end = float(seg[0]), float(seg[1])
        else:
            if idx == 0:
                x_boundary = float(boundaries[0]) if boundaries.size else float(trial_summary[trial_num]["x_med_mm"])
            elif idx - 1 < boundaries.size:
                x_boundary = float(boundaries[idx - 1])
            else:
                x_boundary = float(trial_summary[trial_num]["x_med_mm"])
            x_start = x_boundary
            x_end = x_boundary
        x_lo = min(x_start, x_end)
        x_hi = max(x_start, x_end)
        x_center = 0.5 * (x_lo + x_hi)
        stats.append(
            {
                "x_boundary_mm": x_center,
                "x_span_lo_mm": x_lo,
                "x_span_hi_mm": x_hi,
                "delta_y_mean_mm": float(np.mean(delta_arr)),
                "delta_y_std_mm": float(np.std(delta_arr)) if delta_arr.size > 1 else 0.0,
                "n_sessions": float(delta_arr.size),
                "trial_num": float(trial_num),
            }
        )
    return stats


def _write_plot(
    x: np.ndarray,
    y: np.ndarray,
    trial_summary: Dict[int, Dict[str, float]],
    trial_delta_stats: List[Dict[str, float]],
    bins: int,
    point_size: float,
    point_alpha: float,
    show_trial_overlays: bool,
    x_axis_label: str,
    y_axis_label: str,
    x_lim_mm: Optional[Tuple[float, float]],
    y_lim_mm: Optional[Tuple[float, float]],
    y_zoom_percentiles: Tuple[float, float],
    full_y_range: bool,
    title: str,
    output_path: Path,
    plot_mode: str,
) -> None:
    is_paper_mode = str(plot_mode).lower() == "paper"
    font_scale = PAPER_FONT_SCALE if is_paper_mode else 1.0
    axis_fontsize = 11.0 * font_scale
    tick_fontsize = 10.0 * font_scale
    overlay_fontsize = 8.0 * font_scale
    legend_fontsize = 9.0 * font_scale

    fig_size = (14.0, 10.0) if is_paper_mode else (8.5, 6.2)
    fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=is_paper_mode)
    ax.scatter(
        x,
        y,
        s=float(point_size),
        alpha=float(point_alpha),
        color="tab:blue",
        label="samples",
    )

    centers, means, stds = _binned_mean_std(x, y, bins=int(bins))
    if centers.size > 0:
        ax.plot(centers, means, color="black", linewidth=2.0, label="mean trend")
        ax.fill_between(centers, means - stds, means + stds, color="gray", alpha=0.2, linewidth=0.0, label="±1 std")

    ax.set_xlabel(str(x_axis_label), fontsize=axis_fontsize)
    ax.set_ylabel(str(y_axis_label), fontsize=axis_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize, pad=10)
    ax.tick_params(axis="y", labelsize=tick_fontsize, pad=10)
    if is_paper_mode:
        ax.grid(False)
    else:
        ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.25)
    if not is_paper_mode:
        ax.legend(loc="best", fontsize=legend_fontsize)

    if x_lim_mm is not None:
        ax.set_xlim(float(x_lim_mm[0]), float(x_lim_mm[1]))
    if y_lim_mm is not None:
        ax.set_ylim(float(y_lim_mm[0]), float(y_lim_mm[1]))
    elif not bool(full_y_range):
        auto_lim = _percentile_axis_limits(
            y,
            low_pct=float(y_zoom_percentiles[0]),
            high_pct=float(y_zoom_percentiles[1]),
        )
        if auto_lim is not None:
            ax.set_ylim(auto_lim[0], auto_lim[1])

    if bool(show_trial_overlays) and trial_summary:
        x_data_min = float(np.min(x))
        x_data_max = float(np.max(x))
        ordered_trials, segments, _boundaries = _compute_trial_segments(
            trial_summary,
            x_data_min=x_data_min,
            x_data_max=x_data_max,
        )

        ymin, ymax = ax.get_ylim()
        y_text = ymax - 0.02 * (ymax - ymin)

        drawn = set()
        for trial_num in ordered_trials:
            x_lo, x_hi = segments[trial_num]
            x_mid = 0.5 * (x_lo + x_hi)
            ax.text(
                x_mid,
                y_text,
                f"T{trial_num}",
                fontsize=overlay_fontsize,
                ha="center",
                va="top",
                color="tab:purple",
            )
            for xb in (x_lo, x_hi):
                key = round(float(xb), 6)
                if key in drawn:
                    continue
                label = "trial boundary" if not drawn else None
                boundary_lw = 2.2 if is_paper_mode else 1.4
                boundary_alpha = 0.52 if is_paper_mode else 0.42
                ax.axvline(
                    float(xb),
                    color="tab:purple",
                    linestyle="--",
                    linewidth=boundary_lw,
                    alpha=boundary_alpha,
                    label=label,
                )
                drawn.add(key)

        if not is_paper_mode:
            dy_lines: List[str] = []
            for item in sorted(trial_delta_stats, key=lambda a: int(a["trial_num"])):
                trial_num = int(item["trial_num"])
                dy_lines.append(f"Δy' T{trial_num}: {float(item['delta_y_mean_mm']):+.2f} mm")
            if dy_lines:
                text = "\n".join(dy_lines)
                ax.text(
                    0.02,
                    0.02,
                    text,
                    transform=ax.transAxes,
                    fontsize=overlay_fontsize,
                    va="bottom",
                    ha="left",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.5"},
                )

    if is_paper_mode:
        arrow_linewidth = max(2.0, 1.2 * font_scale)
        arrow_fontsize = axis_fontsize
        arrow_color = "tab:red"
        arrow_head_scale = max(24.0, 0.95 * axis_fontsize)

        # Move axis labels a bit away from ticks so the added direction arrows can
        # sit in separate whitespace and not collide with axis text.
        ax.xaxis.labelpad = 11.0 * font_scale
        ax.yaxis.labelpad = 8.0 * font_scale

        # Vertical direction indicator: placed well outside the y-axis label area.
        ax.annotate(
            "",
            xy=(-0.20, 0.95),
            xytext=(-0.20, 0.05),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "<->",
                "linewidth": arrow_linewidth,
                "color": arrow_color,
                "mutation_scale": arrow_head_scale,
            },
            annotation_clip=False,
        )
        ax.text(
            -0.24,
            0.82,
            "upward",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=arrow_fontsize,
            color=arrow_color,
            rotation=90,
            clip_on=False,
        )
        ax.text(
            -0.24,
            0.18,
            "downward",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=arrow_fontsize,
            color=arrow_color,
            rotation=90,
            clip_on=False,
        )

        # Horizontal direction indicator: placed below x-axis label with extra gap.
        ax.annotate(
            "",
            xy=(0.95, -0.24),
            xytext=(0.05, -0.24),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "<->",
                "linewidth": arrow_linewidth,
                "color": arrow_color,
                "mutation_scale": arrow_head_scale,
            },
            annotation_clip=False,
        )
        ax.text(
            0.03,
            -0.30,
            "closer to flipper",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=arrow_fontsize,
            color=arrow_color,
            clip_on=False,
        )
        ax.text(
            0.97,
            -0.30,
            "further from flipper",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=arrow_fontsize,
            color=arrow_color,
            clip_on=False,
        )

    if not is_paper_mode:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_paper_mode:
        fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    else:
        fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_boundary_delta_y_plot(
    boundary_stats: List[Dict[str, float]],
    x_axis_label: str,
    delta_y_axis_label: str,
    title: str,
    output_path: Path,
    plot_mode: str,
) -> bool:
    if not boundary_stats:
        return False
    is_paper_mode = str(plot_mode).lower() == "paper"
    font_scale = PAPER_FONT_SCALE if is_paper_mode else 1.0
    axis_fontsize = 11.0 * font_scale
    tick_fontsize = 10.0 * font_scale
    overlay_fontsize = 8.0 * font_scale
    legend_fontsize = 9.0 * font_scale

    x_vals = np.asarray([float(item["x_boundary_mm"]) for item in boundary_stats], dtype=float)
    x_lo_vals = np.asarray([float(item.get("x_span_lo_mm", item["x_boundary_mm"])) for item in boundary_stats], dtype=float)
    x_hi_vals = np.asarray([float(item.get("x_span_hi_mm", item["x_boundary_mm"])) for item in boundary_stats], dtype=float)
    y_means = np.asarray([float(item["delta_y_mean_mm"]) for item in boundary_stats], dtype=float)
    y_stds = np.asarray([float(item["delta_y_std_mm"]) for item in boundary_stats], dtype=float)
    labels = [
        f"T{int(item['trial_num'])}"
        for item in boundary_stats
    ]

    fig_size = (14.0, 9.5) if is_paper_mode else (8.5, 5.8)
    fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=is_paper_mode)
    bar_widths = np.maximum(x_hi_vals - x_lo_vals, 2.0)
    ax.bar(
        x_lo_vals,
        y_means,
        width=bar_widths,
        color="tab:green",
        alpha=0.40,
        edgecolor="tab:green",
        linewidth=1.0,
        label="mean Δy'",
        align="edge",
    )
    ax.errorbar(
        x_vals,
        y_means,
        yerr=y_stds,
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=4,
        label="±1 std",
    )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(str(x_axis_label), fontsize=axis_fontsize)
    ax.set_ylabel(str(delta_y_axis_label), fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.02 * (ymax - ymin)
    drawn = set()
    for lbl, x_lo, x_hi in zip(labels, x_lo_vals, x_hi_vals):
        x_mid = 0.5 * (float(x_lo) + float(x_hi))
        ax.text(x_mid, y_text, f"{lbl}", ha="center", va="top", fontsize=overlay_fontsize, color="tab:purple")
        for xb in (float(x_lo), float(x_hi)):
            key = round(xb, 6)
            if key in drawn:
                continue
            ax.axvline(xb, color="tab:purple", linestyle="--", linewidth=0.9, alpha=0.32)
            drawn.add(key)

    if not is_paper_mode:
        ax.legend(loc="best", fontsize=legend_fontsize)
    if not is_paper_mode:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_paper_mode:
        fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    else:
        fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Plot obstacle_y_prime vs leg_z_prime_gap. "
            "obstacle_y_prime := y'(t), leg_z_prime_gap := |z'_obstacle - z'_leg|."
        )
    )
    ap.add_argument(
        "sessions",
        nargs="*",
        type=Path,
        help=(
            "Session directories (e.g., data/session_YYYYMMDD_HHMMSS). "
            "If omitted, uses hardcoded session groups and writes per-object plots."
        ),
    )
    ap.add_argument("--mocap-rb-id", type=int, default=None, help="Mocap rigid-body ID to use.")
    ap.add_argument(
        "--mocap-kind",
        choices=sorted(MOCAP_RB_IDS_BY_KIND.keys()),
        default=None,
        help="Shortcut for --mocap-rb-id using known mapping.",
    )
    ap.add_argument("--leg-x-mm", type=float, default=DEFAULT_LEG_X_MM, help="Leg x before incline rotation (mm).")
    ap.add_argument("--leg-y-mm", type=float, default=DEFAULT_LEG_Y_MM, help="Leg y before incline rotation (mm).")
    ap.add_argument("--leg-z-mm", type=float, default=DEFAULT_LEG_Z_MM, help="Leg z before incline rotation (mm).")
    ap.add_argument("--incline-deg", type=float, default=DEFAULT_INCLINE_DEG, help="Incline angle used for y'/z'.")
    ap.set_defaults(apply_leg_incline=True)
    ap.add_argument(
        "--apply-leg-incline",
        action="store_true",
        dest="apply_leg_incline",
        help="Apply incline rotation to the hardcoded leg point (default).",
    )
    ap.add_argument(
        "--no-leg-incline",
        action="store_false",
        dest="apply_leg_incline",
        help="Use the hardcoded leg point directly (no incline rotation).",
    )
    ap.add_argument("--bins", type=int, default=30, help="Number of x-bins for trend mean±std.")
    ap.add_argument("--point-size", type=float, default=7.0, help="Scatter marker size.")
    ap.add_argument("--point-alpha", type=float, default=0.35, help="Scatter marker alpha.")
    ap.add_argument(
        "--x-axis-label",
        type=str,
        default=DEFAULT_X_AXIS_LABEL,
        help="X-axis label text.",
    )
    ap.add_argument(
        "--y-axis-label",
        type=str,
        default=DEFAULT_Y_AXIS_LABEL,
        help="Y-axis label text.",
    )
    ap.add_argument(
        "--delta-y-axis-label",
        type=str,
        default=DEFAULT_DELTA_Y_AXIS_LABEL,
        help="Y-axis label for trial-wise Δy' bar/error plot.",
    )
    ap.set_defaults(make_delta_y_boundary_plot=True)
    ap.add_argument(
        "--make-delta-y-boundary-plot",
        action="store_true",
        dest="make_delta_y_boundary_plot",
        help="Also write trial-wise Δy' mean/std vs relative-z plot (default).",
    )
    ap.add_argument(
        "--no-delta-y-boundary-plot",
        action="store_false",
        dest="make_delta_y_boundary_plot",
        help="Do not write trial-wise Δy' plot.",
    )
    ap.set_defaults(show_trial_overlays=True)
    ap.add_argument(
        "--show-trial-overlays",
        action="store_true",
        dest="show_trial_overlays",
        help="Show trial range vertical lines and inter-trial Δy' annotations (default).",
    )
    ap.add_argument(
        "--no-trial-overlays",
        action="store_false",
        dest="show_trial_overlays",
        help="Hide trial range vertical lines and inter-trial Δy' annotations.",
    )
    ap.add_argument(
        "--y-zoom-percentiles",
        nargs=2,
        type=float,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
        help="Default y-axis zoom percentiles when full range is not requested.",
    )
    ap.add_argument(
        "--full-y-range",
        action="store_true",
        help="Disable percentile-based y-axis zoom and show full y-range.",
    )
    ap.add_argument(
        "--x-lim-mm",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Explicit x-axis limits in mm (overrides auto x-range behavior).",
    )
    ap.add_argument(
        "--fixed-x-range-mm",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Convenience alias for --x-lim-mm; use this to enforce same x-range across all plots.",
    )
    ap.add_argument(
        "--y-lim-mm",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Explicit y-axis limits in mm (overrides zoom/full-range behavior).",
    )
    ap.add_argument(
        "--fixed-y-range-mm",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Convenience alias for --y-lim-mm; use this to enforce same y-range across all plots.",
    )
    ap.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    ap.add_argument(
        "--delta-y-boundary-output",
        type=Path,
        default=None,
        help="Output path for trial-wise Δy' plot (single-run mode).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=(DATA_ROOT / "obstacle_yprime_vs_leg_zprime_gap"),
        help="Directory used for per-object outputs when sessions are omitted.",
    )
    ap.add_argument(
        "--delta-y-boundary-output-dir",
        type=Path,
        default=None,
        help="Directory for per-object trial-wise Δy' outputs (defaults to --output-dir).",
    )
    ap.add_argument("--title", type=str, default=None, help="Custom figure title.")
    ap.add_argument(
        "--plot-mode",
        type=str,
        choices=("default", "paper"),
        default=DEFAULT_PLOT_MODE,
        help=(
            "Plot styling mode. "
            "'paper' increases text size (~4.5x), removes legends, and adds axis direction arrows."
        ),
    )
    return ap


def main() -> int:
    args = _build_arg_parser().parse_args()
    x_lim_mm: Optional[Tuple[float, float]] = None
    if args.x_lim_mm is not None and args.fixed_x_range_mm is not None:
        raise SystemExit("Use only one of --x-lim-mm or --fixed-x-range-mm.")
    x_range_arg = args.fixed_x_range_mm if args.fixed_x_range_mm is not None else args.x_lim_mm
    if x_range_arg is not None:
        x_min = float(x_range_arg[0])
        x_max = float(x_range_arg[1])
        if x_max <= x_min:
            raise SystemExit(f"--x-lim-mm requires XMAX > XMIN (got {x_min}, {x_max}).")
        x_lim_mm = (x_min, x_max)
    y_lim_mm: Optional[Tuple[float, float]] = None
    if args.y_lim_mm is not None and args.fixed_y_range_mm is not None:
        raise SystemExit("Use only one of --y-lim-mm or --fixed-y-range-mm.")
    y_range_arg = args.fixed_y_range_mm if args.fixed_y_range_mm is not None else args.y_lim_mm
    if y_range_arg is not None:
        y_min = float(y_range_arg[0])
        y_max = float(y_range_arg[1])
        if y_max <= y_min:
            raise SystemExit(f"--y-lim-mm requires YMAX > YMIN (got {y_min}, {y_max}).")
        y_lim_mm = (y_min, y_max)
    low_pct = float(args.y_zoom_percentiles[0])
    high_pct = float(args.y_zoom_percentiles[1])
    if not (0.0 <= low_pct < high_pct <= 100.0):
        raise SystemExit(
            f"--y-zoom-percentiles must satisfy 0 <= LOW < HIGH <= 100 (got {low_pct}, {high_pct})."
        )

    leg_x_m = float(args.leg_x_mm) / M_TO_MM
    leg_y_m = float(args.leg_y_mm) / M_TO_MM
    leg_z_m = float(args.leg_z_mm) / M_TO_MM
    if bool(args.apply_leg_incline):
        _, _, leg_z_prime_m = _rotate_xyz_by_incline_about_x(
            leg_x_m, leg_y_m, leg_z_m, float(args.incline_deg)
        )
    else:
        leg_z_prime_m = leg_z_m

    def _resolve_requested_rb_id(kind: Optional[str], rb_id: Optional[int]) -> int:
        requested_rb_id_local = rb_id
        if kind is not None:
            kind_id = MOCAP_RB_IDS_BY_KIND[kind]
            if requested_rb_id_local is not None and requested_rb_id_local != kind_id:
                raise SystemExit(
                    f"--mocap-rb-id ({requested_rb_id_local}) conflicts with --mocap-kind {kind} ({kind_id})"
                )
            requested_rb_id_local = kind_id
        if requested_rb_id_local is None:
            requested_rb_id_local = int(MOCAP_RB_IDS_BY_KIND[DEFAULT_MOCAP_KIND])
            print(
                "No mocap selection provided; "
                f"using DEFAULT_MOCAP_KIND='{DEFAULT_MOCAP_KIND}' (RB {requested_rb_id_local})."
            )
        return requested_rb_id_local

    if args.sessions:
        requested_rb_id = _resolve_requested_rb_id(args.mocap_kind, args.mocap_rb_id)
        session_dirs = [Path(str(p).strip()) for p in args.sessions]
        for session_dir in session_dirs:
            if not session_dir.is_dir():
                raise SystemExit(f"Not a session directory: {session_dir}")

        x, y, total_trials, total_points, trial_summary, session_trial_summary = _collect_xy_mm(
            session_dirs=session_dirs,
            requested_rb_id=requested_rb_id,
            leg_z_prime_m=float(leg_z_prime_m),
        )
        if x.size == 0:
            raise SystemExit("No finite points found for plotting.")
        trial_delta_stats = _compute_trial_delta_y_stats(
            trial_summary=trial_summary,
            session_trial_summary=session_trial_summary,
            x_data_min=float(np.min(x)),
            x_data_max=float(np.max(x)),
        )

        title = args.title or f"obstacle_y' vs leg_z'_gap | RB {requested_rb_id}"
        output_path = args.output or (Path.cwd() / "obstacle_yprime_vs_leg_zprime_gap.png")
        _write_plot(
            x=x,
            y=y,
            trial_summary=trial_summary,
            trial_delta_stats=trial_delta_stats,
            bins=int(args.bins),
            point_size=float(args.point_size),
            point_alpha=float(args.point_alpha),
            show_trial_overlays=bool(args.show_trial_overlays),
            x_axis_label=str(args.x_axis_label),
            y_axis_label=str(args.y_axis_label),
            x_lim_mm=x_lim_mm,
            y_lim_mm=y_lim_mm,
            y_zoom_percentiles=(low_pct, high_pct),
            full_y_range=bool(args.full_y_range),
            title=title,
            output_path=output_path,
            plot_mode=str(args.plot_mode),
        )
        print(f"Wrote plot: {output_path}")
        print(f"Sessions: {len(session_dirs)}, trials used: {total_trials}, points used: {total_points}")
        if bool(args.make_delta_y_boundary_plot):
            if args.delta_y_boundary_output is not None:
                delta_output_path = args.delta_y_boundary_output
            elif args.output is not None:
                delta_output_path = output_path.with_name(f"{output_path.stem}_delta_y_trial{output_path.suffix}")
            else:
                delta_output_path = Path.cwd() / "obstacle_delta_y_vs_leg_zprime_trial.png"
            delta_title = (args.title + " | trial Δy'") if args.title is not None else (
                f"trial-wise Δy' vs leg-z' segment | RB {requested_rb_id}"
            )
            wrote_delta = _write_boundary_delta_y_plot(
                boundary_stats=trial_delta_stats,
                x_axis_label=str(args.x_axis_label),
                delta_y_axis_label=str(args.delta_y_axis_label),
                title=delta_title,
                output_path=delta_output_path,
                plot_mode=str(args.plot_mode),
            )
            if wrote_delta:
                print(f"Wrote trial Δy' plot: {delta_output_path}")
            else:
                print("Trial Δy' plot skipped (insufficient trial/session data).")
    else:
        if args.mocap_kind is None:
            kinds = list(DEFAULT_SESSIONS_BY_KIND.keys())
            print("No sessions provided; generating plots for all hardcoded objects.")
        else:
            kinds = [args.mocap_kind]
            print(f"No sessions provided; generating hardcoded plot for object '{args.mocap_kind}'.")

        wrote_any = False
        for kind in kinds:
            session_names = DEFAULT_SESSIONS_BY_KIND.get(kind, ())
            if not session_names:
                print(f"{kind}: no hardcoded sessions configured; skipping.")
                continue
            session_dirs = [DATA_ROOT / name for name in session_names]
            missing = [p for p in session_dirs if not p.is_dir()]
            if missing:
                missing_str = ", ".join(p.name for p in missing)
                raise SystemExit(f"{kind}: missing session directories: {missing_str}")

            requested_rb_id = _resolve_requested_rb_id(kind=kind, rb_id=args.mocap_rb_id)
            x, y, total_trials, total_points, trial_summary, session_trial_summary = _collect_xy_mm(
                session_dirs=session_dirs,
                requested_rb_id=requested_rb_id,
                leg_z_prime_m=float(leg_z_prime_m),
            )
            if x.size == 0:
                print(f"{kind}: no finite points found; skipping.")
                continue
            trial_delta_stats = _compute_trial_delta_y_stats(
                trial_summary=trial_summary,
                session_trial_summary=session_trial_summary,
                x_data_min=float(np.min(x)),
                x_data_max=float(np.max(x)),
            )

            output_dir = Path(args.output_dir)
            if args.output is not None and len(kinds) == 1:
                output_path = args.output
            else:
                output_path = output_dir / f"obstacle_yprime_vs_leg_zprime_gap_{kind}.png"
            title = args.title or f"{kind}: obstacle_y' vs leg_z'_gap | RB {requested_rb_id}"
            _write_plot(
                x=x,
                y=y,
                trial_summary=trial_summary,
                trial_delta_stats=trial_delta_stats,
                bins=int(args.bins),
                point_size=float(args.point_size),
                point_alpha=float(args.point_alpha),
                show_trial_overlays=bool(args.show_trial_overlays),
                x_axis_label=str(args.x_axis_label),
                y_axis_label=str(args.y_axis_label),
                x_lim_mm=x_lim_mm,
                y_lim_mm=y_lim_mm,
                y_zoom_percentiles=(low_pct, high_pct),
                full_y_range=bool(args.full_y_range),
                title=title,
                output_path=output_path,
                plot_mode=str(args.plot_mode),
            )
            wrote_any = True
            print(f"{kind}: wrote plot: {output_path}")
            print(f"{kind}: sessions={len(session_dirs)}, trials used={total_trials}, points used={total_points}")
            if bool(args.make_delta_y_boundary_plot):
                delta_output_dir = Path(args.delta_y_boundary_output_dir) if args.delta_y_boundary_output_dir is not None else output_dir
                if args.delta_y_boundary_output is not None and len(kinds) == 1:
                    delta_output_path = args.delta_y_boundary_output
                else:
                    delta_output_path = delta_output_dir / f"obstacle_delta_y_vs_leg_zprime_trial_{kind}.png"
                delta_title = (
                    (args.title + " | trial Δy'") if args.title is not None
                    else f"{kind}: trial-wise Δy' vs leg-z' segment | RB {requested_rb_id}"
                )
                wrote_delta = _write_boundary_delta_y_plot(
                    boundary_stats=trial_delta_stats,
                    x_axis_label=str(args.x_axis_label),
                    delta_y_axis_label=str(args.delta_y_axis_label),
                    title=delta_title,
                    output_path=delta_output_path,
                    plot_mode=str(args.plot_mode),
                )
                if wrote_delta:
                    print(f"{kind}: wrote trial Δy' plot: {delta_output_path}")
                else:
                    print(f"{kind}: trial Δy' plot skipped (insufficient trial/session data).")

        if not wrote_any:
            raise SystemExit("No plots were generated.")

    print("Definition: obstacle_y' := y'(t)")
    print("Definition: leg_z'_gap := |z'_obstacle(t) - z'_robot_leg|")
    print(
        "Leg (input, mm): "
        f"x={float(args.leg_x_mm):.3f}, y={float(args.leg_y_mm):.3f}, z={float(args.leg_z_mm):.3f}"
    )
    print(f"Incline rotation applied to leg: {bool(args.apply_leg_incline)} (incline={float(args.incline_deg):.3f} deg)")
    print(f"Leg z' used (mm): {float(leg_z_prime_m * M_TO_MM):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# need to think about trial boundary
