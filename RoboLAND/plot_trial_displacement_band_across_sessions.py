#!/usr/bin/env python3
"""Plot per-trial mocap displacement across sessions with mean/std bands.

For each input session, this script loads trial_*.npy files, computes per-trial
net displacement (last finite sample minus first finite sample) for x, y', z'
from rotated mocap position arrays, then plots:
  - mean displacement across sessions
  - +/- 1 std band across sessions

Example:
    python3 highlevel/terrain_manipulation/src/utils/plot_trial_displacement_band_across_sessions.py \
      highlevel/terrain_manipulation/data/session_20260301_101010 \
      highlevel/terrain_manipulation/data/session_20260302_101010 \
      highlevel/terrain_manipulation/data/session_20260303_101010 \
      --mocap-rb-id 6
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
# Default sessions used when no positional session args are provided.

# # empty
# DEFAULT_SESSION_NAMES: Sequence[str] = (
#     "session_20260313_111621",
#     "session_20260313_112741",
#     "session_20260313_125836",
#     "session_20260313_121725",
#     "session_20260313_123622",
# )

# lead
DEFAULT_SESSION_NAMES: Sequence[str] = (
    "session_20260314_114731",
    "session_20260314_115216",
    "session_20260314_115730",
    "session_20260314_120728",
    "session_20260314_121349",
    "session_20260314_121907"
)

# #steel
# DEFAULT_SESSION_NAMES: Sequence[str] = (
# "session_20260319_134621",
# "session_20260319_135104",
# "session_20260319_140118",
# "session_20260319_140705",
# "session_20260319_141232",
# "session_20260319_141641"
# )

# resin
# DEFAULT_SESSION_NAMES: Sequence[str] = (
# "session_20260319_142653",
# "session_20260319_145057",
# "session_20260319_145447",
# "session_20260319_145955",
# "session_20260319_151418",
# "session_20260319_151845",
# "session_20260319_152329"
# )

# # sand
# DEFAULT_SESSION_NAMES: Sequence[str] = (
# "session_20260317_153944",
# "session_20260317_154742",
# "session_20260317_155128",
# "session_20260317_155550",
# "session_20260317_160026",
# "session_20260317_160410"
# )



MOCAP_RB_IDS_BY_KIND: Dict[str, int] = {
    "empty": 2,
    "lead": 3,
    "resin": 5,
    "steel": 6,
    "sand": 8,
}
# Used when neither --mocap-rb-id nor --mocap-kind is passed.
DEFAULT_MOCAP_KIND = "lead"
DEFAULT_PLOT_MODE = "default"
PAPER_FONT_SCALE = 3.0
POSTER_FONT_SCALE = 3.5
PAPER_DEFAULT_X_LIM_CM: Tuple[float, float] = (-0.2, 1.0)
PAPER_DEFAULT_Y_LIM_CM: Tuple[float, float] = (-1.2, 1.25)
PAPER_DEFAULT_Z_LIM_CM: Tuple[float, float] = (-5.0, 35.0)


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


def _as_float_array(value: object) -> Optional[np.ndarray]:
    if not isinstance(value, np.ndarray):
        return None
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _extract_first_mocap_series(mocap_state: Dict[str, object], keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        arr = _as_float_array(mocap_state.get(key))
        if arr is not None:
            return arr
    return None


def _extract_first_mocap_series_with_key(
    mocap_state: Dict[str, object],
    keys: Sequence[str],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for key in keys:
        arr = _as_float_array(mocap_state.get(key))
        if arr is not None:
            return arr, key
    return None, None


def _position_family_from_key(position_key: str) -> Optional[str]:
    if position_key.startswith("rotated_zeroed_position_"):
        return "rotated_zeroed"
    if position_key.startswith("rotated_position_"):
        return "rotated"
    if position_key.startswith("zeroed_position_"):
        return "zeroed"
    if position_key.startswith("position_"):
        return "raw"
    return None


def _orientation_keys_for_family(family: str) -> Tuple[str, str, str, str]:
    if family == "rotated_zeroed":
        return (
            "rotated_zeroed_orientation_w",
            "rotated_zeroed_orientation_x",
            "rotated_zeroed_orientation_y",
            "rotated_zeroed_orientation_z",
        )
    if family == "rotated":
        return (
            "rotated_orientation_w",
            "rotated_orientation_x",
            "rotated_orientation_y",
            "rotated_orientation_z",
        )
    if family == "zeroed":
        return (
            "zeroed_orientation_w",
            "zeroed_orientation_x",
            "zeroed_orientation_y",
            "zeroed_orientation_z",
        )
    return (
        "orientation_w",
        "orientation_x",
        "orientation_y",
        "orientation_z",
    )


def _rotate_vector_by_quaternion_batch(quat_wxyz: np.ndarray, vector_xyz: np.ndarray) -> np.ndarray:
    # Uses v' = q * v * q_conjugate with normalized quaternions.
    u = quat_wxyz[:, 1:4]
    s = quat_wxyz[:, :1]
    v = vector_xyz
    dot_uv = np.sum(u * v, axis=1, keepdims=True)
    dot_uu = np.sum(u * u, axis=1, keepdims=True)
    cross_uv = np.cross(u, v)
    return 2.0 * dot_uv * u + (s * s - dot_uu) * v + 2.0 * s * cross_uv


def _apply_body_com_offset_to_series(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    qw: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    offset_body_xyz_m: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    offset = np.asarray(offset_body_xyz_m, dtype=float).reshape(3)
    if np.allclose(offset, 0.0):
        return x, y, z, False
    if not (len(x) == len(y) == len(z) == len(qw) == len(qx) == len(qy) == len(qz)):
        return x, y, z, False

    pos = np.column_stack([x, y, z]).astype(float, copy=True)
    quat = np.column_stack([qw, qx, qy, qz]).astype(float, copy=False)

    finite_mask = np.isfinite(pos).all(axis=1) & np.isfinite(quat).all(axis=1)
    if not np.any(finite_mask):
        return x, y, z, False

    quat_valid = quat[finite_mask]
    norms = np.linalg.norm(quat_valid, axis=1, keepdims=True)
    norm_mask = (norms[:, 0] > 1e-12) & np.isfinite(norms[:, 0])
    if not np.any(norm_mask):
        return x, y, z, False

    quat_valid = quat_valid[norm_mask] / norms[norm_mask]
    rotated_offset = _rotate_vector_by_quaternion_batch(
        quat_valid,
        np.broadcast_to(offset, (quat_valid.shape[0], 3)),
    )

    finite_indices = np.flatnonzero(finite_mask)
    target_indices = finite_indices[norm_mask]
    pos[target_indices] += rotated_offset
    return pos[:, 0], pos[:, 1], pos[:, 2], True


def _get_mocap_state(payload: Dict[str, object], rb_id: int) -> Optional[Dict[str, object]]:
    for container_key in ("mocap_raw", "mocap"):
        mocap = payload.get(container_key)
        if not isinstance(mocap, dict):
            continue
        state = mocap.get(str(rb_id), mocap.get(rb_id))
        if isinstance(state, dict):
            return state
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


def _finite_first_last(values: np.ndarray) -> Tuple[float, float]:
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return (math.nan, math.nan)
    return float(values[finite_idx[0]]), float(values[finite_idx[-1]])


def _trial_delta_xyz_cm(payload: Dict[str, object], rb_id: int, com_offset_y_m: float) -> Dict[str, float]:
    mocap_state = _get_mocap_state(payload, rb_id)
    if mocap_state is None:
        raise ValueError(f"mocap state for RB ID {rb_id} not found")

    x, x_key = _extract_first_mocap_series_with_key(
        mocap_state,
        ("rotated_position_x", "position_x", "rotated_zeroed_position_x", "zeroed_position_x"),
    )
    y, y_key = _extract_first_mocap_series_with_key(
        mocap_state,
        ("rotated_position_y", "position_y", "rotated_zeroed_position_y", "zeroed_position_y"),
    )
    z, z_key = _extract_first_mocap_series_with_key(
        mocap_state,
        ("rotated_position_z", "position_z", "rotated_zeroed_position_z", "zeroed_position_z"),
    )
    if x is None or y is None or z is None:
        raise ValueError("missing mocap position arrays (expected rotated_position_* or fallback position_*)")
    if float(com_offset_y_m) != 0.0:
        x_family = _position_family_from_key(x_key or "")
        y_family = _position_family_from_key(y_key or "")
        z_family = _position_family_from_key(z_key or "")
        if x_family is None or y_family is None or z_family is None or len({x_family, y_family, z_family}) != 1:
            raise ValueError(
                "unable to determine a consistent mocap position family for COM correction "
                f"(x={x_key}, y={y_key}, z={z_key})"
            )
        qw_key, qx_key, qy_key, qz_key = _orientation_keys_for_family(x_family)
        qw = _as_float_array(mocap_state.get(qw_key))
        qx = _as_float_array(mocap_state.get(qx_key))
        qy = _as_float_array(mocap_state.get(qy_key))
        qz = _as_float_array(mocap_state.get(qz_key))
        if qw is None or qx is None or qy is None or qz is None:
            raise ValueError(
                "missing mocap orientation arrays for COM correction "
                f"(expected {qw_key}/{qx_key}/{qy_key}/{qz_key})"
            )
        x, y, z, com_applied = _apply_body_com_offset_to_series(
            x,
            y,
            z,
            qw,
            qx,
            qy,
            qz,
            (0.0, float(com_offset_y_m), 0.0),
        )
        if not com_applied:
            raise ValueError("unable to apply COM correction due to invalid or mismatched mocap pose arrays")

    x0, x1 = _finite_first_last(x)
    y0, y1 = _finite_first_last(y)
    z0, z1 = _finite_first_last(z)
    return {
        "x": (x1 - x0) * 100.0 if np.isfinite([x0, x1]).all() else math.nan,
        "y": (y1 - y0) * 100.0 if np.isfinite([y0, y1]).all() else math.nan,
        "z": (z1 - z0) * 100.0 if np.isfinite([z0, z1]).all() else math.nan,
    }


def _session_trial_deltas_cm(
    session_dir: Path,
    requested_rb_id: Optional[int],
    com_offset_y_m: float,
) -> Dict[int, Dict[str, float]]:
    trials = _list_trials(session_dir)
    if not trials:
        raise ValueError(f"no trial_*.npy files found in {session_dir}")

    out: Dict[int, Dict[str, float]] = {}
    selected_rb_id: Optional[int] = None
    for trial_path in trials:
        trial_num = _trial_number(trial_path)
        if trial_num is None:
            continue
        payload = _load_payload(trial_path)
        rb_id = _select_mocap_rb_id(payload, requested_rb_id=requested_rb_id, trial_path=trial_path)
        if selected_rb_id is None:
            selected_rb_id = rb_id
        elif selected_rb_id != rb_id:
            raise ValueError(
                f"{session_dir.name}: inconsistent selected RB IDs ({selected_rb_id} vs {rb_id})"
            )
        out[trial_num] = _trial_delta_xyz_cm(payload, rb_id, com_offset_y_m=com_offset_y_m)
    return out


def _nanmean_std(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean[~np.isfinite(mean)] = np.nan
    std[~np.isfinite(std)] = np.nan
    return mean, std


def _validate_axis_lim(name: str, lim: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    if lim is None:
        return None
    lo = float(lim[0])
    hi = float(lim[1])
    if hi <= lo:
        raise SystemExit(f"--{name} requires MAX > MIN (got {lo}, {hi}).")
    return lo, hi


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Plot per-trial net displacement (x, y', z') across multiple sessions "
            "with mean and std bands."
        )
    )
    ap.add_argument(
        "sessions",
        nargs="*",
        type=Path,
        help=(
            "Session directories (e.g., data/session_YYYYMMDD_HHMMSS). "
            "If omitted, script uses DEFAULT_SESSION_NAMES."
        ),
    )
    ap.add_argument(
        "--mocap-rb-id",
        type=int,
        default=None,
        help="Mocap rigid-body ID to use (recommended when multiple RBs are present).",
    )
    ap.add_argument(
        "--mocap-kind",
        choices=sorted(MOCAP_RB_IDS_BY_KIND.keys()),
        default=None,
        help="Shortcut for --mocap-rb-id using known mapping.",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom figure title.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: ./trial_displacement_band_across_sessions.png).",
    )
    ap.add_argument(
        "--com-offset-y-m",
        type=float,
        default=0.0,
        help="Constant COM offset along body Y (meters), rotated into world frame per sample.",
    )
    ap.add_argument(
        "--plot-mode",
        type=str,
        choices=("default", "paper", "poster"),
        default=DEFAULT_PLOT_MODE,
        help=(
            "Plot styling mode. "
            "'paper' uses 3x text, removes legends, and adds dy direction arrow on y'. "
            "'poster' uses 1.5x text and presentation-friendly axis labels."
        ),
    )
    ap.add_argument(
        "--x-lim-cm",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Y-limits for x subplot in cm.",
    )
    ap.add_argument(
        "--y-lim-cm",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Y-limits for y' subplot in cm.",
    )
    ap.add_argument(
        "--z-lim-cm",
        nargs=2,
        type=float,
        default=None,
        metavar=("ZMIN", "ZMAX"),
        help="Y-limits for z' subplot in cm.",
    )
    return ap


def main() -> int:
    args = _build_arg_parser().parse_args()
    plot_mode = str(args.plot_mode).lower()
    is_paper_mode = plot_mode in ("paper", "poster")
    is_poster_mode = plot_mode == "poster"
    if is_poster_mode:
        font_scale = POSTER_FONT_SCALE
    elif is_paper_mode:
        font_scale = PAPER_FONT_SCALE
    else:
        font_scale = 1.0
    title_fontsize = 12.0 * font_scale
    axis_label_fontsize = 10.0 * font_scale
    tick_fontsize = 9.0 * font_scale
    legend_fontsize = 9.0 * font_scale

    x_lim_cm = _validate_axis_lim("x-lim-cm", args.x_lim_cm)
    y_lim_cm = _validate_axis_lim("y-lim-cm", args.y_lim_cm)
    z_lim_cm = _validate_axis_lim("z-lim-cm", args.z_lim_cm)
    if is_paper_mode:
        if x_lim_cm is None:
            x_lim_cm = PAPER_DEFAULT_X_LIM_CM
        if y_lim_cm is None:
            y_lim_cm = PAPER_DEFAULT_Y_LIM_CM
        if z_lim_cm is None:
            z_lim_cm = PAPER_DEFAULT_Z_LIM_CM
    requested_rb_id = args.mocap_rb_id
    if args.mocap_kind is not None:
        kind_id = MOCAP_RB_IDS_BY_KIND[args.mocap_kind]
        if requested_rb_id is not None and requested_rb_id != kind_id:
            raise SystemExit(
                f"--mocap-rb-id ({requested_rb_id}) conflicts with --mocap-kind {args.mocap_kind} ({kind_id})"
            )
        requested_rb_id = kind_id

    if args.sessions:
        session_dirs = [Path(str(p).strip()) for p in args.sessions]
    else:
        session_dirs = [DATA_ROOT / name for name in DEFAULT_SESSION_NAMES]
        print("No session args provided; using DEFAULT_SESSION_NAMES from script.")

    if requested_rb_id is None:
        requested_rb_id = int(MOCAP_RB_IDS_BY_KIND[DEFAULT_MOCAP_KIND])
        print(f"No mocap selection provided; using DEFAULT_MOCAP_KIND='{DEFAULT_MOCAP_KIND}' (RB {requested_rb_id}).")

    for session_dir in session_dirs:
        if not session_dir.is_dir():
            raise SystemExit(f"Not a session directory: {session_dir}")

    per_session: List[Dict[int, Dict[str, float]]] = []
    for session_dir in session_dirs:
        trial_deltas = _session_trial_deltas_cm(
            session_dir,
            requested_rb_id=requested_rb_id,
            com_offset_y_m=float(args.com_offset_y_m),
        )
        per_session.append(trial_deltas)
        print(f"{session_dir.name}: loaded {len(trial_deltas)} trial delta(s)")

    trial_numbers = sorted({t for sess in per_session for t in sess.keys()})
    if not trial_numbers:
        raise SystemExit("No trial deltas found across provided sessions.")

    n_sessions = len(per_session)
    n_trials = len(trial_numbers)
    x_mat = np.full((n_sessions, n_trials), np.nan, dtype=float)
    y_mat = np.full((n_sessions, n_trials), np.nan, dtype=float)
    z_mat = np.full((n_sessions, n_trials), np.nan, dtype=float)
    trial_idx_map = {trial_num: idx for idx, trial_num in enumerate(trial_numbers)}

    for sidx, sess in enumerate(per_session):
        for trial_num, deltas in sess.items():
            tidx = trial_idx_map[trial_num]
            x_mat[sidx, tidx] = float(deltas.get("x", math.nan))
            y_mat[sidx, tidx] = float(deltas.get("y", math.nan))
            z_mat[sidx, tidx] = float(deltas.get("z", math.nan))

    x_mean, x_std = _nanmean_std(x_mat)
    y_mean, y_std = _nanmean_std(y_mat)
    z_mean, z_std = _nanmean_std(z_mat)

    fig_size = (16.0, 16.0) if is_paper_mode else (10.0, 10.0)
    fig, axes = plt.subplots(3, 1, figsize=fig_size, sharex=True, constrained_layout=False)
    dims: Sequence[Tuple[str, np.ndarray, np.ndarray, str, str]] = (
        ("x", x_mean, x_std, "tab:purple", "x"),
        ("y'", y_mean, y_std, "tab:orange", "y'"),
        ("z'", z_mean, z_std, "tab:cyan", "z'"),
    )
    axis_lims_by_dim: Dict[str, Optional[Tuple[float, float]]] = {
        "x": x_lim_cm,
        "y'": y_lim_cm,
        "z'": z_lim_cm,
    }

    x_axis = np.asarray(trial_numbers, dtype=int)
    for ax, (dim_label, mean_vals, std_vals, color, legend_label) in zip(axes, dims):
        lower = mean_vals - std_vals
        upper = mean_vals + std_vals
        ax.fill_between(x_axis, lower, upper, color=color, alpha=0.25, linewidth=0.0, label=f"{legend_label} ±1 std")
        ax.plot(x_axis, mean_vals, color=color, linewidth=2.0, marker="o", label=f"{legend_label} mean")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
        if is_poster_mode:
            poster_ylabel_by_dim = {
                "x": "Lateral disp. (cm)",
                "y'": "Vertical disp. (cm)",
                "z'": "Fore-aft disp. (cm)",
            }
            ylabel_text = poster_ylabel_by_dim.get(dim_label, f"{dim_label} delta (cm)")
        else:
            ylabel_text = f"{dim_label} delta (cm)"
        ax.set_ylabel(ylabel_text, fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(True, alpha=0.3)
        lim = axis_lims_by_dim.get(dim_label)
        if lim is not None:
            ax.set_ylim(float(lim[0]), float(lim[1]))
        if not is_paper_mode:
            ax.legend(loc="best", fontsize=legend_fontsize)

    if is_paper_mode:
        dy_arrow_color = "tab:red"
        dy_arrow_lw = max(2.4, 1.1 * font_scale)
        dy_arrow_head = max(20.0, 9.0 * font_scale)
        dy_label_font = axis_label_fontsize

        def _add_direction_arrow(ax, positive_text: str) -> None:
            ax.yaxis.labelpad = 10.0 * font_scale
            ax.annotate(
                "",
                xy=(-0.19, 0.72),
                xytext=(-0.19, 0.28),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops={
                    "arrowstyle": "-|>",
                    "linewidth": dy_arrow_lw,
                    "color": dy_arrow_color,
                    "mutation_scale": dy_arrow_head,
                },
                annotation_clip=False,
            )
            ax.text(
                -0.24,
                0.50,
                positive_text,
                transform=ax.transAxes,
                rotation=90,
                ha="right",
                va="center",
                fontsize=dy_label_font,
                color=dy_arrow_color,
                clip_on=False,
            )

        _add_direction_arrow(axes[0], "x+ left lateral")
        _add_direction_arrow(axes[1], "y+ out of sand")
        _add_direction_arrow(axes[2], "z+ downslope")
        for ax in axes:
            ax.yaxis.set_label_coords(-0.12, 0.5)
        fig.subplots_adjust(left=0.28, right=0.98, top=0.96, bottom=0.08, hspace=0.18)

    axes[-1].set_xlabel("trial number", fontsize=axis_label_fontsize)
    if is_paper_mode:
        if args.title:
            fig.suptitle(args.title, fontsize=title_fontsize)
    else:
        title = args.title or f"Per-trial net displacement across {n_sessions} session(s)"
        fig.suptitle(title, fontsize=title_fontsize)
        fig.tight_layout()

    output_path = args.output or (Path.cwd() / "trial_displacement_band_across_sessions.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_paper_mode:
        fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    else:
        fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")
    print(f"COM offset (body y): {float(args.com_offset_y_m):.6f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
