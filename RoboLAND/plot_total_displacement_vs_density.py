#!/usr/bin/env python3
"""Plot total y'/z' displacement vs obstacle density using hardcoded sessions.

Uses the same hardcoded session groups as plot_placement_offset_yprime_vs_leg_zprime_gap.py
for: empty, resin, steel, lead (sand excluded).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from plot_avg_displacement_vs_density import _session_trial_deltas_cm
from plot_placement_offset_yprime_vs_leg_zprime_gap import (
    DATA_ROOT,
    DEFAULT_SESSIONS_BY_KIND,
    MOCAP_RB_IDS_BY_KIND,
)


DEFAULT_KINDS: Tuple[str, ...] = ("empty", "resin", "steel", "lead")
DEFAULT_TRIAL_NUMBERS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
DEFAULT_PLOT_MODE = "default"
PAPER_FONT_SCALE = 2.4


def _parse_trial_numbers(value: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("trials cannot be empty")
    out: List[int] = []
    for p in parts:
        try:
            t = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid trial number '{p}'") from exc
        if t <= 0:
            raise argparse.ArgumentTypeError(f"trial number must be positive (got {t})")
        out.append(t)
    return tuple(sorted(set(out)))


def _resolve_session_dir(token: str, data_root: Path) -> Path:
    candidate = Path(str(token).strip())
    if candidate.is_dir():
        return candidate
    rooted = data_root / candidate
    if rooted.is_dir():
        return rooted
    raise FileNotFoundError(f"Session directory not found: {token}")


def _session_total_axis_cm(
    session_dir: Path,
    rb_id: int,
    trial_numbers: Sequence[int],
    axis_key: str,
) -> float:
    trial_deltas = _session_trial_deltas_cm(session_dir, requested_rb_id=rb_id)
    vals: List[float] = []
    for t in trial_numbers:
        delta = trial_deltas.get(int(t))
        if delta is None:
            continue
        v = float(delta.get(axis_key, math.nan))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return math.nan
    return float(np.sum(np.asarray(vals, dtype=float)))


def _aggregate_by_kind(
    kinds: Sequence[str],
    trial_numbers: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_means: List[float] = []
    y_stds: List[float] = []
    z_means: List[float] = []
    z_stds: List[float] = []

    for kind in kinds:
        rb_id = int(MOCAP_RB_IDS_BY_KIND[kind])
        session_tokens = list(DEFAULT_SESSIONS_BY_KIND.get(kind, ()))
        if not session_tokens:
            raise SystemExit(f"No sessions configured for kind '{kind}'.")

        y_totals: List[float] = []
        z_totals: List[float] = []
        for token in session_tokens:
            session_dir = _resolve_session_dir(str(token), data_root=DATA_ROOT)
            y_total = _session_total_axis_cm(
                session_dir=session_dir,
                rb_id=rb_id,
                trial_numbers=trial_numbers,
                axis_key="y",
            )
            z_total = _session_total_axis_cm(
                session_dir=session_dir,
                rb_id=rb_id,
                trial_numbers=trial_numbers,
                axis_key="z",
            )
            if np.isfinite(y_total):
                y_totals.append(float(y_total))
            if np.isfinite(z_total):
                z_totals.append(float(z_total))

        if not y_totals or not z_totals:
            raise SystemExit(f"{kind}: no finite totals computed from selected trials {list(trial_numbers)}")

        y_arr = np.asarray(y_totals, dtype=float)
        z_arr = np.asarray(z_totals, dtype=float)
        y_means.append(float(np.mean(y_arr)))
        y_stds.append(float(np.std(y_arr)))
        z_means.append(float(np.mean(z_arr)))
        z_stds.append(float(np.std(z_arr)))
        print(
            f"{kind}: n_sessions={len(session_tokens)} | "
            f"total y' mean={y_means[-1]:.3f} cm std={y_stds[-1]:.3f} | "
            f"total z' mean={z_means[-1]:.3f} cm std={z_stds[-1]:.3f}"
        )

    return (
        np.asarray(y_means, dtype=float),
        np.asarray(y_stds, dtype=float),
        np.asarray(z_means, dtype=float),
        np.asarray(z_stds, dtype=float),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Plot total displacement vs obstacle density using the hardcoded session "
            "groups from plot_placement_offset_yprime_vs_leg_zprime_gap.py"
        )
    )
    ap.add_argument(
        "--trials",
        type=_parse_trial_numbers,
        default=DEFAULT_TRIAL_NUMBERS,
        help="Comma-separated trial numbers to include (default: 1,2,3,4,5,6,7).",
    )
    ap.add_argument(
        "--output-prefix",
        type=Path,
        default=(Path.cwd() / "total_displacement_vs_density"),
        help="Output path prefix; script writes '<prefix>_yprime.png' and '<prefix>_zprime.png'.",
    )
    ap.add_argument("--title", type=str, default=None, help="Optional figure title.")
    ap.add_argument(
        "--plot-mode",
        type=str,
        choices=("default", "paper"),
        default=DEFAULT_PLOT_MODE,
        help="Plot styling mode. 'paper' increases text size and figure size.",
    )
    return ap


def main() -> int:
    args = _build_arg_parser().parse_args()
    is_paper_mode = str(args.plot_mode).lower() == "paper"
    font_scale = PAPER_FONT_SCALE if is_paper_mode else 1.0
    font_bump = 8.0 if is_paper_mode else 0.0
    axis_fontsize = 10.0 * font_scale + font_bump
    tick_fontsize = 9.0 * font_scale + font_bump
    kinds = [k for k in DEFAULT_KINDS if k in DEFAULT_SESSIONS_BY_KIND]
    if len(kinds) != len(DEFAULT_KINDS):
        missing = [k for k in DEFAULT_KINDS if k not in kinds]
        raise SystemExit(f"Missing kinds in DEFAULT_SESSIONS_BY_KIND: {missing}")

    y_mean, y_std, z_mean, z_std = _aggregate_by_kind(kinds=kinds, trial_numbers=tuple(args.trials))

    x = np.arange(len(kinds), dtype=float)

    def _write_single_plot(
        values: np.ndarray,
        stds: np.ndarray,
        color: str,
        ylabel: str,
        axis_short: str,
    ) -> Path:
        fig_size = (10.0, 8.0) if is_paper_mode else (7.0, 5.0)
        fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=True)
        ax.bar(x, values, yerr=stds, capsize=(7 if is_paper_mode else 5), color=color, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(kinds, fontsize=tick_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=tick_fontsize, pad=8)

        if is_paper_mode:
            arrow_color = "tab:red"
            arrow_lw = max(2.2, 1.1 * font_scale)
            arrow_head = max(20.0, 8.0 * font_scale)
            arrow_font = axis_fontsize
            ax.yaxis.labelpad = 10.0 * font_scale
            ax.yaxis.set_label_coords(-0.10, 0.5)
            ax.annotate(
                "",
                xy=(-0.20, 0.78),
                xytext=(-0.20, 0.22),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops={
                    "arrowstyle": "-|>",
                    "linewidth": arrow_lw,
                    "color": arrow_color,
                    "mutation_scale": arrow_head,
                },
                annotation_clip=False,
            )
            if axis_short == "yprime":
                ax.text(
                    -0.25,
                    0.74,
                    "upward",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=arrow_font,
                    color=arrow_color,
                    clip_on=False,
                )
                ax.text(
                    -0.25,
                    0.26,
                    "downward",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=arrow_font,
                    color=arrow_color,
                    clip_on=False,
                )
            elif axis_short == "zprime":
                ax.text(
                    -0.25,
                    0.50,
                    "further from flipper",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=arrow_font,
                    color=arrow_color,
                    clip_on=False,
                )

        if args.title and not is_paper_mode:
            ax.set_title(f"{args.title} ({axis_short})", fontsize=12.0 * font_scale)
        elif args.title and is_paper_mode:
            ax.set_title(f"{args.title} ({axis_short})", fontsize=12.0 * font_scale + font_bump)

        out_prefix = Path(args.output_prefix)
        out_path = out_prefix.parent / f"{out_prefix.name}_{axis_short}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=(0.15 if is_paper_mode else 0.05))
        plt.close(fig)
        return out_path

    y_out = _write_single_plot(
        values=y_mean,
        stds=y_std,
        color="tab:orange",
        ylabel="Net y' displacement (cm)",
        axis_short="yprime",
    )
    z_out = _write_single_plot(
        values=z_mean,
        stds=z_std,
        color="tab:cyan",
        ylabel="Net z' displacement (cm)",
        axis_short="zprime",
    )
    print(f"Wrote plot: {y_out}")
    print(f"Wrote plot: {z_out}")
    print(f"Kinds: {kinds}")
    print(f"Trials: {list(args.trials)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
