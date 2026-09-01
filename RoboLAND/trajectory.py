"""Named right-leg sweep trajectories for terrain manipulation trials.

Each trajectory is a flattened list of waypoints:

    [right_adduction_rad, right_sweeping_rad, speed_rad_s, ...]

Labels use degree and rad/s units:

    {adduction displacement}_{sweep displacement}_{speed rad/s}_{front|back}

For example, ``30_90_2_front`` starts at the centered sweeping angle,
pre-positions to the back side, moves the leg down by adding 30 degrees to
right-adduction, sweeps forward through 90 degrees, lifts back up, and returns
to center.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple


TRAJ_SPEED_RAD_S = 2.0
DEFAULT_TRAJECTORY_NAME = "30_30_2_front"

# Controller coordinates from the current fixed trajectory.
ADDUCTION_HOME_BASE_RAD = 0.5235987756
ADDUCTION_HOME_OFFSET_CORRECTION_DEG = -70.0
ADDUCTION_HOME_RAD = ADDUCTION_HOME_BASE_RAD + math.radians(ADDUCTION_HOME_OFFSET_CORRECTION_DEG)
SWEEP_CENTER_RAD = -0.53

ADDUCTION_DISPLACEMENTS_DEG = (30, 60, 90)
SWEEP_DISPLACEMENTS_DEG = (30, 60, 90)
SWEEP_DIRECTIONS = ("front", "back")

SweepDirection = Literal["front", "back"]
Waypoint = Tuple[float, float]


@dataclass(frozen=True)
class TrajectorySpec:
    name: str
    adduction_deg: int
    sweep_deg: int
    speed_rad_s: float
    direction: SweepDirection
    points: List[float]

    def as_metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "adduction_deg": int(self.adduction_deg),
            "sweep_deg": int(self.sweep_deg),
            "speed_rad_s": float(self.speed_rad_s),
            "direction": self.direction,
        }


def _format_speed(speed_rad_s: float) -> str:
    return f"{float(speed_rad_s):g}".replace(".", "p")


def _canonical_name(
    adduction_displacement_deg: int,
    sweep_displacement_deg: int,
    speed_rad_s: float,
    direction: SweepDirection,
) -> str:
    return (
        f"{int(adduction_displacement_deg):d}_"
        f"{int(sweep_displacement_deg):d}_"
        f"{_format_speed(speed_rad_s)}_"
        f"{direction}"
    )


def _parse_speed(speed_token: str) -> float:
    return float(speed_token.replace("p", "."))


def _flatten_waypoints(waypoints: List[Waypoint], speed_rad_s: float) -> List[float]:
    trajectory: List[float] = []
    for adduction_rad, sweeping_rad in waypoints:
        trajectory.extend([adduction_rad, sweeping_rad, float(speed_rad_s)])
    return trajectory


def make_sweep_trajectory(
    adduction_displacement_deg: int,
    sweep_displacement_deg: int,
    speed_rad_s: float,
    direction: SweepDirection,
) -> List[float]:
    """Build one centered sweep trajectory in controller coordinates."""
    adduction_home = ADDUCTION_HOME_RAD
    adduction_down = adduction_home + math.radians(adduction_displacement_deg)

    half_sweep = math.radians(sweep_displacement_deg) / 2.0
    sweep_back = SWEEP_CENTER_RAD - half_sweep
    sweep_front = SWEEP_CENTER_RAD + half_sweep

    if direction == "front":
        sweep_start = sweep_back
        sweep_end = sweep_front
    elif direction == "back":
        sweep_start = sweep_front
        sweep_end = sweep_back
    else:
        raise ValueError(f"unknown sweep direction: {direction}")

    waypoints = [
        (adduction_home, SWEEP_CENTER_RAD),
        (adduction_home, sweep_start),
        (adduction_down, sweep_start),
        (adduction_down, SWEEP_CENTER_RAD),
        (adduction_down, sweep_end),
        (adduction_home, sweep_end),
        (adduction_home, SWEEP_CENTER_RAD),
    ]
    return _flatten_waypoints(waypoints, speed_rad_s)


def parse_trajectory_name(name: str) -> TrajectorySpec:
    parts = name.strip().split("_")
    if len(parts) != 4:
        raise ValueError(
            f"trajectory name must look like 30_30_2_front, got {name!r}"
        )

    adduction_token, sweep_token, speed_token, direction_token = parts
    if direction_token not in SWEEP_DIRECTIONS:
        raise ValueError(
            f"trajectory direction must be one of {SWEEP_DIRECTIONS}, got {direction_token!r}"
        )

    try:
        adduction_deg = int(adduction_token)
        sweep_deg = int(sweep_token)
        speed_rad_s = _parse_speed(speed_token)
    except ValueError as exc:
        raise ValueError(
            f"trajectory name must look like 30_30_2_front, got {name!r}"
        ) from exc

    direction = direction_token  # type: ignore[assignment]
    points = make_sweep_trajectory(
        adduction_displacement_deg=adduction_deg,
        sweep_displacement_deg=sweep_deg,
        speed_rad_s=speed_rad_s,
        direction=direction,
    )
    canonical_name = _canonical_name(adduction_deg, sweep_deg, speed_rad_s, direction)
    return TrajectorySpec(
        name=canonical_name,
        adduction_deg=adduction_deg,
        sweep_deg=sweep_deg,
        speed_rad_s=speed_rad_s,
        direction=direction,
        points=points,
    )


TRAJECTORIES: Dict[str, List[float]] = {
    _canonical_name(adduction_deg, sweep_deg, TRAJ_SPEED_RAD_S, direction): make_sweep_trajectory(
        adduction_displacement_deg=adduction_deg,
        sweep_displacement_deg=sweep_deg,
        speed_rad_s=TRAJ_SPEED_RAD_S,
        direction=direction,  # type: ignore[arg-type]
    )
    for adduction_deg in ADDUCTION_DISPLACEMENTS_DEG
    for sweep_deg in SWEEP_DISPLACEMENTS_DEG
    for direction in SWEEP_DIRECTIONS
}


TRAJECTORY_REPRESENTATIONS: Dict[str, Tuple[int, int, float, str]] = {
    name: (
        parse_trajectory_name(name).adduction_deg,
        parse_trajectory_name(name).sweep_deg,
        parse_trajectory_name(name).speed_rad_s,
        parse_trajectory_name(name).direction,
    )
    for name in TRAJECTORIES
}
