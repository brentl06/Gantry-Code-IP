"""Serial G-code interface to the gantry's SKR Pro / Marlin controller.

This wraps the serial handshake for the gantry into a small, reusable
GantryController class. Import this from data_collector.py (or any other
script) instead of talking to the serial port directly -- keeps gantry
motion logic in one place, testable on its own without needing cameras or
ROS2 running.

Position model:
The gantry now has working endstops, so this class homes for real: on
connect() it sends G28 (home all axes) and switches to absolute (G90)
positioning. After homing -- and after every move -- position is read back
live from the controller via M114, rather than just trusted from commanded
deltas. That means the logged X/Y/Z is a real, physically repeatable
coordinate (assuming consistent homing/calibration), comparable across
separate script runs, not just within one session.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import serial

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
DEFAULT_FEEDRATE_MM_MIN = 1200.0
DEFAULT_MOTOR_CURRENT_MA = 700
DEFAULT_ACK_TIMEOUT_S = 5.0
DEFAULT_HOMING_TIMEOUT_S = 60.0

_M114_POSITION_RE = re.compile(
    r"X:(-?[\d.]+)\s+Y:(-?[\d.]+)\s+Z:(-?[\d.]+)"
)


@dataclass(frozen=True)
class GantryPosition:
    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_metadata(self) -> dict:
        """JSON-safe dict for embedding in trial/session metadata."""
        return {"x_mm": float(self.x), "y_mm": float(self.y), "z_mm": float(self.z)}


class GantryController:
    """Thin wrapper around the gantry's Marlin serial G-code interface.

    Usage:
        gantry = GantryController(port="/dev/ttyACM0")
        gantry.connect()          # homes (G28), switches to absolute (G90)
        position = gantry.move_to(x=25, y=0)
        ...
        gantry.close()

    Or as a context manager:
        with GantryController(port="/dev/ttyACM0") as gantry:
            gantry.move_to(x=25, y=0)
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        feedrate: float = DEFAULT_FEEDRATE_MM_MIN,
        motor_current_ma: int = DEFAULT_MOTOR_CURRENT_MA,
        ack_timeout_s: float = DEFAULT_ACK_TIMEOUT_S,
        homing_timeout_s: float = DEFAULT_HOMING_TIMEOUT_S,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.feedrate = float(feedrate)
        self.motor_current_ma = int(motor_current_ma)
        self.ack_timeout_s = float(ack_timeout_s)
        self.homing_timeout_s = float(homing_timeout_s)
        self._ser: Optional[serial.Serial] = None
        # Last position read back live from the controller (see get_position).
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0

    # -- connection lifecycle -------------------------------------------------
    def connect(self, home: bool = True) -> None:
        self._ser = serial.Serial(self.port, self.baud, timeout=2)
        time.sleep(2)
        self._ser.reset_input_buffer()
        self._send("M999", extra_wait=0.2)  # clear any halted state
        self._send(
            f"M906 X{self.motor_current_ma} Y{self.motor_current_ma} Z{self.motor_current_ma}"
        )
        self._send("M17")      # enable steppers
        self._send("M211 S1")  # endstops enabled
        if home:
            print("Homing gantry (G28) -- all axes will move to their endstops...")
            self._send("G28", timeout_s=self.homing_timeout_s)
        self._send("G90")      # absolute positioning mode
        position = self._query_position()
        print(f"Gantry connected on {self.port}; position: {position.as_metadata()}")

    def close(self) -> None:
        if self._ser is None:
            return
        try:
            self._send("M211 S1")
        finally:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> "GantryController":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- low-level G-code helpers ------------------------------------------------
    def _send(self, cmd: str, extra_wait: float = 0.0, timeout_s: Optional[float] = None) -> None:
        if self._ser is None:
            raise RuntimeError("GantryController is not connected; call connect() first.")
        deadline = self.ack_timeout_s if timeout_s is None else float(timeout_s)
        print(">>", cmd)
        self._ser.write((cmd + "\n").encode())
        t0 = time.time()
        while True:
            line = self._ser.readline().decode(errors="ignore").strip()
            if line:
                print("<<", line)
            if line.lower().startswith("ok"):
                if extra_wait > 0:
                    time.sleep(extra_wait)
                return
            if time.time() - t0 > deadline:
                raise TimeoutError(f"Timed out waiting for 'ok' from gantry for command: {cmd}")

    def _query_position(self) -> GantryPosition:
        """Send M114 and parse the live position Marlin reports back."""
        if self._ser is None:
            raise RuntimeError("GantryController is not connected; call connect() first.")
        print(">> M114")
        self._ser.write(b"M114\n")
        t0 = time.time()
        parsed: Optional[GantryPosition] = None
        while True:
            line = self._ser.readline().decode(errors="ignore").strip()
            if line:
                print("<<", line)
                match = _M114_POSITION_RE.search(line)
                if match is not None and parsed is None:
                    parsed = GantryPosition(
                        float(match.group(1)), float(match.group(2)), float(match.group(3))
                    )
            if line.lower().startswith("ok"):
                break
            if time.time() - t0 > self.ack_timeout_s:
                raise TimeoutError("Timed out waiting for M114 position report from gantry.")
        if parsed is None:
            raise RuntimeError(f"Could not parse a position out of M114's response.")
        self._x, self._y, self._z = parsed.as_tuple()
        return parsed

    # -- motion -----------------------------------------------------------------
    def move_to(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        feedrate: Optional[float] = None,
    ) -> GantryPosition:
        """Move to an absolute position (mm) in the homed coordinate frame.

        Any axis left as None is not moved. Returns the verified post-move
        position (read back via M114), not just the commanded target.
        """
        f = self.feedrate if feedrate is None else float(feedrate)
        parts = ["G0"]
        if x is not None:
            parts.append(f"X{float(x):g}")
        if y is not None:
            parts.append(f"Y{float(y):g}")
        if z is not None:
            parts.append(f"Z{float(z):g}")
        if len(parts) > 1:
            parts.append(f"F{f:g}")
            self._send(" ".join(parts))
        return self._query_position()

    def move_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        feedrate: Optional[float] = None,
    ) -> GantryPosition:
        """Move by an offset from the current position (still issued as an
        absolute G0 move under the hood, since the controller runs in G90)."""
        current = self.get_position()
        return self.move_to(
            x=(current.x + dx) if dx else None,
            y=(current.y + dy) if dy else None,
            z=(current.z + dz) if dz else None,
            feedrate=feedrate,
        )

    def dwell(self, ms: int) -> None:
        self._send(f"G4 P{int(ms)}")

    def get_position(self) -> GantryPosition:
        """Live position, read back from the controller via M114."""
        return self._query_position()


if __name__ == "__main__":
    # Quick bench test: homes, then traces a small square, printing the
    # verified position after each move. Run this on its own to sanity-check
    # wiring/homing/serial port before wiring the gantry into the full
    # data collector.
    import argparse

    ap = argparse.ArgumentParser(description="Bench-test the gantry controller.")
    ap.add_argument("--port", default=DEFAULT_PORT)
    args = ap.parse_args()

    with GantryController(port=args.port) as gantry:
        home = gantry.get_position()
        print(f"Homed position: {home.as_metadata()}")
        for dx, dy in [(25, 0), (0, 25), (-25, 0), (0, -25)]:
            pos = gantry.move_relative(dx=dx, dy=dy)
            print(f"Now at: {pos.as_metadata()}")
            gantry.dwell(2000)
    print("Done.")
