"""Telemetry logging, mission buffering, and post-flight performance analysis.

This module provides high-frequency telemetry recording, downsampling/ring-buffer
storage, CSV/dictionary export, and automated flight summary extraction
(Max-Q, peak speed, peak altitude, Max G-force, Delta-V expended, and final vehicle state).
All calculations enforce 64-bit floating-point precision (np.float64 / IEEE 754 double).
"""

from __future__ import annotations

import collections
import csv
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from core.constants import g0
from simulation.flight_dynamics import TelemetrySnapshot, VehicleState


class TelemetryLogger:
    """Configurable flight telemetry logger with ring-buffer or downsampled storage.

    Parameters
    ----------
    sample_dt : float, default=0.0
        Minimum simulation time increment between recordings [s].
        If 0.0, every submitted snapshot is recorded.
    max_capacity : Optional[int], default=None
        Maximum number of snapshots to store in memory.
        If specified, acts as a FIFO ring buffer when full.
    """

    def __init__(
        self,
        sample_dt: float = 0.0,
        max_capacity: Optional[int] = None,
    ) -> None:
        self.sample_dt: float = float(np.float64(sample_dt))
        self.max_capacity: Optional[int] = (
            int(max_capacity) if max_capacity is not None and max_capacity > 0 else None
        )

        self._snapshots: collections.deque[TelemetrySnapshot] = (
            collections.deque(maxlen=self.max_capacity)
            if self.max_capacity is not None
            else collections.deque()
        )
        self._last_recorded_time: float = -float("inf")
        self._delta_v_expended: float = 0.0
        self._prev_snapshot: Optional[TelemetrySnapshot] = None
        self._final_state: Optional[Union[str, VehicleState]] = None

    @property
    def snapshots(self) -> list[TelemetrySnapshot]:
        """Return shallow copy of all currently buffered snapshots."""
        return list(self._snapshots)

    def __len__(self) -> int:
        """Number of recorded telemetry snapshots."""
        return len(self._snapshots)

    def clear(self) -> None:
        """Clear all buffered telemetry data and reset cumulative metrics."""
        self._snapshots.clear()
        self._last_recorded_time = -float("inf")
        self._delta_v_expended = 0.0
        self._prev_snapshot = None
        self._final_state = None

    def record(
        self,
        snapshot: TelemetrySnapshot,
        force: bool = False,
        state: Optional[Union[str, VehicleState]] = None,
    ) -> bool:
        """Record a telemetry snapshot if the sampling interval requirement is met.

        Parameters
        ----------
        snapshot : TelemetrySnapshot
            Discrete flight telemetry snapshot.
        force : bool, default=False
            If True, bypasses the sample_dt check and forces recording.
        state : Optional[Union[str, VehicleState]], default=None
            Optional explicit vehicle operational state.

        Returns
        -------
        bool
            True if the snapshot was recorded, False if skipped due to sampling interval.
        """
        # Maintain cumulative Delta-V via sub-step tracking or Riemann fallback
        snap_dv = getattr(snapshot, "delta_v_expended", 0.0)
        if snap_dv > 0.0:
            self._delta_v_expended = float(snap_dv)
        elif self._prev_snapshot is not None:
            dt = snapshot.time - self._prev_snapshot.time
            if dt > 0.0:
                prev_stage = getattr(self._prev_snapshot, "stage", None)
                curr_stage = getattr(snapshot, "stage", None)
                if prev_stage is not None and curr_stage is not None and prev_stage != curr_stage:
                    m_mid = snapshot.mass
                else:
                    m_mid = 0.5 * (self._prev_snapshot.mass + snapshot.mass)
                t_mid = 0.5 * (self._prev_snapshot.thrust + snapshot.thrust)
                if m_mid > 0.0 and t_mid > 0.0:
                    self._delta_v_expended += (t_mid / m_mid) * dt

        self._prev_snapshot = snapshot

        v_state = state or getattr(snapshot, "vehicle_state", None) or getattr(snapshot, "state", None)
        if v_state is not None:
            self._final_state = v_state

        # Check sample_dt gate
        if not force and len(self._snapshots) > 0:
            if (snapshot.time - self._last_recorded_time) < (self.sample_dt - 1e-9):
                return False

        self._snapshots.append(snapshot)
        self._last_recorded_time = snapshot.time
        return True

    def to_dict(self) -> dict[str, list[float]]:
        """Export all recorded snapshot fields into columnar dictionary format.

        Returns
        -------
        dict[str, list[float]]
            Dictionary mapping telemetry metric names to lists of 64-bit float values.
        """
        return {
            "time": [float(s.time) for s in self._snapshots],
            "altitude": [float(s.altitude) for s in self._snapshots],
            "speed": [float(s.speed) for s in self._snapshots],
            "mach": [float(s.mach) for s in self._snapshots],
            "dynamic_pressure": [float(s.dynamic_pressure) for s in self._snapshots],
            "thrust": [float(s.thrust) for s in self._snapshots],
            "mass": [float(s.mass) for s in self._snapshots],
            "pitch": [float(s.pitch) for s in self._snapshots],
            "angle_of_attack": [float(s.angle_of_attack) for s in self._snapshots],
            "apoapsis": [float(s.apoapsis) for s in self._snapshots],
            "periapsis": [float(s.periapsis) for s in self._snapshots],
            "stage": [float(s.stage) for s in self._snapshots],
        }

    def to_csv(self, filepath: Union[str, Path]) -> None:
        """Write all recorded telemetry snapshots to a comma-separated values (CSV) file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Destination path for the output CSV file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "time",
            "altitude",
            "speed",
            "mach",
            "dynamic_pressure",
            "thrust",
            "mass",
            "pitch",
            "angle_of_attack",
            "apoapsis",
            "periapsis",
            "stage",
            "vehicle_state",
        ]

        with open(path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in self._snapshots:
                writer.writerow(
                    {
                        "time": f"{s.time:.6f}",
                        "altitude": f"{s.altitude:.4f}",
                        "speed": f"{s.speed:.4f}",
                        "mach": f"{s.mach:.4f}",
                        "dynamic_pressure": f"{s.dynamic_pressure:.2f}",
                        "thrust": f"{s.thrust:.2f}",
                        "mass": f"{s.mass:.3f}",
                        "pitch": f"{s.pitch:.6f}",
                        "angle_of_attack": f"{s.angle_of_attack:.6f}",
                        "apoapsis": f"{s.apoapsis:.4f}",
                        "periapsis": f"{s.periapsis:.4f}",
                        "stage": s.stage,
                        "vehicle_state": str(getattr(s, "vehicle_state", "") or ""),
                    }
                )

    def get_summary(self) -> dict[str, Any]:
        """Compute comprehensive post-flight mission performance metrics.

        Returns
        -------
        dict[str, Any]
            Key performance summary dictionary containing:
                - max_altitude / peak_altitude [m]
                - max_velocity / max_speed [m/s]
                - max_dynamic_pressure / max_q [Pa]
                - max_q_time / max_q_timestamp [s]
                - max_g_force [g0]
                - total_mission_elapsed_time / elapsed_time [s]
                - apoapsis [m]
                - periapsis [m]
                - delta_v_expended [m/s]
                - final_vehicle_state
        """
        if not self._snapshots:
            return {
                "max_altitude": 0.0,
                "peak_altitude": 0.0,
                "max_velocity": 0.0,
                "max_speed": 0.0,
                "max_dynamic_pressure": 0.0,
                "max_q": 0.0,
                "max_q_time": 0.0,
                "max_q_timestamp": 0.0,
                "max_g_force": 0.0,
                "total_mission_elapsed_time": 0.0,
                "elapsed_time": 0.0,
                "apoapsis": 0.0,
                "periapsis": 0.0,
                "delta_v_expended": 0.0,
                "final_vehicle_state": None,
                "final_state": None,
            }

        max_alt = max(s.altitude for s in self._snapshots)
        max_speed = max(s.speed for s in self._snapshots)

        max_q = 0.0
        max_q_time = 0.0
        for s in self._snapshots:
            if s.dynamic_pressure > max_q:
                max_q = float(s.dynamic_pressure)
                max_q_time = float(s.time)

        max_g = max(
            (s.thrust / (s.mass * g0) if s.mass > 0.0 else 0.0 for s in self._snapshots),
            default=0.0,
        )

        total_time = float(self._snapshots[-1].time - self._snapshots[0].time)
        peak_ap = max(
            (s.apoapsis for s in self._snapshots if not math.isinf(s.apoapsis)),
            default=self._snapshots[-1].apoapsis,
        )
        final_pe = float(self._snapshots[-1].periapsis)

        final_st = (
            self._final_state
            or getattr(self._snapshots[-1], "vehicle_state", None)
            or getattr(self._snapshots[-1], "state", None)
        )

        return {
            "max_altitude": float(max_alt),
            "peak_altitude": float(max_alt),
            "max_velocity": float(max_speed),
            "max_speed": float(max_speed),
            "max_dynamic_pressure": float(max_q),
            "max_q": float(max_q),
            "max_q_time": float(max_q_time),
            "max_q_timestamp": float(max_q_time),
            "max_g_force": float(max_g),
            "total_mission_elapsed_time": float(total_time),
            "elapsed_time": float(total_time),
            "mission_time": float(self._snapshots[-1].time),
            "apoapsis": float(peak_ap),
            "periapsis": float(final_pe),
            "delta_v_expended": float(self._delta_v_expended),
            "final_vehicle_state": final_st,
            "final_state": final_st,
        }


__all__ = [
    "TelemetryLogger",
]
