"""Simulation orchestrator supporting Physics Warp and Keplerian On-Rails Warp.

This module provides:
    - Dual-mode time acceleration:
        * PHYSICS Warp: Sub-stepped RK4 numerical integration for atmospheric flight or active thrust.
        * ON_RAILS Warp: Analytical Keplerian propagation for fast-forwarding coasting orbits in vacuum.
    - Automated safety warp-drop rules (atmospheric reentry, active thrust, ground proximity).
    - Integrated flight telemetry buffering and mission lifecycle tracking.

All calculations enforce 64-bit floating point precision (np.float64 / IEEE 754 double).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Union

import numpy as np

from core.integrator import wrap_angle_pi
from core.vector2d import Vector2D
from physics.celestial import CelestialBody
from physics.orbit import calculate_orbit, propagate_kepler
from simulation.autopilot import SASMode
from simulation.flight_dynamics import FlightVehicle, TelemetrySnapshot, VehicleState
from simulation.telemetry import TelemetryLogger


class WarpMode(str, Enum):
    """Operational mode for simulation time advancement."""

    PHYSICS = "PHYSICS"
    ON_RAILS = "ON_RAILS"


class Simulation:
    """Simulation orchestrator coordinating vehicles, celestial bodies, and time warp.

    Parameters
    ----------
    vehicle : FlightVehicle
        Active rocket / flight vehicle instance.
    body : Optional[CelestialBody], default=None
        Primary celestial body providing gravitation and atmosphere.
        If None, retrieved directly from vehicle.body.
    logger : Optional[TelemetryLogger], default=None
        Telemetry logger. If None, an empty TelemetryLogger is instantiated.
    time_warp : float, default=1.0
        Initial time acceleration factor.
    warp_mode : WarpMode, default=WarpMode.PHYSICS
        Initial time warp mode (PHYSICS or ON_RAILS).
    max_physics_substep : float, default=0.05
        Maximum allowable physical integration sub-step dt_sub in seconds [s].
    sample_dt : float, default=0.0
        Sampling interval for the default telemetry logger [s].
    """

    def __init__(
        self,
        vehicle: FlightVehicle,
        body: Optional[CelestialBody] = None,
        logger: Optional[TelemetryLogger] = None,
        time_warp: float = 1.0,
        warp_mode: WarpMode = WarpMode.PHYSICS,
        max_physics_substep: float = 0.05,
        sample_dt: float = 0.0,
    ) -> None:
        self.vehicle: FlightVehicle = vehicle
        self.body: CelestialBody = body if body is not None else vehicle.body
        self.logger: TelemetryLogger = (
            logger if logger is not None else TelemetryLogger(sample_dt=sample_dt)
        )

        self.time_warp: float = float(np.float64(time_warp))
        self.warp_mode: WarpMode = (
            WarpMode(warp_mode) if isinstance(warp_mode, str) else warp_mode
        )
        self.max_physics_substep: float = float(np.float64(max_physics_substep))
        self.elapsed_time: float = float(np.float64(self.vehicle.time))

        # Enforce safety rules on initial warp factor
        if self.time_warp > 4.0:
            self.set_time_warp(self.time_warp)

    # --- Property Helpers ---

    @property
    def cutoff_altitude(self) -> float:
        """Atmospheric cutoff boundary altitude above mean planetary surface [m]."""
        if self.body.atmosphere is not None:
            return float(self.body.atmosphere.cutoff_altitude)
        return 0.0

    def is_thrusting(self) -> bool:
        """Determine whether the vehicle has active engines producing thrust.

        Returns
        -------
        bool
            True if any engine in the active stage is ignited and throttled with propellant.
        """
        active_stage = self.vehicle.rocket.active_stage
        if active_stage is not None and not active_stage.is_separated:
            if active_stage.propellant_mass() > 0.0:
                for eng in active_stage.engines:
                    if eng.is_ignited and eng.throttle > 0.0:
                        return True
        return False

    # --- Time Warp Management ---

    def set_time_warp(self, warp_factor: float) -> None:
        """Configure time warp factor and enforce automated safety mode selection.

        Rules:
            1. warp_factor <= 4.0:
               Always allowed in WarpMode.PHYSICS.
            2. warp_factor > 4.0:
               Requires:
                 - VehicleState == IN_FLIGHT
                 - Geometric altitude h >= cutoff_altitude (vacuum)
                 - Active thrust == 0 (engines shutdown or zero throttle)
               If safety checks pass:
                 warp_mode is set to WarpMode.ON_RAILS.
               If safety checks fail:
                 warp_factor is clamped to 4.0 and warp_mode remains WarpMode.PHYSICS.

        Parameters
        ----------
        warp_factor : float
            Desired time warp multiplier (e.g. 1.0, 2.0, 4.0, 10.0, 100.0, 1000.0).
        """
        factor = float(np.float64(max(0.0, warp_factor)))

        if factor <= 4.0:
            self.time_warp = factor
            self.warp_mode = WarpMode.PHYSICS
            return

        # Check safety conditions for high-speed on-rails warp
        is_in_flight = self.vehicle.state == VehicleState.IN_FLIGHT
        alt = self.body.altitude(self.vehicle.position)
        is_outside_atmosphere = alt >= (self.cutoff_altitude - 1e-4)
        is_engines_idle = not self.is_thrusting()

        if is_in_flight and is_outside_atmosphere and is_engines_idle:
            self.time_warp = factor
            self.warp_mode = WarpMode.ON_RAILS
        else:
            # Clamp to maximum physics warp
            self.time_warp = 4.0
            self.warp_mode = WarpMode.PHYSICS

    # --- Simulation Advancing ---

    def step(
        self,
        wall_dt: float,
        target_pitch: Optional[float] = None,
        throttle: Optional[float] = None,
    ) -> TelemetrySnapshot:
        """Advance the simulation by one discrete wall-clock increment wall_dt.

        Parameters
        ----------
        wall_dt : float
            Wall-clock elapsed increment in seconds [s] (wall_dt > 0).
        target_pitch : Optional[float], default=None
            Commanded attitude pitch angle theta_target [rad].
        throttle : Optional[float], default=None
            Engine throttle setting [0.0, 1.0].

        Returns
        -------
        TelemetrySnapshot
            Snapshot of vehicle telemetry at the end of the simulation step.
        """
        w_dt = float(np.float64(wall_dt))
        if w_dt < 0.0:
            raise ValueError(f"wall_dt must be non-negative, got {wall_dt}")

        dt_sim = w_dt * self.time_warp

        # If thrust is commanded while on rails, immediately drop to physics warp
        if self.warp_mode == WarpMode.ON_RAILS and throttle is not None and throttle > 0.0:
            self.set_time_warp(1.0)

        # 1. Physics Warp Mode: Sub-stepped numerical RK4 integration
        if self.warp_mode == WarpMode.PHYSICS:
            if dt_sim > 0.0:
                substep_limit = max(1e-6, self.max_physics_substep)
                num_substeps = max(1, math.ceil(dt_sim / substep_limit))
                dt_sub = dt_sim / num_substeps

                snap: Optional[TelemetrySnapshot] = None
                for _ in range(num_substeps):
                    snap = self.vehicle.step(
                        dt=dt_sub,
                        target_pitch=target_pitch,
                        throttle=throttle,
                    )
            else:
                snap = self.vehicle.get_telemetry_snapshot()

        # 2. On-Rails Warp Mode: Analytical Keplerian conic propagation
        else:
            if dt_sim > 0.0:
                step_dt = dt_sim
                # Check for atmospheric reentry boundary interception
                cutoff = self.cutoff_altitude
                r_atm = self.body.radius + cutoff
                orbit = calculate_orbit(self.vehicle.position, self.vehicle.velocity, self.body)

                # If orbit intersects atmosphere, calculate time until atmospheric entry
                if orbit.periapsis_radius < r_atm and orbit.eccentricity < 1.0:
                    p = orbit.semi_major_axis * (1.0 - orbit.eccentricity ** 2)
                    cos_nu_entry = (p / r_atm - 1.0) / orbit.eccentricity
                    if -1.0 <= cos_nu_entry <= 1.0:
                        nu_entry = 2.0 * math.pi - math.acos(cos_nu_entry)
                        e_anom_entry = 2.0 * math.atan(
                            math.sqrt((1.0 - orbit.eccentricity) / (1.0 + orbit.eccentricity))
                            * math.tan(nu_entry / 2.0)
                        ) % (2.0 * math.pi)
                        m_entry = e_anom_entry - orbit.eccentricity * math.sin(e_anom_entry)

                        nu_curr = orbit.true_anomaly
                        e_anom_curr = 2.0 * math.atan(
                            math.sqrt((1.0 - orbit.eccentricity) / (1.0 + orbit.eccentricity))
                            * math.tan(nu_curr / 2.0)
                        ) % (2.0 * math.pi)
                        m_curr = e_anom_curr - orbit.eccentricity * math.sin(e_anom_curr)

                        n = math.sqrt(self.body.mu / (orbit.semi_major_axis ** 3))
                        dt_to_entry = (m_entry - m_curr) % (2.0 * math.pi) / n

                        if 0.0 < dt_to_entry <= dt_sim:
                            step_dt = dt_to_entry
                            self.time_warp = 1.0
                            self.warp_mode = WarpMode.PHYSICS

                new_pos, new_vel = propagate_kepler(
                    self.vehicle.position,
                    self.vehicle.velocity,
                    self.body,
                    step_dt,
                )
                self.vehicle.position = new_pos
                self.vehicle.velocity = new_vel

                # Attitude hold / orbital reference alignment
                if self.vehicle.autopilot.mode == SASMode.PROGRADE:
                    self.vehicle.attitude = float(
                        np.float64(math.atan2(self.vehicle.velocity.y, self.vehicle.velocity.x))
                    )
                    self.vehicle.autopilot.target_attitude = self.vehicle.attitude
                elif self.vehicle.autopilot.mode == SASMode.RETROGRADE:
                    self.vehicle.attitude = float(
                        np.float64(
                            wrap_angle_pi(
                                math.atan2(self.vehicle.velocity.y, self.vehicle.velocity.x)
                                + math.pi
                            )
                        )
                    )
                    self.vehicle.autopilot.target_attitude = self.vehicle.attitude
                elif target_pitch is not None:
                    self.vehicle.attitude = float(np.float64(target_pitch))
                    self.vehicle.autopilot.target_attitude = float(np.float64(target_pitch))
                else:
                    self.vehicle.angular_velocity = 0.0

                self.vehicle.time += step_dt

                # Safety warp-drop check:
                # If periapsis intersects atmosphere and vehicle has reached or crossed into atmosphere
                alt = self.vehicle.altitude
                cutoff = self.cutoff_altitude
                orbit = calculate_orbit(self.vehicle.position, self.vehicle.velocity, self.body)
                r_atm = self.body.radius + cutoff

                if orbit.periapsis_radius < r_atm and alt <= (cutoff + 1e-4):
                    self.time_warp = 1.0
                    self.warp_mode = WarpMode.PHYSICS

            snap = self.vehicle.get_telemetry_snapshot()

        self.elapsed_time = float(np.float64(self.vehicle.time))
        self.logger.record(snap)
        return snap

    def run(
        self,
        duration: float,
        wall_dt: float = 0.1,
        target_pitch: Optional[float] = None,
        throttle: Optional[float] = None,
    ) -> list[TelemetrySnapshot]:
        # Guard against freeze when simulation is paused (time_warp == 0)
        if self.time_warp <= 1e-9:
            snap = self.step(wall_dt, target_pitch=target_pitch, throttle=throttle)
            return [snap]
        """Run the simulation for a specified simulation duration in seconds.

        Parameters
        ----------
        duration : float
            Total simulation time increment to advance [s].
        wall_dt : float, default=0.1
            Base wall-clock step size in seconds [s].
        target_pitch : Optional[float], default=None
            Target pitch angle [rad].
        throttle : Optional[float], default=None
            Engine throttle setting [0.0, 1.0].

        Returns
        -------
        list[TelemetrySnapshot]
            Recorded telemetry snapshots during the run.
        """
        target_time = self.elapsed_time + duration
        recorded: list[TelemetrySnapshot] = []

        while self.elapsed_time < target_time - 1e-9:
            step_wdt = wall_dt
            # Adjust last step so simulation time matches duration exactly
            rem_sim = target_time - self.elapsed_time
            if rem_sim < step_wdt * self.time_warp:
                step_wdt = rem_sim / max(1e-9, self.time_warp)

            snap = self.step(step_wdt, target_pitch=target_pitch, throttle=throttle)
            recorded.append(snap)

            if self.vehicle.state in (VehicleState.LANDED, VehicleState.DESTROYED):
                break

        return recorded


__all__ = [
    "Simulation",
    "WarpMode",
]
