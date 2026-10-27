"""PID attitude controller and SAS (Stability Augmentation System) flight autopilot.

This module implements:
    - PIDController: Generic Proportional-Integral-Derivative controller with anti-windup
      and symmetric output clamping.
    - AutopilotSAS: 3-DOF attitude autopilot controlling nozzle gimbal deflection
      to stabilize and orient rocket vehicles across flight modes.

All operations enforce 64-bit floating point precision (np.float64 / IEEE 754 double).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Sequence, Union

import numpy as np

from core.vector2d import Vector2D


class PIDController:
    """Proportional-Integral-Derivative (PID) controller with anti-windup clamping.

    Parameters
    ----------
    kp : float, default=1.0
        Proportional gain K_p.
    ki : float, default=0.0
        Integral gain K_i.
    kd : float, default=0.0
        Derivative gain K_d.
    limit : Optional[float], default=None
        Symmetric output clamp [-limit, +limit]. If None, output is unclamped.
    integral_limit : Optional[float], default=None
        Symmetric bound on the integral accumulator [-integral_limit, +integral_limit].
        If None and limit is specified with ki > 0, defaults to limit / ki (standard anti-windup).
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        limit: Optional[float] = None,
        integral_limit: Optional[float] = None,
    ) -> None:
        self.kp: float = float(np.float64(kp))
        self.ki: float = float(np.float64(ki))
        self.kd: float = float(np.float64(kd))
        self.limit: Optional[float] = (
            float(np.float64(abs(limit))) if limit is not None else None
        )

        if integral_limit is not None:
            self.integral_limit: Optional[float] = float(np.float64(abs(integral_limit)))
        elif self.limit is not None and self.ki > 0.0:
            self.integral_limit = float(np.float64(self.limit / self.ki))
        else:
            self.integral_limit = None

        self.integral: float = 0.0
        self.prev_error: Optional[float] = None

    def reset(self) -> None:
        """Reset internal integrator accumulator and derivative state."""
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        """Compute the control signal for a given tracking error and elapsed time step dt.

        Parameters
        ----------
        error : float
            Current tracking error (setpoint - measured).
        dt : float
            Time step increment in seconds [s].

        Returns
        -------
        float
            Control output command (64-bit float).
        """
        err = float(np.float64(error))
        dt_val = float(np.float64(dt))

        # Proportional term
        p_term = self.kp * err

        # Integral term with anti-windup accumulator bounding
        if dt_val > 0.0:
            self.integral += err * dt_val
            if self.integral_limit is not None:
                self.integral = max(
                    -self.integral_limit, min(self.integral_limit, self.integral)
                )

        i_term = self.ki * self.integral

        # Derivative term
        if self.prev_error is not None and dt_val > 0.0:
            d_term = self.kd * ((err - self.prev_error) / dt_val)
        else:
            d_term = 0.0

        self.prev_error = err

        output = p_term + i_term + d_term

        # Symmetric output clamping
        if self.limit is not None:
            output = max(-self.limit, min(self.limit, output))

        return float(np.float64(output))


class SASMode(str, Enum):
    """Operational guidance and attitude control modes for AutopilotSAS."""

    OFF = "OFF"
    HOLD_ATTITUDE = "HOLD_ATTITUDE"
    PROGRADE = "PROGRADE"
    RETROGRADE = "RETROGRADE"


class AutopilotSAS:
    """Stability Augmentation System (SAS) attitude autopilot for rocket gimbal control.

    Controls engine gimbal deflection angle delta to stabilize rocket attitude and track
    commanded target orientations:
        Delta_theta = atan2(sin(theta_target - theta), cos(theta_target - theta))
        delta_cmd = clamp(Kp * Delta_theta - Kd * omega, -delta_max, delta_max)

    Parameters
    ----------
    kp : float, default=2.0
        Proportional attitude tracking gain.
    kd : float, default=2.0
        Derivative angular rate damping gain.
    ki : float, default=0.0
        Integral gain (optional, default 0.0).
    delta_max : float, default=0.08726646259971647
        Maximum nozzle gimbal deflection limit in radians (~5 degrees).
    mode : Union[SASMode, str], default=SASMode.HOLD_ATTITUDE
        Initial flight guidance mode.
    target_attitude : float, default=0.0
        Initial target attitude angle in radians [rad].
    gimbal_limit : Optional[float], default=None
        Alias keyword argument for delta_max.
    """

    OFF: SASMode = SASMode.OFF
    HOLD_ATTITUDE: SASMode = SASMode.HOLD_ATTITUDE
    PROGRADE: SASMode = SASMode.PROGRADE
    RETROGRADE: SASMode = SASMode.RETROGRADE

    def __init__(
        self,
        kp: float = 2.0,
        kd: float = 2.0,
        ki: float = 0.0,
        delta_max: float = 0.08726646259971647,
        mode: Union[SASMode, str] = SASMode.HOLD_ATTITUDE,
        target_attitude: float = 0.0,
        gimbal_limit: Optional[float] = None,
    ) -> None:
        self.kp: float = float(np.float64(kp))
        self.kd: float = float(np.float64(kd))
        self.ki: float = float(np.float64(ki))

        effective_limit = gimbal_limit if gimbal_limit is not None else delta_max
        self.delta_max: float = float(np.float64(abs(effective_limit)))

        if isinstance(mode, str):
            self.mode: SASMode = SASMode(mode.upper())
        else:
            self.mode = mode

        self.target_attitude: float = float(np.float64(target_attitude))
        self.integral: float = 0.0
        self.last_gimbal_command: float = 0.0

    @property
    def target_pitch(self) -> float:
        """Alias for target_attitude in radians [rad]."""
        return self.target_attitude

    @target_pitch.setter
    def target_pitch(self, pitch: float) -> None:
        self.target_attitude = float(np.float64(pitch))

    def set_target_attitude(self, theta: float) -> None:
        """Set the commanded target attitude in radians."""
        self.target_attitude = float(np.float64(theta))

    def set_target_pitch(self, pitch: float) -> None:
        """Set the commanded target pitch in radians."""
        self.target_attitude = float(np.float64(pitch))

    def set_mode(self, mode: Union[SASMode, str]) -> None:
        """Set the current guidance mode (OFF, HOLD_ATTITUDE, PROGRADE, RETROGRADE)."""
        if isinstance(mode, str):
            self.mode = SASMode(mode.upper())
        else:
            self.mode = mode

    def reset(self) -> None:
        """Reset internal integrator state and last output command."""
        self.integral = 0.0
        self.last_gimbal_command = 0.0

    def update(
        self,
        theta: float,
        omega: float = 0.0,
        dt: float = 0.0,
        velocity: Optional[Union[Vector2D, Sequence[float]]] = None,
    ) -> float:
        """Advance attitude controller and return commanded engine gimbal deflection.

        Parameters
        ----------
        theta : float
            Current body attitude angle in radians [rad].
        omega : float, default=0.0
            Current body angular velocity in radians per second [rad/s].
        dt : float, default=0.0
            Simulation time increment in seconds [s].
        velocity : Optional[Union[Vector2D, Sequence[float]]], default=None
            Inertial velocity vector used to update PROGRADE and RETROGRADE targets.

        Returns
        -------
        float
            Commanded nozzle gimbal angle delta in radians, clamped to [-delta_max, +delta_max].
        """
        if self.mode == SASMode.OFF:
            self.last_gimbal_command = 0.0
            return 0.0

        th = float(np.float64(theta))
        om = float(np.float64(omega))
        dt_val = float(np.float64(dt))

        # Update target attitude based on flight modes
        if self.mode == SASMode.PROGRADE and velocity is not None:
            vel_vec = velocity if isinstance(velocity, Vector2D) else Vector2D.from_tuple(velocity)
            if vel_vec.magnitude() > 1e-4:
                self.target_attitude = math.atan2(vel_vec.y, vel_vec.x)

        elif self.mode == SASMode.RETROGRADE and velocity is not None:
            vel_vec = velocity if isinstance(velocity, Vector2D) else Vector2D.from_tuple(velocity)
            if vel_vec.magnitude() > 1e-4:
                self.target_attitude = math.atan2(-vel_vec.y, -vel_vec.x)

        # Angular error Delta_theta wrapped to [-pi, pi]
        delta_theta = float(
            np.float64(math.atan2(math.sin(self.target_attitude - th), math.cos(self.target_attitude - th)))
        )

        # PD attitude control with rate damping on omega
        raw_cmd = self.kp * delta_theta - self.kd * om

        # Optional integral action with anti-windup clamping
        if self.ki > 0.0 and dt_val > 0.0:
            self.integral += delta_theta * dt_val
            int_limit = self.delta_max / self.ki
            self.integral = max(-int_limit, min(int_limit, self.integral))
            raw_cmd += self.ki * self.integral

        # Output clamping
        gimbal_cmd = float(
            np.float64(max(-self.delta_max, min(self.delta_max, raw_cmd)))
        )
        self.last_gimbal_command = gimbal_cmd
        return gimbal_cmd

    def compute_gimbal(
        self,
        theta: float,
        omega: float = 0.0,
        dt: float = 0.0,
        velocity: Optional[Union[Vector2D, Sequence[float]]] = None,
    ) -> float:
        """Alias for update."""
        return self.update(theta=theta, omega=omega, dt=dt, velocity=velocity)


__all__ = [
    "AutopilotSAS",
    "PIDController",
    "SASMode",
]
