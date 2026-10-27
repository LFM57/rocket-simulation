"""High-precision numerical integration suite for planar 6-DOF rocket dynamics.

Dynamic Planar State Vector Convention:
    S = [x, y, v_x, v_y, theta, omega]^T

Where:
    - x      (idx 0) : Position along inertial X-axis [meters, m]
    - y      (idx 1) : Position along inertial Y-axis [meters, m]
    - v_x    (idx 2) : Velocity along inertial X-axis [meters/second, m/s]
    - v_y    (idx 3) : Velocity along inertial Y-axis [meters/second, m/s]
    - theta  (idx 4) : Attitude angle from +X axis [radians, rad]
    - omega  (idx 5) : Angular velocity [radians/second, rad/s]

Derivative Vector Convention:
    dS/dt = [v_x, v_y, a_x, a_y, omega, alpha]^T
Where:
    - a_x, a_y : Linear acceleration [m/s^2]
    - alpha    : Angular acceleration [rad/s^2]
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Callable

import numpy as np

STATE_DIM: int = 6

IDX_X: int = 0
IDX_Y: int = 1
IDX_VX: int = 2
IDX_VY: int = 3
IDX_THETA: int = 4
IDX_OMEGA: int = 5


class StateIndex(IntEnum):
    """Named indices for the 6D planar state vector."""

    X = 0
    Y = 1
    VX = 2
    VY = 3
    THETA = 4
    OMEGA = 5


DerivativeFunc = Callable[[float, np.ndarray], np.ndarray]


def make_state(
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    theta: float = 0.0,
    omega: float = 0.0,
) -> np.ndarray:
    """Create a standardized 6D state vector enforcing np.float64."""
    return np.array([x, y, vx, vy, theta, omega], dtype=np.float64)


def wrap_angle_pi(theta: float) -> float:
    """Wrap angle to [-pi, pi) radians.

    Note: The integrator integrates unconstrained continuous theta to avoid
    artificial discontinuity in derivatives. Use this utility for sensor reading,
    guidance feedback, and telemetry reporting.
    """
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def rk4_step(
    f: DerivativeFunc,
    t: float,
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance the state vector over time step dt using 4th-Order Runge-Kutta (RK4).

    General-purpose explicit ODE integrator for non-conservative, time-dependent,
    aerodynamic, and control-actuated derivatives:
        dS/dt = f(t, S)

    Parameters
    ----------
    f : Callable[[float, np.ndarray], np.ndarray]
        Callable computing dS/dt given current simulation time t (s) and state S.
    t : float
        Current simulation time in seconds (s).
    state : np.ndarray
        State vector of shape (6,), dtype np.float64.
    dt : float
        Time step in seconds (s).

    Returns
    -------
    np.ndarray
        Updated state vector at t + dt of shape (6,), dtype np.float64.
    """
    s = np.asarray(state, dtype=np.float64)
    dt_64 = float(np.float64(dt))
    half_dt = 0.5 * dt_64

    k1 = np.asarray(f(t, s), dtype=np.float64)
    k2 = np.asarray(f(t + half_dt, s + half_dt * k1), dtype=np.float64)
    k3 = np.asarray(f(t + half_dt, s + half_dt * k2), dtype=np.float64)
    k4 = np.asarray(f(t + dt_64, s + dt_64 * k3), dtype=np.float64)

    return s + (dt_64 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def velocity_verlet_step(
    f: DerivativeFunc,
    t: float,
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance the state vector over dt using the 2nd-order symplectic Velocity Verlet method.

    Velocity Verlet integrates second-order Hamiltonian dynamical systems:
        d^2(q)/dt^2 = a(t, q)
    Preserves symplectic 2-form structure and bounds energy error over long durations,
    making it ideal for orbital mechanics and conservative gravity fields.

    Parameters
    ----------
    f : Callable[[float, np.ndarray], np.ndarray]
        Callable computing dS/dt given time t (s) and state S.
        Must return dS/dt = [v_x, v_y, a_x, a_y, omega, alpha]^T.
    t : float
        Current simulation time in seconds (s).
    state : np.ndarray
        Current state vector of shape (6,), dtype np.float64.
    dt : float
        Time step in seconds (s).

    Returns
    -------
    np.ndarray
        Updated state vector at t + dt of shape (6,), dtype np.float64.
    """
    s = np.asarray(state, dtype=np.float64)
    dt_64 = float(np.float64(dt))
    half_dt = 0.5 * dt_64
    dt_sq_half = 0.5 * dt_64 * dt_64

    # Evaluate accelerations at initial state S_n
    deriv_n = np.asarray(f(t, s), dtype=np.float64)
    ax_n = deriv_n[IDX_VX]
    ay_n = deriv_n[IDX_VY]
    alpha_n = deriv_n[IDX_OMEGA]

    # Step 1: Advance generalized coordinates q_{n+1} = q_n + v_n*dt + 0.5*a_n*dt^2
    x_next = s[IDX_X] + s[IDX_VX] * dt_64 + ax_n * dt_sq_half
    y_next = s[IDX_Y] + s[IDX_VY] * dt_64 + ay_n * dt_sq_half
    theta_next = s[IDX_THETA] + s[IDX_OMEGA] * dt_64 + alpha_n * dt_sq_half

    # Step 2: Intermediate half-step velocities v_{n+1/2} = v_n + 0.5*a_n*dt
    vx_half = s[IDX_VX] + ax_n * half_dt
    vy_half = s[IDX_VY] + ay_n * half_dt
    omega_half = s[IDX_OMEGA] + alpha_n * half_dt

    # Step 3: Evaluate accelerations at advanced position and half-step velocity
    s_inter = np.array(
        [x_next, y_next, vx_half, vy_half, theta_next, omega_half],
        dtype=np.float64,
    )
    deriv_next = np.asarray(f(t + dt_64, s_inter), dtype=np.float64)
    ax_next = deriv_next[IDX_VX]
    ay_next = deriv_next[IDX_VY]
    alpha_next = deriv_next[IDX_OMEGA]

    # Step 4: Finalize velocities v_{n+1} = v_{n+1/2} + 0.5*a_{n+1}*dt
    vx_next = vx_half + ax_next * half_dt
    vy_next = vy_half + ay_next * half_dt
    omega_next = omega_half + alpha_next * half_dt

    return np.array(
        [x_next, y_next, vx_next, vy_next, theta_next, omega_next],
        dtype=np.float64,
    )


def integrate_trajectory(
    stepper: Callable[[DerivativeFunc, float, np.ndarray, float], np.ndarray],
    f: DerivativeFunc,
    state0: np.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a state vector from t_span[0] to t_span[1] with fixed step dt.

    Parameters
    ----------
    stepper : Callable
        Single-step integrator (rk4_step or velocity_verlet_step).
    f : DerivativeFunc
        Derivative function f(t, S) -> dS/dt.
    state0 : np.ndarray
        Initial state vector S(t0) of shape (6,).
    t_span : tuple[float, float]
        Simulation start and end times (t_start, t_end) in seconds.
    dt : float
        Integration step size in seconds (dt > 0).

    Returns
    -------
    times : np.ndarray
        1D array of simulation timestamps in seconds, shape (N,).
    states : np.ndarray
        2D array of state vectors, shape (N, 6).
    """
    t_start, t_end = t_span
    if dt <= 0.0:
        raise ValueError(f"Integration time step dt must be positive, got {dt}")
    if t_end < t_start:
        raise ValueError(f"t_end ({t_end}) must be greater than or equal to t_start ({t_start})")

    total_time = t_end - t_start
    num_steps = int(math.ceil(total_time / dt))

    times = np.zeros(num_steps + 1, dtype=np.float64)
    states = np.zeros((num_steps + 1, STATE_DIM), dtype=np.float64)

    times[0] = t_start
    states[0] = np.asarray(state0, dtype=np.float64)

    curr_t = float(t_start)
    curr_state = states[0].copy()

    for i in range(num_steps):
        step_dt = min(dt, t_end - curr_t)
        if step_dt <= 1e-15:
            times = times[: i + 1]
            states = states[: i + 1]
            break
        curr_state = stepper(f, curr_t, curr_state, step_dt)
        curr_t += step_dt
        times[i + 1] = curr_t
        states[i + 1] = curr_state

    return times, states