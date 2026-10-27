"""Unit test suite verifying vector algebra and numerical integrator convergence."""

import math
import numpy as np
import pytest

from core.integrator import (
    IDX_OMEGA,
    IDX_THETA,
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    integrate_trajectory,
    make_state,
    rk4_step,
    velocity_verlet_step,
)
from core.vector2d import Vector2D


# ============================================================================
# Vector2D Tests
# ============================================================================

def test_vector2d_initialization_and_immutability():
    """Verify Vector2D enforces 64-bit precision and immutability."""
    v = Vector2D(1, 2)
    assert isinstance(v.x, float)
    assert isinstance(v.y, float)
    assert v.x == 1.0
    assert v.y == 2.0

    # Immutability check (frozen dataclass)
    with pytest.raises(AttributeError):
        v.x = 10.0  # type: ignore


def test_vector2d_arithmetic():
    """Verify vector addition, subtraction, multiplication, division, and negation."""
    v1 = Vector2D(3.0, -4.0)
    v2 = Vector2D(1.0, 2.0)

    # Addition & Subtraction
    v_add = v1 + v2
    assert v_add == Vector2D(4.0, -2.0)

    v_sub = v1 - v2
    assert v_sub == Vector2D(2.0, -6.0)

    # Scalar multiplication & division
    assert v1 * 2.5 == Vector2D(7.5, -10.0)
    assert 2.0 * v2 == Vector2D(2.0, 4.0)
    assert v1 / 2.0 == Vector2D(1.5, -2.0)
    assert -v1 == Vector2D(-3.0, 4.0)
    assert +v1 == v1

    with pytest.raises(ZeroDivisionError):
        _ = v1 / 0.0


def test_vector2d_dot_and_cross_products():
    """Verify dot and 2D cross product operations."""
    i_hat = Vector2D(1.0, 0.0)
    j_hat = Vector2D(0.0, 1.0)

    # Dot product of orthogonal basis vectors is zero
    assert i_hat.dot(j_hat) == 0.0
    assert (i_hat @ j_hat) == 0.0

    # Parallel vectors
    v1 = Vector2D(2.0, 3.0)
    v2 = Vector2D(4.0, -1.0)
    # 2*4 + 3*(-1) = 8 - 3 = 5
    assert v1.dot(v2) == 5.0
    assert (v1 @ v2) == 5.0

    # Cross product (2D determinant)
    # i_hat x j_hat = +1
    assert i_hat.cross(j_hat) == 1.0
    # j_hat x i_hat = -1
    assert j_hat.cross(i_hat) == -1.0
    # Collinear cross product is 0
    assert v1.cross(v1 * 3.0) == 0.0


def test_vector2d_magnitude_and_normalization():
    """Verify magnitude, squared magnitude, and unit vector generation."""
    v = Vector2D(3.0, 4.0)
    assert v.magnitude_squared() == 25.0
    assert v.magnitude() == 5.0

    u = v.normalized()
    assert math.isclose(u.magnitude(), 1.0, abs_tol=1e-15)
    assert math.isclose(u.x, 0.6, abs_tol=1e-15)
    assert math.isclose(u.y, 0.8, abs_tol=1e-15)

    zero_v = Vector2D(0.0, 0.0)
    with pytest.raises(ZeroDivisionError):
        zero_v.normalized()


def test_vector2d_angle_and_rotation():
    """Verify angle extraction and counter-clockwise rotation."""
    v_east = Vector2D(5.0, 0.0)
    v_north = Vector2D(0.0, 5.0)
    v_west = Vector2D(-5.0, 0.0)
    v_south = Vector2D(0.0, -5.0)

    assert math.isclose(v_east.angle(), 0.0, abs_tol=1e-15)
    assert math.isclose(v_north.angle(), math.pi / 2.0, abs_tol=1e-15)
    assert math.isclose(abs(v_west.angle()), math.pi, abs_tol=1e-15)
    assert math.isclose(v_south.angle(), -math.pi / 2.0, abs_tol=1e-15)

    # Rotate east vector by 90 deg counter-clockwise -> should equal north vector
    v_rot90 = v_east.rotate(math.pi / 2.0)
    assert v_rot90.is_close(v_north)

    # Rotate 180 deg
    v_rot180 = v_east.rotate(math.pi)
    assert v_rot180.is_close(v_west)

    # Perpendicular helper
    assert v_east.perpendicular().is_close(v_north)


def test_vector2d_distance_and_conversions():
    """Verify distance computations and array/tuple conversions."""
    p1 = Vector2D(1.0, 1.0)
    p2 = Vector2D(4.0, 5.0)

    assert p1.distance_squared_to(p2) == 25.0
    assert p1.distance_to(p2) == 5.0

    # Conversions
    np_arr = p1.to_numpy()
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.dtype == np.float64
    assert np.allclose(np_arr, [1.0, 1.0])

    p1_from_np = Vector2D.from_numpy(np_arr)
    assert p1 == p1_from_np

    tup = p1.to_tuple()
    assert tup == (1.0, 1.0)
    assert Vector2D.from_tuple(tup) == p1

    # Sequence unpacking
    x, y = p1
    assert x == 1.0 and y == 1.0
    assert p1[0] == 1.0 and p1[1] == 1.0


# ============================================================================
# Integrator Benchmark Tests
# ============================================================================

def test_rk4_ballistic_1d_trajectory():
    """Verify RK4 accuracy against an analytical 1D ballistic flight path under uniform gravity.

    Analytical Equations:
        y(t)  = y0 + v0*t - 0.5*g*t^2
        vy(t) = v0 - g*t

    Because y(t) is a 2nd-degree polynomial and RK4 has local truncation error O(dt^5),
    RK4 computes the exact solution up to floating-point roundoff error (< 1e-11).
    """
    g = 9.80665  # m/s^2
    y0 = 0.0     # m
    v0 = 100.0   # m/s
    t_end = 10.0 # s
    dt = 0.5     # s

    state0 = make_state(x=0.0, y=y0, vx=0.0, vy=v0, theta=0.0, omega=0.0)

    def ballistic_derivatives(_t: float, s: np.ndarray) -> np.ndarray:
        return np.array([0.0, s[IDX_VY], 0.0, -g, 0.0, 0.0], dtype=np.float64)

    times, states = integrate_trajectory(rk4_step, ballistic_derivatives, state0, (0.0, t_end), dt)

    y_final_computed = states[-1, IDX_Y]
    vy_final_computed = states[-1, IDX_VY]

    y_final_analytical = y0 + v0 * t_end - 0.5 * g * (t_end**2)
    vy_final_analytical = v0 - g * t_end

    assert math.isclose(y_final_computed, y_final_analytical, rel_tol=1e-11, abs_tol=1e-11)
    assert math.isclose(vy_final_computed, vy_final_analytical, rel_tol=1e-11, abs_tol=1e-11)


def test_rk4_simple_harmonic_oscillator():
    """Benchmark RK4 against an analytical Simple Harmonic Oscillator (SHO).

    Equation of Motion:
        d^2(x)/dt^2 = -omega_0^2 * x
    Analytical:
        x(t)  = x0 * cos(omega_0 * t) + (v0/omega_0) * sin(omega_0 * t)
        vx(t) = -x0 * omega_0 * sin(omega_0 * t) + v0 * cos(omega_0 * t)
    """
    omega_0 = 2.0  # rad/s
    x0 = 1.0       # m
    v0 = 0.0       # m/s
    t_end = 5.0    # s
    dt = 0.01      # s

    state0 = make_state(x=x0, y=0.0, vx=v0, vy=0.0, theta=0.0, omega=0.0)

    def sho_derivatives(_t: float, s: np.ndarray) -> np.ndarray:
        return np.array([s[IDX_VX], 0.0, -(omega_0**2) * s[IDX_X], 0.0, 0.0, 0.0], dtype=np.float64)

    times, states = integrate_trajectory(rk4_step, sho_derivatives, state0, (0.0, t_end), dt)

    x_computed = states[-1, IDX_X]
    vx_computed = states[-1, IDX_VX]

    x_exact = x0 * math.cos(omega_0 * t_end) + (v0 / omega_0) * math.sin(omega_0 * t_end)
    vx_exact = -x0 * omega_0 * math.sin(omega_0 * t_end) + v0 * math.cos(omega_0 * t_end)

    assert math.isclose(x_computed, x_exact, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(vx_computed, vx_exact, rel_tol=1e-6, abs_tol=1e-6)


def test_rk4_fourth_order_global_convergence():
    """Verify that RK4 exhibits 4th-order global error convergence: O(dt^4).

    When halving dt, the global state-space norm error must decrease by
    a factor of approximately 2^4 = 16 (empirical order p ~ 4.0).
    """
    omega_0 = 1.0  # rad/s
    x0 = 1.0
    v0 = 0.0
    t_end = 5.0    # s

    def sho_derivatives(_t: float, s: np.ndarray) -> np.ndarray:
        return np.array([s[IDX_VX], 0.0, -(omega_0**2) * s[IDX_X], 0.0, 0.0, 0.0], dtype=np.float64)

    state0 = make_state(x=x0, y=0.0, vx=v0, vy=0.0, theta=0.0, omega=0.0)

    # Successively halved step sizes within the asymptotic convergence regime
    dt_coarse = 0.1
    dt_medium = 0.05
    dt_fine = 0.025

    _, states_coarse = integrate_trajectory(rk4_step, sho_derivatives, state0, (0.0, t_end), dt_coarse)
    _, states_medium = integrate_trajectory(rk4_step, sho_derivatives, state0, (0.0, t_end), dt_medium)
    _, states_fine = integrate_trajectory(rk4_step, sho_derivatives, state0, (0.0, t_end), dt_fine)

    # Exact analytical state at t_end
    x_exact = x0 * math.cos(omega_0 * t_end)
    vx_exact = -x0 * omega_0 * math.sin(omega_0 * t_end)

    # Compute Euclidean state-space error norm: ||S_computed - S_exact||
    err_coarse = math.hypot(states_coarse[-1, IDX_X] - x_exact, states_coarse[-1, IDX_VX] - vx_exact)
    err_medium = math.hypot(states_medium[-1, IDX_X] - x_exact, states_medium[-1, IDX_VX] - vx_exact)
    err_fine = math.hypot(states_fine[-1, IDX_X] - x_exact, states_fine[-1, IDX_VX] - vx_exact)

    # Error reduction ratios (theoretical value = 2^4 = 16.0)
    ratio_1 = err_coarse / err_medium
    ratio_2 = err_medium / err_fine

    # Empirical order of convergence: p = log2(error_ratio)
    order_1 = math.log2(ratio_1)
    order_2 = math.log2(ratio_2)

    # Verify ~16x error reduction (p ~ 4.0 within +/- 0.1)
    assert 15.0 < ratio_1 < 17.0, f"Expected ratio ~16, got {ratio_1:.3f}"
    assert 15.0 < ratio_2 < 17.0, f"Expected ratio ~16, got {ratio_2:.3f}"
    assert math.isclose(order_1, 4.0, abs_tol=0.1)
    assert math.isclose(order_2, 4.0, abs_tol=0.1)


def test_velocity_verlet_harmonic_oscillator_and_energy_conservation():
    """Verify Velocity Verlet symplectic properties: 2nd-order convergence and bounded energy."""
    omega_0 = 1.0  # rad/s
    x0 = 1.0
    v0 = 0.0
    t_end = 20.0   # s (~3.18 periods)

    def sho_derivatives(_t: float, s: np.ndarray) -> np.ndarray:
        return np.array([s[IDX_VX], 0.0, -(omega_0**2) * s[IDX_X], 0.0, 0.0, 0.0], dtype=np.float64)

    state0 = make_state(x=x0, y=0.0, vx=v0, vy=0.0, theta=0.0, omega=0.0)

    # Test 2nd-order convergence: halving dt reduces error by approx 2^2 = 4
    dt_1 = 0.2
    dt_2 = 0.1
    dt_3 = 0.05

    x_exact = x0 * math.cos(omega_0 * t_end)

    _, states_1 = integrate_trajectory(velocity_verlet_step, sho_derivatives, state0, (0.0, t_end), dt_1)
    _, states_2 = integrate_trajectory(velocity_verlet_step, sho_derivatives, state0, (0.0, t_end), dt_2)
    _, states_3 = integrate_trajectory(velocity_verlet_step, sho_derivatives, state0, (0.0, t_end), dt_3)

    err_1 = abs(states_1[-1, IDX_X] - x_exact)
    err_2 = abs(states_2[-1, IDX_X] - x_exact)
    err_3 = abs(states_3[-1, IDX_X] - x_exact)

    ratio_1 = err_1 / err_2
    ratio_2 = err_2 / err_3

    assert 3.7 < ratio_1 < 4.3, f"Expected Verlet ratio ~4, got {ratio_1:.3f}"
    assert 3.7 < ratio_2 < 4.3, f"Expected Verlet ratio ~4, got {ratio_2:.3f}"

    # Verify Bounded Energy Conservation (Symplectic property)
    # Total mechanical energy per unit mass: E = 0.5 * v^2 + 0.5 * omega_0^2 * x^2
    times, states_long = integrate_trajectory(
        velocity_verlet_step, sho_derivatives, state0, (0.0, 50.0), dt=0.05
    )
    e0 = 0.5 * (v0**2) + 0.5 * (omega_0**2) * (x0**2)

    energies = 0.5 * (states_long[:, IDX_VX] ** 2) + 0.5 * (omega_0**2) * (states_long[:, IDX_X] ** 2)
    relative_energy_deviations = np.abs(energies - e0) / e0

    # Max energy oscillation is strictly bounded without secular drift
    assert np.max(relative_energy_deviations) < 1e-3
    assert math.isclose(energies[-1], e0, rel_tol=1e-3)


def test_rotational_dynamics():
    """Verify rotational degree of freedom integration: constant angular acceleration."""
    alpha_const = 0.5  # rad/s^2
    theta0 = 0.0       # rad
    omega0 = 0.1       # rad/s
    t_end = 4.0        # s
    dt = 0.1           # s

    state0 = make_state(x=0.0, y=0.0, vx=0.0, vy=0.0, theta=theta0, omega=omega0)

    def rotational_derivatives(_t: float, s: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, s[IDX_OMEGA], alpha_const], dtype=np.float64)

    times, states = integrate_trajectory(rk4_step, rotational_derivatives, state0, (0.0, t_end), dt)

    theta_exact = theta0 + omega0 * t_end + 0.5 * alpha_const * (t_end**2)
    omega_exact = omega0 + alpha_const * t_end

    assert math.isclose(states[-1, IDX_THETA], theta_exact, rel_tol=1e-12)
    assert math.isclose(states[-1, IDX_OMEGA], omega_exact, rel_tol=1e-12)