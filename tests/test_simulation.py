"""Comprehensive unit and integration test suite for dual-mode Time Warp, Keplerian propagation, and Simulation orchestrator.

This module verifies:
    1. Kepler Equation Solver: Precision convergence of E - e * sin(E) = M for e in [0.0, 0.95] (residual < 10^-11).
    2. On-Rails Closed Orbit Invariance: 50 full revolutions in circular (300 km) and eccentric orbits with zero drift (< 10^-8).
    3. Hyperbolic Flyby Propagation: Escape trajectory propagation with energy & angular momentum conservation (< 10^-8).
    4. Physics Warp Sub-Stepping Stability: Numerical consistency and stability of 4x Physics Warp vs 1x standard integration.
    5. Auto-Drop Time Warp on Reentry: Automated drop from 100x On-Rails to 1x Physics Warp upon crossing Karman line.
    6. Telemetry Logger & Mission Summary: Max-Q, Max speed, peak altitude, Delta-V expended, dictionary and CSV export.
    7. Safety Rules & Clamps: Atmospheric, active thrust, and flight state constraints preventing invalid On-Rails activation.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.vector2d import Vector2D
from models.presets import create_falcon_9
from physics.celestial import get_earth
from physics.orbit import (
    calculate_orbit,
    propagate_kepler,
    solve_kepler_elliptic,
    solve_kepler_hyperbolic,
)
from simulation.flight_dynamics import FlightVehicle, TelemetrySnapshot, VehicleState
from simulation.simulation import Simulation, WarpMode
from simulation.telemetry import TelemetryLogger


# ============================================================================
# 1. Kepler Equation Solver Tests
# ============================================================================


def test_kepler_equation_solver_elliptic_precision():
    """Verify Kepler equation solver across eccentricities from 0.0 to 0.95.

    Requirements:
        - Solve f(E) = E - e * sin(E) - M = 0 via Newton-Raphson.
        - Residual |E - e * sin(E) - M| < 10^-11 across all tested angles and eccentricities.
    """
    eccentricities = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95]
    mean_anomalies = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)

    for e in eccentricities:
        for M in mean_anomalies:
            E = solve_kepler_elliptic(float(M), e, tol=1e-12)
            M_norm = float(M) % (2.0 * math.pi)
            residual = abs(E - e * math.sin(E) - M_norm)
            assert residual < 1e-11, (
                f"Kepler elliptic residual {residual:.2e} exceeded 1e-11 at e={e}, M={M}"
            )


def test_kepler_equation_solver_hyperbolic():
    """Verify hyperbolic Kepler solver: e * sinh(H) - H - M_h = 0 for e > 1.0."""
    eccentricities = [1.05, 1.2, 1.5, 2.0, 5.0]
    mean_anomalies = [-10.0, -2.5, -0.5, 0.0, 0.5, 2.5, 10.0]

    for e in eccentricities:
        for M_h in mean_anomalies:
            H = solve_kepler_hyperbolic(M_h, e, tol=1e-12)
            residual = abs(e * math.sinh(H) - H - M_h)
            assert residual < 1e-11, (
                f"Kepler hyperbolic residual {residual:.2e} exceeded 1e-11 at e={e}, M_h={M_h}"
            )


# ============================================================================
# 2. On-Rails Closed Orbit Invariance Tests
# ============================================================================


def test_on_rails_closed_orbit_invariance_circular():
    """Verify 50 complete orbits in one call for a 300 km circular orbit.

    Requirements:
        - a, e, energy, and h remain unchanged to within 10^-8 relative error.
        - Final position and velocity match initial state to within 10^-8 relative error.
    """
    earth = get_earth()
    alt = 300_000.0  # 300 km
    r_circ = earth.radius + alt
    v_circ = math.sqrt(earth.mu / r_circ)

    pos0 = Vector2D(r_circ, 0.0)
    vel0 = Vector2D(0.0, v_circ)

    orb0 = calculate_orbit(pos0, vel0, earth)
    assert orb0.orbital_period is not None
    assert orb0.is_circular

    # Fast-forward 50 complete revolutions in a single analytical call
    dt_50_orbits = 50.0 * orb0.orbital_period
    pos_50, vel_50 = propagate_kepler(pos0, vel0, earth, dt_50_orbits)
    orb_50 = calculate_orbit(pos_50, vel_50, earth)

    # Invariant checks
    rel_err_a = abs(orb_50.semi_major_axis - orb0.semi_major_axis) / orb0.semi_major_axis
    err_e = abs(orb_50.eccentricity - orb0.eccentricity)
    rel_err_energy = abs(orb_50.specific_orbital_energy - orb0.specific_orbital_energy) / abs(
        orb0.specific_orbital_energy
    )
    rel_err_h = abs(orb_50.specific_angular_momentum - orb0.specific_angular_momentum) / abs(
        orb0.specific_angular_momentum
    )

    assert rel_err_a < 1e-8, f"Semi-major axis drift {rel_err_a:.2e} exceeds 1e-8"
    assert err_e < 1e-8, f"Eccentricity drift {err_e:.2e} exceeds 1e-8"
    assert rel_err_energy < 1e-8, f"Energy drift {rel_err_energy:.2e} exceeds 1e-8"
    assert rel_err_h < 1e-8, f"Angular momentum drift {rel_err_h:.2e} exceeds 1e-8"

    # State vector match
    pos_rel_diff = (pos_50 - pos0).magnitude() / r_circ
    vel_rel_diff = (vel_50 - vel0).magnitude() / v_circ
    assert pos_rel_diff < 1e-8
    assert vel_rel_diff < 1e-8


def test_on_rails_closed_orbit_invariance_elliptic():
    """Verify 50 complete orbits in one call for a 300 km x 1000 km elliptical orbit.

    Requirements:
        - a, e, energy, and h remain unchanged to within 10^-8 relative error.
        - Final state matches initial state to within 10^-8.
    """
    earth = get_earth()
    rp = earth.radius + 300_000.0
    ra = earth.radius + 1_000_000.0
    a = 0.5 * (rp + ra)
    vp = math.sqrt(earth.mu * (2.0 / rp - 1.0 / a))

    pos0 = Vector2D(rp, 0.0)
    vel0 = Vector2D(0.0, vp)

    orb0 = calculate_orbit(pos0, vel0, earth)
    assert orb0.orbital_period is not None

    dt_50_orbits = 50.0 * orb0.orbital_period
    pos_50, vel_50 = propagate_kepler(pos0, vel0, earth, dt_50_orbits)
    orb_50 = calculate_orbit(pos_50, vel_50, earth)

    rel_err_a = abs(orb_50.semi_major_axis - orb0.semi_major_axis) / orb0.semi_major_axis
    rel_err_e = abs(orb_50.eccentricity - orb0.eccentricity) / orb0.eccentricity
    rel_err_energy = abs(orb_50.specific_orbital_energy - orb0.specific_orbital_energy) / abs(
        orb0.specific_orbital_energy
    )
    rel_err_h = abs(orb_50.specific_angular_momentum - orb0.specific_angular_momentum) / abs(
        orb0.specific_angular_momentum
    )

    assert rel_err_a < 1e-8
    assert rel_err_e < 1e-8
    assert rel_err_energy < 1e-8
    assert rel_err_h < 1e-8

    pos_rel_diff = (pos_50 - pos0).magnitude() / rp
    vel_rel_diff = (vel_50 - vel0).magnitude() / vp
    assert pos_rel_diff < 1e-8
    assert vel_rel_diff < 1e-8


# ============================================================================
# 3. Hyperbolic Flyby Propagation Test
# ============================================================================


def test_hyperbolic_flyby_propagation():
    """Verify hyperbolic escape trajectory propagation and energy conservation.

    Requirements:
        - Propagate escape trajectory (v > v_esc) analytically forward by 3600 s.
        - Position moves outward (r_final > r_initial).
        - Specific energy and angular momentum conserved to within 10^-8.
        - Vis-viva relation v^2 = mu * (2/r - 1/a) holds exactly at propagated state.
    """
    earth = get_earth()
    r0 = earth.radius + 500_000.0  # 500 km
    v_esc = math.sqrt(2.0 * earth.mu / r0)
    v0 = 1.30 * v_esc  # 30% above escape velocity

    pos0 = Vector2D(r0, 0.0)
    vel0 = Vector2D(0.0, v0)

    orb0 = calculate_orbit(pos0, vel0, earth)
    assert orb0.is_hyperbolic
    assert orb0.eccentricity > 1.0

    # Propagate 1 hour forward
    dt = 3600.0
    pos_prop, vel_prop = propagate_kepler(pos0, vel0, earth, dt)
    orb_prop = calculate_orbit(pos_prop, vel_prop, earth)

    # Spacecraft has receded away from Earth
    assert pos_prop.magnitude() > pos0.magnitude()

    # Energy and momentum invariance
    energy_rel_diff = abs(orb_prop.specific_orbital_energy - orb0.specific_orbital_energy) / abs(
        orb0.specific_orbital_energy
    )
    h_rel_diff = abs(orb_prop.specific_angular_momentum - orb0.specific_angular_momentum) / abs(
        orb0.specific_angular_momentum
    )
    a_rel_diff = abs(orb_prop.semi_major_axis - orb0.semi_major_axis) / abs(orb0.semi_major_axis)
    e_rel_diff = abs(orb_prop.eccentricity - orb0.eccentricity) / orb0.eccentricity

    assert energy_rel_diff < 1e-8
    assert h_rel_diff < 1e-8
    assert a_rel_diff < 1e-8
    assert e_rel_diff < 1e-8

    # Vis-viva invariant check
    expected_v_sq = earth.mu * (2.0 / pos_prop.magnitude() - 1.0 / orb_prop.semi_major_axis)
    actual_v_sq = vel_prop.magnitude_squared()
    assert math.isclose(actual_v_sq, expected_v_sq, rel_tol=1e-8)


# ============================================================================
# 4. Physics Warp Sub-Stepping Stability Test
# ============================================================================


def test_physics_warp_substepping_stability():
    """Verify numerical consistency of 4x Physics Warp vs 1x standard integration.

    Requirements:
        - Run atmospheric ascent under 4x physics warp and 1x standard warp.
        - Sub-stepping (dt_sub <= 0.05 s) ensures numerical integration remains
          stable and matches 1x within < 0.01% tolerance.
    """
    earth = get_earth()

    # Setup Simulation 1: 1x warp
    f9_1 = create_falcon_9()
    v1 = FlightVehicle(f9_1, earth, Vector2D(0.0, earth.radius))
    v1.set_throttle(1.0)
    sim_1x = Simulation(v1, earth, time_warp=1.0, max_physics_substep=0.05)

    # Setup Simulation 2: 4x warp
    f9_2 = create_falcon_9()
    v2 = FlightVehicle(f9_2, earth, Vector2D(0.0, earth.radius))
    v2.set_throttle(1.0)
    sim_4x = Simulation(v2, earth, time_warp=4.0, max_physics_substep=0.05)

    # Run for 10.0 seconds of simulation time:
    # 1x warp: 200 steps at wall_dt = 0.05 s (dt_sim = 0.05 s)
    for _ in range(200):
        sim_1x.step(wall_dt=0.05)

    # 4x warp: 50 steps at wall_dt = 0.05 s (dt_sim = 0.20 s, 4 sub-steps of 0.05 s)
    for _ in range(50):
        sim_4x.step(wall_dt=0.05)

    assert math.isclose(sim_1x.elapsed_time, 10.0, rel_tol=1e-7)
    assert math.isclose(sim_4x.elapsed_time, 10.0, rel_tol=1e-7)

    # Both simulations executed identical sub-steps; state must match strictly
    assert math.isclose(sim_1x.vehicle.altitude, sim_4x.vehicle.altitude, rel_tol=1e-6)
    assert math.isclose(sim_1x.vehicle.speed, sim_4x.vehicle.speed, rel_tol=1e-6)
    assert math.isclose(
        sim_1x.vehicle.rocket.total_mass(), sim_4x.vehicle.rocket.total_mass(), rel_tol=1e-6
    )


# ============================================================================
# 5. Auto-Drop Time Warp on Reentry Test
# ============================================================================


def test_auto_drop_time_warp_on_reentry():
    """Verify simulation auto-drops to 1x Physics Warp upon crossing atmospheric threshold.

    Requirements:
        - Place vehicle on an orbit with periapsis inside atmosphere (hp = 50 km)
          and apoapsis in space (ha = 200 km).
        - Engage 100x On-Rails warp.
        - Advance simulation until descending across the 100 km Karman line.
        - Verify warp mode automatically drops to WarpMode.PHYSICS with time_warp = 1.0.
    """
    earth = get_earth()
    ha = 200_000.0  # 200 km (in space)
    hp = 50_000.0   # 50 km (inside atmosphere)
    ra = earth.radius + ha
    rp = earth.radius + hp
    a = 0.5 * (ra + rp)
    va = math.sqrt(earth.mu * (2.0 / ra - 1.0 / a))

    # Spacecraft placed at apoapsis heading toward periapsis
    pos = Vector2D(-ra, 0.0)
    vel = Vector2D(0.0, -va)

    f9 = create_falcon_9()
    vehicle = FlightVehicle(
        rocket=f9,
        body=earth,
        position=pos,
        velocity=vel,
        attitude=math.atan2(vel.y, vel.x),
        state=VehicleState.IN_FLIGHT,
    )

    sim = Simulation(vehicle, earth)
    sim.set_time_warp(100.0)
    assert sim.time_warp == 100.0
    assert sim.warp_mode == WarpMode.ON_RAILS

    # Advance until warp drops
    max_steps = 100
    steps = 0
    while sim.warp_mode == WarpMode.ON_RAILS and steps < max_steps:
        sim.step(wall_dt=1.0)
        steps += 1

    # Verify auto-drop safety triggered
    assert sim.warp_mode == WarpMode.PHYSICS
    assert sim.time_warp == 1.0
    assert sim.vehicle.altitude <= (earth.atmosphere.cutoff_altitude + 1e-4)


# ============================================================================
# 6. Telemetry Logger, Mission Summary & Export Tests
# ============================================================================


def test_telemetry_summary_and_export():
    """Verify TelemetryLogger metrics summary extraction and CSV export.

    Requirements:
        - Accurately report Max-Q (dynamic pressure), Max speed, peak altitude,
          and Delta-V expended.
        - to_dict() returns columnar lists of floats matching summary metrics.
        - to_csv() produces a valid CSV file with expected headers and row count.
    """
    earth = get_earth()
    f9 = create_falcon_9()
    vehicle = FlightVehicle(f9, earth, Vector2D(0.0, earth.radius))
    vehicle.set_throttle(1.0)

    sim = Simulation(vehicle, earth)

    # Run ascent flight for 60 seconds
    for _ in range(60):
        sim.step(wall_dt=1.0)

    summary = sim.logger.get_summary()

    # Core performance metric validations
    assert summary["max_q"] > 1000.0, f"Max-Q too low: {summary['max_q']}"
    assert summary["max_q_time"] > 0.0
    assert summary["max_speed"] > 100.0
    assert summary["max_altitude"] > 5000.0
    assert summary["delta_v_expended"] > 500.0
    assert summary["max_g_force"] > 1.0
    assert summary["final_vehicle_state"] == VehicleState.IN_FLIGHT

    # Verify dictionary export
    t_dict = sim.logger.to_dict()
    assert len(t_dict["time"]) == 60
    assert max(t_dict["dynamic_pressure"]) == summary["max_q"]
    assert max(t_dict["speed"]) == summary["max_speed"]
    assert max(t_dict["altitude"]) == summary["max_altitude"]

    # Verify CSV export
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        csv_path = Path(tf.name)
        sim.logger.to_csv(csv_path)

        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()
            # 1 header line + 60 recorded snapshot rows
            assert len(lines) == 61
            assert "dynamic_pressure" in lines[0]
            assert "altitude" in lines[0]


# ============================================================================
# 7. Time Warp Safety Rules & Clamping Tests
# ============================================================================


def test_time_warp_safety_clamps():
    """Verify safety rules preventing illegal On-Rails warp activation.

    Rules tested:
        1. In atmosphere (altitude < cutoff): clamped to 4.0x Physics Warp.
        2. Engines actively thrusting (T > 0): clamped to 4.0x Physics Warp.
        3. Vehicle not IN_FLIGHT (e.g. ON_PAD): clamped to 4.0x Physics Warp.
        4. Throttling engines while in ON_RAILS drops warp to 1.0x Physics Warp.
    """
    earth = get_earth()
    f9 = create_falcon_9()

    # 1. On pad: cannot engage On-Rails warp
    v_pad = FlightVehicle(f9, earth, Vector2D(0.0, earth.radius), state=VehicleState.ON_PAD)
    sim_pad = Simulation(v_pad, earth)
    sim_pad.set_time_warp(100.0)
    assert sim_pad.warp_mode == WarpMode.PHYSICS
    assert sim_pad.time_warp == 4.0

    # 2. In atmosphere (50 km altitude): cannot engage On-Rails warp
    f9_2 = create_falcon_9()
    v_atm = FlightVehicle(
        f9_2,
        earth,
        Vector2D(0.0, earth.radius + 50_000.0),
        state=VehicleState.IN_FLIGHT,
    )
    sim_atm = Simulation(v_atm, earth)
    sim_atm.set_time_warp(50.0)
    assert sim_atm.warp_mode == WarpMode.PHYSICS
    assert sim_atm.time_warp == 4.0

    # 3. Outside atmosphere (200 km) but engines firing: cannot engage On-Rails warp
    f9_3 = create_falcon_9()
    v_burn = FlightVehicle(
        f9_3,
        earth,
        Vector2D(0.0, earth.radius + 200_000.0),
        state=VehicleState.IN_FLIGHT,
    )
    v_burn.ignite()
    v_burn.set_throttle(1.0)
    sim_burn = Simulation(v_burn, earth)
    sim_burn.set_time_warp(100.0)
    assert sim_burn.warp_mode == WarpMode.PHYSICS
    assert sim_burn.time_warp == 4.0

    # 4. Outside atmosphere, coasting (T=0): allows On-Rails warp
    v_burn.set_throttle(0.0)
    sim_burn.set_time_warp(100.0)
    assert sim_burn.warp_mode == WarpMode.ON_RAILS
    assert sim_burn.time_warp == 100.0

    # 5. Commanding throttle while on rails automatically drops to 1.0x Physics
    sim_burn.step(wall_dt=0.1, throttle=1.0)
    assert sim_burn.warp_mode == WarpMode.PHYSICS
    assert sim_burn.time_warp == 1.0
