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


def test_hyperbolic_reentry_interception_on_rails():
    """Verify hyperbolic trajectories do not bypass entry interception in on-rails warp."""
    earth = get_earth()
    f9 = create_falcon_9()
    cutoff = 100_000.0  # 100 km Karman line
    r_atm = earth.radius + cutoff

    # Place vehicle at 300 km altitude inbound on a hyperbolic trajectory with periapsis at 50 km (inside atmosphere)
    r_init = earth.radius + 300_000.0
    rp_target = earth.radius + 50_000.0
    e_hyp = 1.2
    # Semi-major axis: rp = a * (e - 1) => a = rp / (e - 1)
    a_h = rp_target / (e_hyp - 1.0)
    # v^2 = mu * (2/r + 1/a_h)
    v_mag = math.sqrt(earth.mu * (2.0 / r_init + 1.0 / a_h))
    # Inward flight path angle
    # r = a_h * (e * cosh(H) - 1)
    cosh_h = (1.0 + r_init / a_h) / e_hyp
    h_anom = -math.acosh(cosh_h)
    # r_dot_v = sqrt(mu * a_h) * e * sinh(H)
    r_dot_v = math.sqrt(earth.mu * a_h) * e_hyp * math.sinh(h_anom)
    # Inbound radial velocity component
    v_r = r_dot_v / r_init
    v_theta = -math.sqrt(max(0.0, v_mag**2 - v_r**2))

    pos = Vector2D(r_init, 0.0)
    vel = Vector2D(v_r, v_theta)

    vehicle = FlightVehicle(f9, earth, pos, vel, state=VehicleState.IN_FLIGHT)
    sim = Simulation(vehicle, earth)
    sim.set_time_warp(100.0)
    assert sim.warp_mode == WarpMode.ON_RAILS

    # Step forward with large macro-step: should drop to PHYSICS before tunneling into planet
    sim.step(wall_dt=10.0)
    assert sim.warp_mode == WarpMode.PHYSICS
    assert sim.time_warp == 1.0
    # Altitude should be clamped at entry boundary, not deeply negative
    assert vehicle.altitude >= -100.0


def test_airless_body_impact_detection_on_rails():
    """Verify that on airless bodies, on-rails warp evaluates crash conditions upon impact."""
    from physics.celestial import get_moon
    moon = get_moon()
    f9 = create_falcon_9()

    # Place vehicle at 50 km altitude falling vertically at 1000 m/s onto Moon
    pos = Vector2D(0.0, moon.radius + 50_000.0)
    vel = Vector2D(0.0, -1000.0)

    vehicle = FlightVehicle(f9, moon, pos, vel, state=VehicleState.IN_FLIGHT)
    sim = Simulation(vehicle, moon)
    sim.set_time_warp(100.0)
    assert sim.warp_mode == WarpMode.ON_RAILS

    # Step forward 1.0s * 100 = 100 s: should impact surface, drop warp, and transition to DESTROYED
    sim.step(wall_dt=1.0)
    assert sim.warp_mode == WarpMode.PHYSICS
    assert sim.vehicle.state == VehicleState.DESTROYED


def test_substep_burnout_delta_v_accuracy():
    """Verify that delta_v_expended does not overestimate burnout impulse during sub-stepped macro-steps."""
    from models.parts import Engine, FuelTank
    from models.rocket import Rocket, Stage
    from core.constants import g0
    earth = get_earth()

    # Engine: 100 kN thrust, 300 s Isp -> ~33.99 kg/s
    eng = Engine(
        id="eng",
        name="Engine",
        dry_mass=100.0,
        length=1.0,
        radius=0.5,
        thrust_vac=100_000.0,
        thrust_sl=100_000.0,
        isp_vac=300.0,
        isp_sl=300.0,
        is_ignited=True,
        throttle=1.0,
    )
    # Fuel for exactly 0.05 seconds = ~1.7 kg
    tank = FuelTank(
        id="tank",
        name="Tank",
        dry_mass=100.0,
        length=2.0,
        radius=0.5,
        propellant_mass=1.7,
        propellant_capacity=100.0,
    )
    stage = Stage(stage_id="s1", name="Stage 1", parts=[eng, tank])
    rocket = Rocket(stages=[stage])

    # Deep space vehicle
    pos = Vector2D(0.0, 1e11)
    vehicle = FlightVehicle(rocket, earth, pos, Vector2D(0.0, 0.0), state=VehicleState.IN_FLIGHT)
    sim = Simulation(vehicle, earth, max_physics_substep=0.05)

    # Macro step of wall_dt = 1.0 s -> 20 sub-steps of 0.05 s
    # Fuel burns out on sub-step 1!
    sim.step(wall_dt=1.0)

    summary = sim.logger.get_summary()
    dv_expended = summary["delta_v_expended"]

    # Total theoretical delta_v is for ~1.7 kg burned from ~201.7 kg:
    # dv ~= 300 * 9.80665 * ln(201.7 / 200.0) ~= 24.9 m/s
    # The old bug integrated 0.5 * 100 kN across the full 1.0 s macro-step (~250 m/s, 10x too large!)
    assert dv_expended < 40.0, f"Delta-v overestimated: {dv_expended:.1f} m/s"
    assert dv_expended > 15.0


def test_telemetry_max_g_force_includes_aerodynamics():
    """Verify TelemetryLogger.get_summary()['max_g_force'] includes aerodynamic deceleration when thrust is zero."""
    from models.parts import Part
    from models.rocket import Rocket, Stage

    earth = get_earth()
    # Unpowered capsule (no engines, pure aerodynamic drag)
    capsule = Part(
        id="capsule",
        name="Reentry Capsule",
        dry_mass=1000.0,
        length=3.0,
        radius=1.5,
    )
    stage = Stage(stage_id="s1", name="Capsule Stage", parts=[capsule])
    rocket = Rocket(stages=[stage])

    # Reentering atmosphere at 15 km altitude with 1500 m/s speed
    pos = Vector2D(0.0, earth.radius + 15000.0)
    vel = Vector2D(1500.0, 0.0)
    vehicle = FlightVehicle(
        rocket=rocket,
        body=earth,
        position=pos,
        velocity=vel,
        attitude=0.0,
        state=VehicleState.IN_FLIGHT,
        cd_axial=0.8,
    )
    sim = Simulation(vehicle, earth)

    # Step simulation
    snap = sim.step(wall_dt=0.1)

    assert snap.thrust == 0.0
    # At 15 km with 1500 m/s, dynamic pressure is substantial:
    # Drag deceleration should produce > 1 g of aerodynamic force
    assert snap.g_force > 1.0, f"Expected aerodynamic g_force > 1.0, got {snap.g_force}"

    summary = sim.logger.get_summary()
    assert summary["max_g_force"] == pytest.approx(snap.g_force, rel=1e-5)
    assert summary["max_g_force"] > 1.0


def test_simulation_run_docstring_and_wall_dt_validation():
    """Verify Simulation.run has a valid docstring and rejects wall_dt <= 0 with ValueError."""
    from models.presets import create_falcon_9
    earth = get_earth()
    f9 = create_falcon_9()
    vehicle = FlightVehicle(f9, earth, Vector2D(0.0, earth.radius), state=VehicleState.ON_PAD)
    sim = Simulation(vehicle, earth)

    # 1. Docstring must be present and valid
    assert Simulation.run.__doc__ is not None
    assert "Run the simulation for a specified simulation duration" in Simulation.run.__doc__

    # 2. wall_dt <= 0 must raise ValueError to prevent infinite loop
    with pytest.raises(ValueError, match="wall_dt must be positive"):
        sim.run(duration=10.0, wall_dt=0.0)

    with pytest.raises(ValueError, match="wall_dt must be positive"):
        sim.run(duration=10.0, wall_dt=-0.1)

    # 3. duration <= 0 should return empty list
    assert sim.run(duration=0.0, wall_dt=0.1) == []
    assert sim.run(duration=-5.0, wall_dt=0.1) == []




