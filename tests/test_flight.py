"""Comprehensive unit and integration flight test suite for planar 3-DOF rocket dynamics.

This module verifies:
    1. PID Controller: P, I, D actions, symmetric output bounds, integral anti-windup clamping.
    2. Autopilot SAS: Gimbal deflection, rate damping, angular wrapping, Prograde / Retrograde tracking.
    3. Vehicle Kinematics & Derivatives: Gravitational, thrust, aerodynamic force & torque coupling.
    4. Launchpad Static Equilibrium: Surface clamping and planetary co-rotation over 10.0 s.
    5. Vertical Ascent & Liftoff: TWR > 1.3 liftoff threshold, state transition, atmospheric climb.
    6. SAS Attitude Stabilization: Dynamic damping of omega_0 = 0.1 rad/s disturbance back to target pitch.
    7. Ground Impact Detection: Free-fall from 1000 m resulting in DESTROYED state at ground contact.
    8. Safe Touchdown: Gentle descent at low velocity resulting in LANDED state.
    9. Suborbital Flight & Multi-Stage Staging: Ascent, pitch-over profile (85 deg at 2 km, 60 deg at 15 km),
       Stage 1 burnout detection, clean separation, Stage 2 ignition, and space apoapsis > 100 km.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.vector2d import Vector2D
from models.presets import create_falcon_9
from models.rocket import Rocket, Stage
from physics.celestial import get_earth
from simulation.autopilot import AutopilotSAS, PIDController, SASMode
from simulation.flight_dynamics import FlightVehicle, TelemetrySnapshot, VehicleState


# ============================================================================
# 1. PID Controller & SAS Autopilot Tests
# ============================================================================


def test_pid_controller_proportional_and_derivative():
    """Verify PID controller proportional gain and derivative rate tracking."""
    pid = PIDController(kp=2.0, ki=0.0, kd=1.0, limit=10.0)

    # Initial step: error = 3.0, dt = 1.0 (no previous error, D=0)
    out1 = pid.update(error=3.0, dt=1.0)
    assert math.isclose(out1, 6.0, rel_tol=1e-12)

    # Second step: error drops from 3.0 to 1.0, d(error)/dt = -2.0
    # Output: P (2.0 * 1.0) + D (1.0 * -2.0) = 2.0 - 2.0 = 0.0
    out2 = pid.update(error=1.0, dt=1.0)
    assert math.isclose(out2, 0.0, abs_tol=1e-12)


def test_pid_controller_anti_windup_and_clamping():
    """Verify PID controller output clamp and integral anti-windup accumulator bounding."""
    pid = PIDController(kp=1.0, ki=2.0, kd=0.0, limit=5.0, integral_limit=2.5)

    # Accumulate large error over several steps
    for _ in range(10):
        out = pid.update(error=10.0, dt=1.0)
        assert out <= 5.0
        assert out >= -5.0

    # Integral accumulator must be clamped to integral_limit = 2.5
    assert math.isclose(pid.integral, 2.5, rel_tol=1e-12)

    # Reset controller state
    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_error is None


def test_autopilot_sas_gimbal_command_and_modes():
    """Verify AutopilotSAS modes, rate damping, and nozzle gimbal deflection limits."""
    sas = AutopilotSAS(
        kp=2.0,
        kd=1.5,
        delta_max=0.087266,  # 5 degrees
        mode=SASMode.HOLD_ATTITUDE,
        target_attitude=math.pi / 2,
    )

    # Attitude aligned (error=0), disturbance omega = 0.1 rad/s
    # delta_cmd = clamp(2.0 * 0.0 - 1.5 * 0.1, -0.087266, 0.087266) = -0.15 clamped to -0.087266
    cmd = sas.update(theta=math.pi / 2, omega=0.1, dt=0.1)
    assert math.isclose(cmd, -0.087266, rel_tol=1e-9)

    # SAS OFF mode should command 0.0 deflection
    sas.set_mode(SASMode.OFF)
    assert sas.update(theta=0.0, omega=0.5, dt=0.1) == 0.0


def test_autopilot_sas_angle_wrapping_and_prograde_tracking():
    """Verify angle wrapping across +/- pi and orbital prograde orientation mode."""
    sas = AutopilotSAS(
        kp=1.0,
        kd=0.0,
        delta_max=0.5,
        mode=SASMode.HOLD_ATTITUDE,
        target_attitude=3.10,
    )

    # Current attitude is -3.10 rad. Shortest path across +/- pi is +0.083 rad, not -6.20
    cmd = sas.update(theta=-3.10, omega=0.0, dt=0.1)
    expected_error = math.atan2(math.sin(3.10 - (-3.10)), math.cos(3.10 - (-3.10)))
    assert math.isclose(cmd, expected_error, rel_tol=1e-9)

    # Prograde mode: orientation follows velocity vector
    sas.set_mode(SASMode.PROGRADE)
    _ = sas.update(theta=0.0, omega=0.0, dt=0.1, velocity=Vector2D(100.0, 100.0))
    assert math.isclose(sas.target_attitude, math.pi / 4, rel_tol=1e-9)


# ============================================================================
# 2. Kinematics, Coordinate Frames & Derivative Solver Tests
# ============================================================================


def test_flight_vehicle_coordinate_frames_and_vectors():
    """Verify heading and transverse normal unit vector conventions."""
    earth = get_earth()
    f9 = create_falcon_9()
    vehicle = FlightVehicle(
        rocket=f9,
        body=earth,
        position=Vector2D(0.0, earth.radius + 10000.0),
        velocity=Vector2D(100.0, 200.0),
        attitude=math.pi / 2,  # Pointing +Y
        angular_velocity=0.05,
    )

    # Heading u_hat = (cos(pi/2), sin(pi/2)) = (0, 1)
    assert math.isclose(vehicle.heading_vector.x, 0.0, abs_tol=1e-12)
    assert math.isclose(vehicle.heading_vector.y, 1.0, abs_tol=1e-12)
    assert math.isclose(vehicle.u_hat.y, 1.0, abs_tol=1e-12)

    # Transverse n_hat = (-sin(pi/2), cos(pi/2)) = (-1, 0)
    assert math.isclose(vehicle.normal_vector.x, -1.0, abs_tol=1e-12)
    assert math.isclose(vehicle.normal_vector.y, 0.0, abs_tol=1e-12)

    assert vehicle.altitude == 10000.0
    assert math.isclose(vehicle.speed, math.hypot(100.0, 200.0), rel_tol=1e-12)


def test_flight_vehicle_compute_derivatives():
    """Verify derivative vector dS/dt accounts for gravity, thrust, and aero forces."""
    earth = get_earth()
    f9 = create_falcon_9()
    vehicle = FlightVehicle(
        rocket=f9,
        body=earth,
        position=Vector2D(0.0, earth.radius + 50000.0),
        velocity=Vector2D(0.0, 0.0),
        attitude=math.pi / 2,
    )
    vehicle.ignite()
    vehicle.set_throttle(1.0)

    deriv = vehicle.compute_derivatives(vehicle.state_vector, 0.0)
    assert isinstance(deriv, np.ndarray)
    assert deriv.shape == (6,)
    # d(pos)/dt = vel = [0, 0]
    assert math.isclose(deriv[0], 0.0, abs_tol=1e-12)
    assert math.isclose(deriv[1], 0.0, abs_tol=1e-12)

    # Net vertical acceleration a_y = (T - m*g) / m > 0 (TWR > 1.3)
    assert deriv[3] > 0.0


# ============================================================================
# 3. Core Flight Simulation Scenarios
# ============================================================================


def test_launchpad_static_equilibrium():
    """Launchpad Static Equilibrium Test.

    Place vehicle on Earth's surface without thrust. Verify over 10.0 s that
    altitude remains 0.0, position rotates synchronously with Earth, and state
    remains ON_PAD.
    """
    earth = get_earth()
    f9 = create_falcon_9()
    # Place on Earth surface at North Pole: (0, R_earth)
    pos_init = Vector2D(0.0, earth.radius)
    vehicle = FlightVehicle(rocket=f9, body=earth, position=pos_init)

    assert vehicle.state == VehicleState.ON_PAD
    assert math.isclose(vehicle.altitude, 0.0, abs_tol=1e-5)

    omega_earth = 2.0 * math.pi / earth.rotation_period
    dt = 0.1
    total_duration = 10.0
    num_steps = int(total_duration / dt)

    for step_idx in range(num_steps):
        snap = vehicle.step(dt=dt, throttle=0.0)
        assert vehicle.state == VehicleState.ON_PAD
        assert math.isclose(snap.altitude, 0.0, abs_tol=1e-5)
        assert math.isclose(vehicle.altitude, 0.0, abs_tol=1e-5)

    # Verify synchronous rotation with Earth after 10.0 s
    expected_rotation_angle = (math.pi / 2) + (omega_earth * total_duration)
    actual_angle = math.atan2(vehicle.position.y, vehicle.position.x)
    assert math.isclose(actual_angle, expected_rotation_angle, rel_tol=1e-7)
    assert math.isclose(vehicle.position.magnitude(), earth.radius, rel_tol=1e-9)


def test_vertical_ascent_liftoff():
    """Vertical Ascent (Liftoff) Test.

    Set throttle to 1.0 on a rocket with TWR > 1.3. Verify state transitions
    from ON_PAD to IN_FLIGHT, clears the dense atmosphere, and vertical velocity
    is strictly positive throughout.
    """
    earth = get_earth()
    f9 = create_falcon_9()
    pos_init = Vector2D(0.0, earth.radius)
    vehicle = FlightVehicle(rocket=f9, body=earth, position=pos_init)

    assert vehicle.state == VehicleState.ON_PAD

    # Step 1: Set throttle to 1.0 (Falcon 9 TWR ~1.4 > 1.3)
    snap0 = vehicle.step(dt=0.1, throttle=1.0)
    assert vehicle.state == VehicleState.IN_FLIGHT
    assert snap0.altitude > 0.0

    # Advance vertical ascent until clearing dense atmosphere (> 15 km)
    dt = 0.5
    for _ in range(140):
        snap = vehicle.step(dt=dt)
        r_hat = vehicle.position.normalized()
        v_radial = vehicle.velocity.dot(r_hat)
        # Vertical velocity strictly positive
        assert v_radial > 0.0, f"Vertical velocity became non-positive: {v_radial}"

    # Verify vehicle successfully cleared the dense troposphere/stratosphere
    assert vehicle.altitude > 15000.0
    assert vehicle.state == VehicleState.IN_FLIGHT


def test_sas_attitude_stabilization():
    """SAS Attitude Stabilization Test.

    Put an active rocket in flight with an intentional initial angular velocity
    disturbance (omega_0 = 0.1 rad/s). Verify that SAS gimbal deflection damps
    omega back toward 0.0 and restores commanded target pitch.
    """
    earth = get_earth()
    f9 = create_falcon_9()
    # High-altitude flight (50 km) to test active propulsion attitude damping
    pos = Vector2D(0.0, earth.radius + 50000.0)
    target_pitch = math.pi / 2  # Commanded pitch straight up (+Y)

    vehicle = FlightVehicle(
        rocket=f9,
        body=earth,
        position=pos,
        velocity=Vector2D(0.0, 500.0),
        attitude=target_pitch,
        angular_velocity=0.1,  # Disturbance omega_0 = 0.1 rad/s
    )
    vehicle.autopilot.set_target_pitch(target_pitch)
    vehicle.set_throttle(1.0)

    assert vehicle.angular_velocity == 0.1
    assert vehicle.state == VehicleState.IN_FLIGHT

    # Simulate closed-loop flight with SAS active over 10.0 s
    dt = 0.1
    for _ in range(100):
        snap = vehicle.step(dt=dt)

    # Verify rate damping: omega damped to near 0.0 rad/s
    assert abs(vehicle.angular_velocity) < 0.005, (
        f"Angular velocity was not damped: {vehicle.angular_velocity}"
    )

    # Verify attitude restoration: pitch angle restored to target pitch
    pitch_error = abs(vehicle.attitude - target_pitch)
    assert pitch_error < 0.01, f"Pitch error too large: {pitch_error} rad"


def test_ground_impact_detection():
    """Ground Impact Detection Test.

    Place a vehicle in free-fall from 1000 m above the surface without engines.
    Verify that when altitude reaches 0.0, state becomes DESTROYED.
    """
    earth = get_earth()
    f9 = create_falcon_9()
    # Free-fall from 1000 m
    pos = Vector2D(0.0, earth.radius + 1000.0)
    vehicle = FlightVehicle(
        rocket=f9,
        body=earth,
        position=pos,
        velocity=Vector2D(0.0, 0.0),
        attitude=math.pi / 2,
    )

    assert vehicle.state == VehicleState.IN_FLIGHT

    dt = 0.1
    step_count = 0
    max_steps = 300

    while vehicle.state == VehicleState.IN_FLIGHT and step_count < max_steps:
        snap = vehicle.step(dt=dt, throttle=0.0)
        step_count += 1

    # High-speed free-fall impact (> 100 m/s >> 12 m/s limit)
    assert vehicle.state == VehicleState.DESTROYED
    assert math.isclose(vehicle.altitude, 0.0, abs_tol=1e-4)


def test_safe_touchdown_landing():
    """Verify gentle touchdown (< 12 m/s and vertical) transitions to LANDED state."""
    earth = get_earth()
    f9 = create_falcon_9()
    # 0.5 m above ground descending gently at 2.0 m/s
    pos = Vector2D(0.0, earth.radius + 0.5)
    v_surface = earth.wind_velocity_at(pos)
    vel = v_surface - Vector2D(0.0, 2.0)

    vehicle = FlightVehicle(
        rocket=f9,
        body=earth,
        position=pos,
        velocity=vel,
        attitude=math.pi / 2,  # Upright vertical
    )

    assert vehicle.state == VehicleState.IN_FLIGHT

    vehicle.step(dt=0.5, throttle=0.0)
    assert vehicle.state == VehicleState.LANDED
    assert math.isclose(vehicle.altitude, 0.0, abs_tol=1e-5)


def test_suborbital_flight_and_staging():
    """Suborbital Flight & Staging Integration Test.

    Run a multi-step ascent flight with a pitch-over maneuver:
        - theta = 85 deg at 2 km
        - theta = 60 deg at 15 km
    Trigger staging when Stage 1 burns out:
        - Verify Stage 1 separates cleanly.
        - Verify Stage 2 ignites.
        - Verify vehicle successfully reaches an apoapsis > 100 km (space).
    """
    earth = get_earth()
    f9 = create_falcon_9()
    pos_init = Vector2D(0.0, earth.radius)
    vehicle = FlightVehicle(rocket=f9, body=earth, position=pos_init)
    vehicle.set_throttle(1.0)

    dt = 1.0
    staged = False
    stage1_burnout = False

    for _ in range(250):
        alt = vehicle.altitude

        # Guidance pitch program
        if alt < 2000.0:
            target_pitch = math.radians(90.0)
        elif alt < 15000.0:
            target_pitch = math.radians(85.0)
        else:
            target_pitch = math.radians(60.0)

        snap = vehicle.step(dt=dt, target_pitch=target_pitch)

        # Stage 1 burnout detection
        if not staged and vehicle.rocket.active_stage is not None:
            if vehicle.rocket.active_stage.propellant_mass() <= 0.0 or snap.thrust == 0.0:
                stage1_burnout = True
                separated_stage = vehicle.stage(ignite=True)
                staged = True

                # Assert Stage 1 separated cleanly
                assert separated_stage is not None
                assert separated_stage.is_separated is True
                assert vehicle.rocket.active_stage_index == 1
                assert vehicle.rocket.active_stage.is_active is True
                assert any(
                    eng.is_ignited for eng in vehicle.rocket.active_stage.engines
                )

        # Termination criterion: space apoapsis reached after staging
        if staged and snap.apoapsis > 100000.0:
            break

    assert stage1_burnout, "Stage 1 did not reach propellant burnout"
    assert staged, "Vehicle did not trigger staging separation"
    assert snap.apoapsis > 100000.0, (
        f"Vehicle failed to reach 100 km space apoapsis: {snap.apoapsis / 1000:.1f} km"
    )


def test_burnout_step_applies_thrust_in_rk4():
    """Verify that thrust is applied in RK4 dynamics even on the propellant burnout step."""
    from models.parts import Engine, FuelTank
    earth = get_earth()
    eng = Engine(
        id="eng",
        name="Engine",
        dry_mass=100.0,
        length=1.0,
        radius=0.5,
        position_offset=0.0,
        thrust_vac=50000.0,
        thrust_sl=50000.0,
        isp_vac=300.0,
        isp_sl=300.0,
        is_ignited=True,
        throttle=1.0,
    )
    # Mass flow = 50000 / (300 * 9.80665) = ~16.995 kg/s
    # Give fuel for 0.5 seconds = ~8.5 kg
    tank = FuelTank(
        id="tank",
        name="Tank",
        dry_mass=100.0,
        length=2.0,
        radius=0.5,
        position_offset=1.0,
        propellant_mass=8.5,
        propellant_capacity=100.0,
    )
    stage = Stage(stage_id="s1", name="Stage 1", parts=[eng, tank], position_offset=0.0)
    rocket = Rocket(stages=[stage])
    
    # Place in space (no drag) with zero initial velocity
    pos = Vector2D(0.0, earth.radius + 200000.0)
    vehicle = FlightVehicle(rocket=rocket, body=earth, position=pos, velocity=Vector2D(0.0, 0.0), state=VehicleState.IN_FLIGHT)
    vehicle.attitude = math.pi / 2.0  # Pointing radially outward (+y)
    
    # Single step of dt = 1.0 s, fuel should burn out halfway
    snap = vehicle.step(dt=1.0)
    
    assert tank.propellant_mass == 0.0
    assert eng.is_ignited is False
    assert snap.thrust > 0.0
    
    # Velocity should reflect upward thrust minus gravity, NOT just pure gravity freefall
    g_acc = earth.gravitational_acceleration_at(pos).magnitude()
    freefall_vy = -g_acc * 1.0
    assert vehicle.velocity.y > freefall_vy + 10.0, "Thrust was not applied in RK4 during burnout step"

