"""Unit test suite verifying modular rocket representation, staging, and dynamics.

Tests:
    1. Part Model & Subclasses (Part, FuelTank, Engine, Decoupler, Payload).
    2. JSON Roundtrip serialization and deserialization.
    3. Tsiolkovsky delta-v and mass flow rate integration validation.
    4. Dynamic Center of Mass (CoM) forward shift during propellant depletion.
    5. Parallel Axis Theorem & dynamic Moment of Inertia (I_zz).
    6. Engine gimbal deflection, thrust vectoring, and rotational torque around dynamic CoM.
    7. Multi-stage separation, mass drop, and second-stage readiness.
    8. Propellant starvation and engine cutoff.
    9. Pressure-dependent Isp and thrust scaling.
    10. Aerodynamic Center of Pressure (CoP) calculation.
"""

from __future__ import annotations

import json
import math
import numpy as np
import pytest

from core.constants import g0
from core.vector2d import Vector2D
from models.parts import (
    P0,
    STANDARD_SEA_LEVEL_PRESSURE,
    Decoupler,
    Engine,
    FuelTank,
    Part,
    Payload,
)
from models.presets import (
    FALCON_9_CONFIG,
    create_falcon_9,
    make_falcon_9,
)
from models.rocket import Rocket, Stage


# ============================================================================
# 1. Part Model & Subclass Unit Tests
# ============================================================================


def test_part_base_properties():
    """Verify base Part geometric, mass, and inertial properties."""
    part = Part(
        id="structural_tube",
        name="Structural Interstage Tube",
        dry_mass=500.0,
        length=4.0,
        radius=1.5,
        position_offset=10.0,
    )
    assert part.id == "structural_tube"
    assert part.current_mass == 500.0
    # Projected area: 2 * r * L = 2 * 1.5 * 4.0 = 12.0 m^2
    assert math.isclose(part.projected_area, 12.0, rel_tol=1e-12)

    # I_local = (1/12) * m * (3 * r^2 + L^2)
    # = (1/12) * 500 * (3 * 2.25 + 16.0) = (500 / 12) * 22.75 = 947.9166666666666
    expected_i = (1.0 / 12.0) * 500.0 * (3.0 * 1.5**2 + 4.0**2)
    assert math.isclose(part.local_moment_of_inertia(), expected_i, rel_tol=1e-12)


def test_fuel_tank_drain_and_mass():
    """Verify FuelTank propellant draining, clamping, and dynamic inertia."""
    tank = FuelTank(
        id="test_tank",
        name="Test Fuel Tank",
        dry_mass=1000.0,
        length=10.0,
        radius=2.0,
        position_offset=5.0,
        propellant_mass=8000.0,
        propellant_capacity=10000.0,
    )
    assert tank.current_mass == 9000.0

    # Normal drain
    drained = tank.drain(3000.0)
    assert math.isclose(drained, 3000.0, rel_tol=1e-12)
    assert math.isclose(tank.propellant_mass, 5000.0, rel_tol=1e-12)
    assert math.isclose(tank.current_mass, 6000.0, rel_tol=1e-12)

    # Drain more than remaining: clamped at remaining propellant
    drained_excess = tank.drain(7000.0)
    assert math.isclose(drained_excess, 5000.0, rel_tol=1e-12)
    assert math.isclose(tank.propellant_mass, 0.0, abs_tol=1e-12)
    assert math.isclose(tank.current_mass, 1000.0, rel_tol=1e-12)

    # Drain when empty
    drained_empty = tank.drain(500.0)
    assert math.isclose(drained_empty, 0.0, abs_tol=1e-12)

    # Negative drain ignored
    drained_neg = tank.drain(-100.0)
    assert math.isclose(drained_neg, 0.0, abs_tol=1e-12)


def test_engine_throttling_and_gimbal_clamping():
    """Verify Engine throttle and gimbal clamping to prescribed limits."""
    eng = Engine(
        id="eng1",
        name="Test Engine",
        dry_mass=400.0,
        length=2.0,
        radius=0.5,
        position_offset=1.0,
        thrust_vac=100000.0,
        thrust_sl=85000.0,
        isp_vac=300.0,
        isp_sl=270.0,
        gimbal_limit=0.08726646,  # 5 degrees
        current_gimbal=0.0,
        throttle=1.0,
    )

    # Over-throttle clamped to 1.0
    eng.set_throttle(1.5)
    assert math.isclose(eng.throttle, 1.0, rel_tol=1e-12)

    # Under-throttle clamped to 0.0
    eng.set_throttle(-0.2)
    assert math.isclose(eng.throttle, 0.0, rel_tol=1e-12)

    # Normal throttle
    eng.set_throttle(0.65)
    assert math.isclose(eng.throttle, 0.65, rel_tol=1e-12)

    # Gimbal positive limit clamp
    eng.set_gimbal(0.2)
    assert math.isclose(eng.current_gimbal, 0.08726646, rel_tol=1e-12)

    # Gimbal negative limit clamp
    eng.set_gimbal(-0.2)
    assert math.isclose(eng.current_gimbal, -0.08726646, rel_tol=1e-12)

    # Valid gimbal angle
    eng.set_gimbal(0.04)
    assert math.isclose(eng.current_gimbal, 0.04, rel_tol=1e-12)


# ============================================================================
# 2. Pressure-Dependent Isp and Thrust Scaling
# ============================================================================


def test_pressure_dependent_isp_and_thrust():
    """Verify Isp and thrust interpolation between vacuum and sea-level pressures."""
    eng = Engine(
        id="m1d",
        name="Merlin 1D",
        dry_mass=470.0,
        length=2.7,
        radius=0.6,
        position_offset=1.35,
        thrust_vac=914000.0,
        thrust_sl=845000.0,
        isp_vac=311.0,
        isp_sl=282.0,
        throttle=1.0,
        is_ignited=True,
    )

    # 1. Vacuum condition (P = 0)
    isp_vac = eng.get_isp(0.0)
    assert math.isclose(isp_vac, 311.0, rel_tol=1e-12)
    t_vac, mdot_vac = eng.get_thrust_and_mass_flow(0.0)
    assert math.isclose(t_vac, 914000.0, rel_tol=1e-12)
    expected_mdot_vac = 914000.0 / (311.0 * g0)
    assert math.isclose(mdot_vac, expected_mdot_vac, rel_tol=1e-12)

    # 2. Standard Sea-Level condition (P = 101325 Pa)
    isp_sl = eng.get_isp(101325.0)
    assert math.isclose(isp_sl, 282.0, rel_tol=1e-12)
    t_sl, mdot_sl = eng.get_thrust_and_mass_flow(101325.0)
    assert math.isclose(t_sl, 845000.0, rel_tol=1e-12)
    expected_mdot_sl = 845000.0 / (282.0 * g0)
    assert math.isclose(mdot_sl, expected_mdot_sl, rel_tol=1e-12)

    # 3. Half atmosphere (P = 50662.5 Pa)
    p_half = 0.5 * STANDARD_SEA_LEVEL_PRESSURE
    isp_half = eng.get_isp(p_half)
    expected_isp_half = 311.0 - (311.0 - 282.0) * 0.5
    assert math.isclose(isp_half, expected_isp_half, rel_tol=1e-12)
    t_half, _ = eng.get_thrust_and_mass_flow(p_half)
    expected_t_half = 914000.0 - (914000.0 - 845000.0) * 0.5
    assert math.isclose(t_half, expected_t_half, rel_tol=1e-12)

    # 4. Super-atmospheric pressure clamped at sea-level values
    isp_high = eng.get_isp(200000.0)
    assert math.isclose(isp_high, 282.0, rel_tol=1e-12)
    t_high, _ = eng.get_thrust_and_mass_flow(200000.0)
    assert math.isclose(t_high, 845000.0, rel_tol=1e-12)

    # 5. Shut down engine produces zero thrust and zero mass flow
    eng.shutdown()
    t_off, mdot_off = eng.get_thrust_and_mass_flow(0.0)
    assert t_off == 0.0 and mdot_off == 0.0


# ============================================================================
# 3. JSON Roundtrip Tests
# ============================================================================


def test_json_roundtrip_falcon_9():
    """Verify loading, serializing to JSON, and reloading preserves all parameters."""
    original_rocket = create_falcon_9()

    # Verify preset metrics
    initial_mass = original_rocket.total_mass()
    initial_com = original_rocket.center_of_mass()
    initial_moi = original_rocket.moment_of_inertia()
    initial_cop = original_rocket.center_of_pressure()

    # 1. Export to dict
    rocket_dict = original_rocket.to_dict()
    assert isinstance(rocket_dict, dict)
    assert rocket_dict["name"] == "Falcon 9"
    assert len(rocket_dict["stages"]) == 2

    # 2. Export to JSON string
    json_str = original_rocket.to_json(indent=2)
    assert isinstance(json_str, str)

    # 3. Reconstruct from JSON string
    reloaded_from_str = Rocket.from_json(json_str)
    assert reloaded_from_str.name == original_rocket.name
    assert len(reloaded_from_str.stages) == len(original_rocket.stages)
    assert math.isclose(reloaded_from_str.total_mass(), initial_mass, rel_tol=1e-12)
    assert math.isclose(reloaded_from_str.center_of_mass(), initial_com, rel_tol=1e-12)
    assert math.isclose(reloaded_from_str.moment_of_inertia(), initial_moi, rel_tol=1e-12)
    assert math.isclose(reloaded_from_str.center_of_pressure(), initial_cop, rel_tol=1e-12)

    # 4. Reconstruct from dictionary
    reloaded_from_dict = Rocket.from_json(rocket_dict)
    assert math.isclose(reloaded_from_dict.total_mass(), initial_mass, rel_tol=1e-12)

    # 5. Verify every stage and part attribute matches
    for orig_stage, reload_stage in zip(original_rocket.stages, reloaded_from_str.stages):
        assert orig_stage.stage_id == reload_stage.stage_id
        assert orig_stage.name == reload_stage.name
        assert orig_stage.staging_order == reload_stage.staging_order
        assert orig_stage.is_active == reload_stage.is_active
        assert orig_stage.is_separated == reload_stage.is_separated
        assert math.isclose(orig_stage.position_offset, reload_stage.position_offset, rel_tol=1e-12)
        assert len(orig_stage.parts) == len(reload_stage.parts)

        for orig_part, reload_part in zip(orig_stage.parts, reload_stage.parts):
            assert orig_part.__class__ is reload_part.__class__
            assert orig_part.id == reload_part.id
            assert orig_part.name == reload_part.name
            assert math.isclose(orig_part.dry_mass, reload_part.dry_mass, rel_tol=1e-12)
            assert math.isclose(orig_part.length, reload_part.length, rel_tol=1e-12)
            assert math.isclose(orig_part.radius, reload_part.radius, rel_tol=1e-12)
            assert math.isclose(
                orig_part.position_offset, reload_part.position_offset, rel_tol=1e-12
            )

            if isinstance(orig_part, FuelTank):
                assert math.isclose(
                    orig_part.propellant_mass, reload_part.propellant_mass, rel_tol=1e-12
                )
                assert math.isclose(
                    orig_part.propellant_capacity,
                    reload_part.propellant_capacity,
                    rel_tol=1e-12,
                )
            elif isinstance(orig_part, Engine):
                assert math.isclose(orig_part.thrust_vac, reload_part.thrust_vac, rel_tol=1e-12)
                assert math.isclose(orig_part.thrust_sl, reload_part.thrust_sl, rel_tol=1e-12)
                assert math.isclose(orig_part.isp_vac, reload_part.isp_vac, rel_tol=1e-12)
                assert math.isclose(orig_part.isp_sl, reload_part.isp_sl, rel_tol=1e-12)
                assert math.isclose(
                    orig_part.gimbal_limit, reload_part.gimbal_limit, rel_tol=1e-12
                )
            elif isinstance(orig_part, Decoupler):
                assert math.isclose(
                    orig_part.separation_impulse,
                    reload_part.separation_impulse,
                    rel_tol=1e-12,
                )


# ============================================================================
# 4. Tsiolkovsky Delta-V and Mass Flow Validation
# ============================================================================


def test_tsiolkovsky_delta_v_and_mass_conservation():
    """Verify propellant exhaustion burn matches Tsiolkovsky delta-v and mass flow integral."""
    # Construct a clean single-stage rocket for analytical validation
    eng = Engine(
        id="eng_tsiolkovsky",
        name="Tsiolkovsky Engine",
        dry_mass=1000.0,
        length=2.0,
        radius=1.0,
        position_offset=1.0,
        thrust_vac=200000.0,
        thrust_sl=200000.0,  # Constant thrust for exact analytical check
        isp_vac=300.0,
        isp_sl=300.0,        # Constant Isp = 300 s
        throttle=1.0,
        is_ignited=True,
    )

    tank = FuelTank(
        id="tank_tsiolkovsky",
        name="Tsiolkovsky Tank",
        dry_mass=2000.0,
        length=8.0,
        radius=1.0,
        position_offset=6.0,
        propellant_mass=15000.0,
        propellant_capacity=15000.0,
    )

    stage = Stage(
        stage_id="s1",
        name="Validation Stage",
        parts=[eng, tank],
        staging_order=0,
        is_active=True,
        position_offset=0.0,
    )

    rocket = Rocket(name="Analytical Test", stages=[stage], active_stage_index=0)

    m0 = rocket.total_mass()  # 1000 + 2000 + 15000 = 18000 kg
    m_dry = rocket.dry_mass()  # 3000 kg
    prop_initial = rocket.propellant_mass()  # 15000 kg

    analytical_delta_v = 300.0 * g0 * math.log(m0 / m_dry)

    # Static burn integration loop
    dt = 0.5  # seconds
    total_burned_mass = 0.0
    numerical_delta_v = 0.0
    burn_steps = 0

    while True:
        m_before = rocket.total_mass()
        thrust, _, m_dot = rocket.burn_fuel(dt, ambient_pressure=0.0)
        if thrust <= 0.0:
            break

        actual_dm = m_dot * dt
        total_burned_mass += actual_dm

        # Midpoint mass for high-precision numerical Riemann integration
        m_after = rocket.total_mass()
        m_mid = 0.5 * (m_before + m_after)
        numerical_delta_v += (thrust / m_mid) * dt
        burn_steps += 1

    # 1. Total mass loss equals propellant capacity
    mass_loss = m0 - rocket.total_mass()
    assert math.isclose(mass_loss, prop_initial, rel_tol=1e-9)

    # 2. Burned mass matches integral of m_dot * dt within < 0.01%
    mass_error = abs(total_burned_mass - prop_initial) / prop_initial
    assert mass_error < 0.0001, f"Burned mass error {mass_error:.6f} exceeded 0.01%"

    # 3. Numerical delta_v matches analytical Tsiolkovsky within < 0.01%
    dv_error = abs(numerical_delta_v - analytical_delta_v) / analytical_delta_v
    assert dv_error < 0.0001, (
        f"Delta-V error {dv_error:.6f} exceeded 0.01% "
        f"(numerical: {numerical_delta_v:.3f}, analytical: {analytical_delta_v:.3f})"
    )


# ============================================================================
# 5. Dynamic Center of Mass (CoM) Shift
# ============================================================================


def test_dynamic_com_forward_shift():
    """Verify that as fuel drains from the lower stage, the rocket's CoM moves forward toward payload."""
    rocket = create_falcon_9()

    # Ignite Stage 1 engines
    for eng in rocket.active_stage.engines:
        eng.ignite()

    initial_com = rocket.center_of_mass()
    com_history = [initial_com]

    # Burn 10 increments of 12 seconds (total 120s of the ~152s burn)
    for _ in range(10):
        t, _, _ = rocket.burn_fuel(12.0, ambient_pressure=0.0)
        assert t > 0.0
        com = rocket.center_of_mass()
        com_history.append(com)

    # Verify monotonic forward movement of CoM
    for i in range(1, len(com_history)):
        assert com_history[i] > com_history[i - 1], (
            f"CoM did not advance forward at step {i}: {com_history[i]} <= {com_history[i-1]}"
        )

    # Final burned CoM should be much higher than initial CoM
    assert com_history[-1] > initial_com + 5.0


# ============================================================================
# 6. Parallel Axis Theorem & Moment of Inertia (I_zz)
# ============================================================================


def test_parallel_axis_theorem_and_moi_decrease():
    """Verify analytical Parallel Axis Theorem calculation and monotonic decrease during burn."""
    # Test with a controlled 2-part rocket to verify formula:
    # I_zz = sum(I_local,i + m_i * (y_i - y_com)^2)
    p1 = Part(
        id="base_p",
        name="Base Part",
        dry_mass=100.0,
        length=2.0,
        radius=1.0,
        position_offset=1.0,  # y1 = 1.0
    )
    p2 = Part(
        id="nose_p",
        name="Nose Part",
        dry_mass=100.0,
        length=2.0,
        radius=1.0,
        position_offset=9.0,  # y2 = 9.0
    )
    st = Stage(stage_id="s_test", name="Test Stage", parts=[p1, p2], position_offset=0.0)
    r_test = Rocket(name="Test", stages=[st], active_stage_index=0)

    # CoM: (100 * 1 + 100 * 9) / 200 = 5.0 m
    assert math.isclose(r_test.center_of_mass(), 5.0, rel_tol=1e-12)

    # I_local for each: (1/12) * 100 * (3 * 1^2 + 2^2) = (100 / 12) * 7 = 58.333333333333336
    i_local = (1.0 / 12.0) * 100.0 * (3.0 * 1.0**2 + 2.0**2)
    # Parallel axis term for each: 100 * (1 - 5)^2 = 100 * 16 = 1600.0
    d1_sq = (1.0 - 5.0) ** 2
    d2_sq = (9.0 - 5.0) ** 2
    expected_total_i = (i_local + 100.0 * d1_sq) + (i_local + 100.0 * d2_sq)

    assert math.isclose(r_test.moment_of_inertia(), expected_total_i, rel_tol=1e-12)

    # Now verify on Falcon 9 that I_zz decreases as fuel is consumed
    falcon = create_falcon_9()
    for eng in falcon.active_stage.engines:
        eng.ignite()

    initial_moi = falcon.moment_of_inertia()
    moi_history = [initial_moi]

    for _ in range(5):
        falcon.burn_fuel(25.0, ambient_pressure=0.0)
        moi_history.append(falcon.moment_of_inertia())

    for i in range(1, len(moi_history)):
        assert moi_history[i] < moi_history[i - 1], (
            f"MoI did not decrease at step {i}: {moi_history[i]} >= {moi_history[i-1]}"
        )


# ============================================================================
# 7. Engine Gimbal Deflection and Rotational Torque
# ============================================================================


def test_gimbal_thrust_vector_and_torque():
    """Verify thrust vectoring and torque generation about dynamic CoM."""
    rocket = create_falcon_9()
    stage = rocket.active_stage
    assert stage is not None

    # Ignite only one engine for straightforward validation
    for eng in stage.engines:
        eng.shutdown()
    eng0 = stage.engines[0]
    eng0.ignite()

    y_com = rocket.center_of_mass()
    y_eng = stage.position_offset + eng0.position_offset
    arm = y_eng - y_com
    # Engine is at the bottom, so arm should be strictly negative
    assert arm < 0.0

    # Case 1: Gimbal = 0.0 -> Torque must be zero
    eng0.set_gimbal(0.0)
    force, torque = rocket.thrust_force_and_torque(ambient_pressure=0.0)
    assert math.isclose(force.x, 0.0, abs_tol=1e-12)
    assert math.isclose(force.y, eng0.thrust_vac, rel_tol=1e-12)
    assert math.isclose(torque, 0.0, abs_tol=1e-12)

    # Case 2: Gimbal = +5 degrees (+0.08726646 rad)
    delta_pos = 0.08726646259971647
    eng0.set_gimbal(delta_pos)
    force, torque = rocket.thrust_force_and_torque(ambient_pressure=0.0)

    expected_fx = eng0.thrust_vac * math.sin(delta_pos)
    expected_fy = eng0.thrust_vac * math.cos(delta_pos)
    # r x F = -arm * fx. Since arm < 0 and fx > 0, torque is positive (counter-clockwise)
    expected_tau = -arm * expected_fx

    assert math.isclose(force.x, expected_fx, rel_tol=1e-12)
    assert math.isclose(force.y, expected_fy, rel_tol=1e-12)
    assert math.isclose(torque, expected_tau, rel_tol=1e-12)
    assert torque > 0.0

    # Case 3: Gimbal = -5 degrees -> Torque flips sign
    delta_neg = -0.08726646259971647
    eng0.set_gimbal(delta_neg)
    force_neg, torque_neg = rocket.thrust_force_and_torque(ambient_pressure=0.0)
    assert math.isclose(torque_neg, -expected_tau, rel_tol=1e-12)
    assert torque_neg < 0.0


# ============================================================================
# 8. Multi-Stage Separation
# ============================================================================


def test_staging_separation_and_stage2_readiness():
    """Verify stage decoupling, instantaneous mass drop, and Stage 2 activation."""
    rocket = create_falcon_9()

    assert rocket.active_stage_index == 0
    assert rocket.active_stage.stage_id == "stage_1"
    initial_total_mass = rocket.total_mass()
    stage1_mass = rocket.active_stage.total_mass()
    stage2_mass = rocket.stages[1].total_mass()

    # Ignite Stage 1
    for eng in rocket.active_stage.engines:
        eng.ignite()

    # Trigger separation
    decoupled_stage = rocket.stage()

    # Verify decoupled stage attributes
    assert decoupled_stage is not None
    assert decoupled_stage.stage_id == "stage_1"
    assert decoupled_stage.is_separated is True
    assert decoupled_stage.is_active is False
    # All Stage 1 engines should be shut down
    for eng in decoupled_stage.engines:
        assert not eng.is_ignited

    # Verify rocket active state after staging
    assert rocket.active_stage_index == 1
    assert rocket.active_stage is not None
    assert rocket.active_stage.stage_id == "stage_2"
    assert rocket.active_stage.is_active is True
    assert rocket.active_stage.is_separated is False

    # Mass drops by exactly stage 1 mass
    current_mass = rocket.total_mass()
    assert math.isclose(current_mass, stage2_mass, rel_tol=1e-12)
    assert math.isclose(current_mass, initial_total_mass - stage1_mass, rel_tol=1e-12)

    # Stage 2 is ready to fire
    mvac = rocket.active_stage.engines[0]
    assert mvac.id == "merlin_vac"
    mvac.ignite()
    thrust, gimbal, mdot = rocket.burn_fuel(1.0, ambient_pressure=0.0)
    assert math.isclose(thrust, 981000.0, rel_tol=1e-12)
    assert mdot > 0.0

    # Further staging when no more stages exist returns None
    second_decouple = rocket.stage()
    assert second_decouple.stage_id == "stage_2"
    third_decouple = rocket.stage()
    assert third_decouple is None


# ============================================================================
# 9. Propellant Starvation and Automatic Cutoff
# ============================================================================


def test_propellant_starvation_and_cutoff():
    """Verify engines cut off to zero thrust immediately when fuel is exhausted."""
    # Create a small stage with limited fuel
    eng = Engine(
        id="small_eng",
        name="Small Engine",
        dry_mass=50.0,
        length=1.0,
        radius=0.3,
        position_offset=0.5,
        thrust_vac=1000.0,
        thrust_sl=1000.0,
        isp_vac=100.0,
        isp_sl=100.0,
        throttle=1.0,
        is_ignited=True,
    )
    # Mass flow = 1000 / (100 * 9.80665) = 1.0197 kg/s
    # Tank has 2.0 kg fuel -> burns for ~1.96 seconds
    tank = FuelTank(
        id="small_tank",
        name="Small Tank",
        dry_mass=10.0,
        length=1.0,
        radius=0.3,
        position_offset=1.5,
        propellant_mass=2.0,
        propellant_capacity=2.0,
    )
    stage = Stage(
        stage_id="starve_stage",
        name="Starvation Test",
        parts=[eng, tank],
        is_active=True,
    )
    rocket = Rocket(name="Starvation Rocket", stages=[stage])

    # Step 1: 1.0 second burn -> drains ~1.0197 kg, ~0.98 kg remains
    t1, _, mdot1 = rocket.burn_fuel(1.0, ambient_pressure=0.0)
    assert t1 == 1000.0
    assert 0.0 < tank.propellant_mass < 2.0
    assert eng.is_ignited is True

    # Step 2: 2.0 second burn -> remaining ~0.98 kg exhausted during this step
    t2, _, mdot2 = rocket.burn_fuel(2.0, ambient_pressure=0.0)
    # Fractional burn
    assert 0.0 < t2 < 1000.0
    assert math.isclose(tank.propellant_mass, 0.0, abs_tol=1e-12)
    # Engine is shut down automatically
    assert eng.is_ignited is False

    # Step 3: subsequent burns return strictly 0 thrust and 0 mass flow
    t3, _, mdot3 = rocket.burn_fuel(1.0, ambient_pressure=0.0)
    assert t3 == 0.0
    assert mdot3 == 0.0
    force, torque = rocket.thrust_force_and_torque(ambient_pressure=0.0)
    assert force.magnitude() == 0.0
    assert torque == 0.0


# ============================================================================
# 10. Aerodynamic Center of Pressure (CoP) Calculation
# ============================================================================


def test_center_of_pressure_calculation():
    """Verify area-weighted Center of Pressure y_CoP = sum(A_i * y_i) / sum(A_i)."""
    # Part 1: r=1.0, L=2.0 -> A1 = 2 * 1 * 2 = 4.0; y1 = 2.0
    p1 = Part("p1", "Part 1", dry_mass=10.0, length=2.0, radius=1.0, position_offset=2.0)
    # Part 2: r=2.0, L=3.0 -> A2 = 2 * 2 * 3 = 12.0; y2 = 8.0
    p2 = Part("p2", "Part 2", dry_mass=10.0, length=3.0, radius=2.0, position_offset=8.0)

    stage = Stage("cop_stage", "CoP Stage", parts=[p1, p2], position_offset=0.0)
    rocket = Rocket("CoP Rocket", stages=[stage])

    # Expected CoP: (4.0 * 2.0 + 12.0 * 8.0) / (4.0 + 12.0) = (8.0 + 96.0) / 16.0 = 104 / 16 = 6.5 m
    expected_cop = (4.0 * 2.0 + 12.0 * 8.0) / 16.0
    assert math.isclose(rocket.center_of_pressure(), expected_cop, rel_tol=1e-12)
