"""Unit test suite verifying planetary atmospheric models, co-rotation, and aerodynamics.

Tests:
    1. Atmospheric density profiles, scale height decay, cubic tapering, and Karman cutoff.
    2. Speed of sound, tropospheric lapse rates, ideal gas pressure, and Mach number.
    3. Atmospheric planetary co-rotation and zero-relative-velocity launchpad condition.
    4. Rocket ascent dynamic pressure profile and Max Q emergence.
    5. Aerodynamic restoring and destabilizing torques around the Center of Mass.
    6. Orbital decay of a Low Earth Orbit satellite in the upper atmosphere using RK4.
    7. Cross-flow drag transitions, slender-body lift forces, and vacuum cutoffs.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.constants import g0
from core.integrator import (
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    integrate_trajectory,
    make_state,
    rk4_step,
)
from core.vector2d import Vector2D
from physics.aerodynamics import (
    AeroState,
    calculate_aero_forces,
    drag_force,
    dynamic_pressure,
    lift_coefficient,
    lift_force,
    mach_number,
    relative_velocity,
)
from physics.atmosphere import EARTH_ATMOSPHERE, AtmosphereModel
from physics.celestial import (
    EARTH,
    MOON,
    CelestialBody,
    get_earth,
    get_moon,
)
from physics.orbit import calculate_orbit


# ============================================================================
# 1. Density Profile Tests
# ============================================================================


def test_density_profile():
    """Verify Earth atmospheric density profile.

    Requirements:
        - Sea level density ~= 1.225 kg/m^3
        - Monotonic decrease across altitude range [0, 100 km]
        - Smooth cubic tapering near cutoff_altitude (90 km to 100 km)
        - Density is identically 0.0 at and above cutoff_altitude (100 km)
        - Density is identically 0.0 below sea level (negative altitude)
    """
    atmos = AtmosphereModel()

    # 1. Sea level density
    rho_0 = atmos.density(0.0)
    assert math.isclose(rho_0, 1.225, rel_tol=1e-12)

    # 2. Strict monotonic decrease from sea level to cutoff
    altitudes = np.linspace(0.0, 99990.0, 1000)
    densities = [atmos.density(float(h)) for h in altitudes]

    for i in range(len(densities) - 1):
        assert (
            densities[i] > densities[i + 1]
        ), f"Density not monotonically decreasing at alt {altitudes[i]:.1f} m"

    # 3. Barometric formula exactness before tapering (e.g. h = 10 km, 50 km)
    # Taper starts at 0.9 * 100,000 = 90,000 m
    for test_alt in [5000.0, 15000.0, 50000.0, 80000.0]:
        expected_rho = 1.225 * math.exp(-test_alt / 8500.0)
        assert math.isclose(atmos.density(test_alt), expected_rho, rel_tol=1e-12)

    # 4. Smooth cubic tapering in [90 km, 100 km]
    h_taper_start = 90000.0
    rho_at_taper_start = atmos.density(h_taper_start)
    expected_untapered_90k = 1.225 * math.exp(-h_taper_start / 8500.0)
    assert math.isclose(rho_at_taper_start, expected_untapered_90k, rel_tol=1e-12)

    # At midpoint of tapering: h = 95 km
    u_mid = 0.5
    smoothstep_factor = (1.0 - u_mid) ** 2 * (1.0 + 2.0 * u_mid)  # 0.25 * 2.0 = 0.5
    expected_rho_95k = 1.225 * math.exp(-95000.0 / 8500.0) * smoothstep_factor
    assert math.isclose(atmos.density(95000.0), expected_rho_95k, rel_tol=1e-12)

    # 5. Exact zero at and above cutoff altitude (100 km)
    assert atmos.density(100000.0) == 0.0
    assert atmos.density(100000.1) == 0.0
    assert atmos.density(150000.0) == 0.0
    assert atmos.density(400000.0) == 0.0

    # 6. Below surface
    assert atmos.density(-1.0) == 0.0
    assert atmos.density(-1000.0) == 0.0


def test_atmosphere_model_validation():
    """Verify input validation and error raising for invalid AtmosphereModel parameters."""
    with pytest.raises(ValueError, match="Sea level density"):
        AtmosphereModel(sea_level_density=-1.0)
    with pytest.raises(ValueError, match="scale height"):
        AtmosphereModel(scale_height=0.0)
    with pytest.raises(ValueError, match="Cutoff altitude"):
        AtmosphereModel(cutoff_altitude=-100.0)
    with pytest.raises(ValueError, match="Sea level temperature"):
        AtmosphereModel(sea_level_temperature=0.0)
    with pytest.raises(ValueError, match="Specific gas constant"):
        AtmosphereModel(specific_gas_constant=-10.0)
    with pytest.raises(ValueError, match="Adiabatic index"):
        AtmosphereModel(adiabatic_index=0.0)
    with pytest.raises(ValueError, match="Minimum temperature"):
        AtmosphereModel(min_temperature=-5.0)
    with pytest.raises(ValueError, match="Taper start ratio"):
        AtmosphereModel(taper_start_ratio=1.5)


# ============================================================================
# 2. Speed of Sound, Temperature & Pressure Tests
# ============================================================================


def test_speed_of_sound_and_temperature():
    """Verify atmospheric thermodynamic relations.

    Requirements:
        - Sea level temperature T_0 = 288.15 K
        - Sea level speed of sound a_0 = sqrt(gamma * R * T_0) ~= 340.29 m/s
        - Linear tropospheric lapse rate T(h) = T_0 - L * h
        - Temperature floor at T_min = 200.0 K
        - Static pressure via ideal gas law P = rho * R * T
        - Correct Mach number calculations
    """
    atmos = AtmosphereModel()

    # 1. Sea level conditions
    assert math.isclose(atmos.temperature(0.0), 288.15, rel_tol=1e-12)
    expected_a0 = math.sqrt(1.4 * 287.05 * 288.15)
    assert math.isclose(atmos.speed_of_sound(0.0), expected_a0, rel_tol=1e-12)
    assert math.isclose(atmos.speed_of_sound(0.0), 340.2935, abs_tol=0.01)

    # 2. Temperature lapse in troposphere (e.g. h = 5 km)
    expected_t_5km = 288.15 - 0.0065 * 5000.0  # 255.65 K
    assert math.isclose(atmos.temperature(5000.0), expected_t_5km, rel_tol=1e-12)

    # 3. Minimum temperature cap T_min = 200.0 K
    # Lapse rate reaches 200 K at h = (288.15 - 200.0) / 0.0065 = 13561.54 m
    assert atmos.temperature(20000.0) == 200.0
    assert atmos.temperature(50000.0) == 200.0
    assert atmos.temperature(90000.0) == 200.0

    # Speed of sound at isothermal cap
    expected_a_cap = math.sqrt(1.4 * 287.05 * 200.0)  # ~283.58 m/s
    assert math.isclose(atmos.speed_of_sound(20000.0), expected_a_cap, rel_tol=1e-12)
    assert math.isclose(atmos.speed_of_sound(80000.0), expected_a_cap, rel_tol=1e-12)

    # 4. Static pressure (ideal gas law: P = rho * R_spec * T)
    p_0 = atmos.pressure(0.0)
    expected_p0 = 1.225 * 287.05 * 288.15  # ~101323.08 Pa
    assert math.isclose(p_0, expected_p0, rel_tol=1e-12)

    # Pressure at 10 km
    rho_10k = atmos.density(10000.0)
    t_10k = atmos.temperature(10000.0)
    assert math.isclose(atmos.pressure(10000.0), rho_10k * 287.05 * t_10k, rel_tol=1e-12)

    # Vacuum pressure
    assert atmos.pressure(100000.0) == 0.0
    assert atmos.pressure(150000.0) == 0.0

    # 5. Mach number tests
    a_sl = atmos.speed_of_sound(0.0)
    assert math.isclose(mach_number(0.0, a_sl), 0.0, abs_tol=1e-12)
    assert math.isclose(mach_number(0.5 * a_sl, a_sl), 0.5, rel_tol=1e-12)
    assert math.isclose(mach_number(a_sl, a_sl), 1.0, rel_tol=1e-12)
    assert math.isclose(mach_number(2.0 * a_sl, a_sl), 2.0, rel_tol=1e-12)
    assert math.isclose(mach_number(5.0 * a_sl, a_sl), 5.0, rel_tol=1e-12)
    # Zero sound speed safety
    assert mach_number(300.0, 0.0) == 0.0


# ============================================================================
# 3. Atmospheric Co-rotation & Surface Wind Tests
# ============================================================================


def test_corotation_surface_wind():
    """Verify planetary atmospheric co-rotation and zero-aerodynamic-force launchpad state.

    Requirements:
        - A craft stationary on the equator rotates with Earth's surface at v_surface = omega * R
        - Atmospheric wind co-rotates at exactly v_wind = omega * R
        - Relative velocity v_rel = v_craft - v_wind ~= 0
        - Sitting on the launchpad produces exactly zero aerodynamic force and torque
        - Moon has no atmosphere -> wind velocity is strictly Vector2D(0, 0)
        - Non-rotating celestial body produces zero wind
    """
    earth = get_earth()
    r_eq = earth.radius
    omega = 2.0 * math.pi / earth.rotation_period

    # Position on equator (+X axis)
    launchpad_pos = Vector2D(r_eq, 0.0)

    # Spacecraft inertial velocity when stationary on the rotating Earth surface
    # v = omega x r = Vector2D(-omega * 0, omega * r_eq) = Vector2D(0, omega * r_eq)
    v_surface = Vector2D(0.0, omega * r_eq)
    assert math.isclose(v_surface.y, 464.58, abs_tol=1.0)  # ~465 m/s eastward

    # 1. Atmospheric wind velocity at launchpad
    v_wind = earth.wind_velocity_at(launchpad_pos)
    assert math.isclose(v_wind.x, 0.0, abs_tol=1e-12)
    assert math.isclose(v_wind.y, v_surface.y, rel_tol=1e-12)

    # 2. Relative flow velocity between rocket and atmosphere
    v_rel = relative_velocity(v_surface, v_wind)
    assert math.isclose(v_rel.magnitude(), 0.0, abs_tol=1e-12)

    # 3. Aerodynamic forces on launchpad vehicle
    # Rocket standing vertically (heading pointing away from center, +X axis)
    aero: AeroState = calculate_aero_forces(
        craft_pos=launchpad_pos,
        craft_vel=v_surface,
        attitude_angle=0.0,
        body=earth,
        reference_area=10.0,
        length=50.0,
        cd_axial=0.3,
        cd_transverse=1.2,
        center_of_mass_offset=25.0,
        center_of_pressure_offset=20.0,
    )

    assert math.isclose(aero.force.magnitude(), 0.0, abs_tol=1e-12)
    assert math.isclose(aero.torque, 0.0, abs_tol=1e-12)
    assert math.isclose(aero.dynamic_pressure, 0.0, abs_tol=1e-12)
    assert math.isclose(aero.mach, 0.0, abs_tol=1e-12)

    # 4. Moon (no atmosphere)
    moon = get_moon()
    moon_wind = moon.wind_velocity_at(Vector2D(moon.radius, 0.0))
    assert moon_wind == Vector2D(0.0, 0.0)

    # 5. Non-rotating Earth
    non_rot_earth = CelestialBody(
        name="NonRotatingEarth",
        mass=5.9722e24,
        radius=6.371e6,
        rotation_period=0.0,
        atmosphere=AtmosphereModel(),
    )
    assert non_rot_earth.wind_velocity_at(launchpad_pos) == Vector2D(0.0, 0.0)

    # 6. Outside atmospheric cutoff altitude (e.g. 150 km)
    pos_vacuum = Vector2D(earth.radius + 150000.0, 0.0)
    assert earth.wind_velocity_at(pos_vacuum) == Vector2D(0.0, 0.0)


# ============================================================================
# 4. Dynamic Pressure & Max Q Tests
# ============================================================================


def test_dynamic_pressure_and_max_q():
    """Simulate a vertical rocket ascent profile and verify Max Q emergence.

    Requirements:
        - Dynamic pressure starts at 0.0 on the launchpad.
        - Rises to a distinct global maximum ("Max Q") in the troposphere/stratosphere.
        - Beyond Max Q, density falloff dominates and dynamic pressure drops monotonically.
        - Dynamic pressure is identically 0.0 at and above cutoff altitude (100 km).
    """
    earth = get_earth()
    omega = 2.0 * math.pi / earth.rotation_period

    # Simulate vertical ascent with constant net acceleration a = 15.0 m/s^2
    net_acc = 15.0
    altitudes = np.linspace(0.0, 120000.0, 601)  # 200 m resolution
    q_profile: list[float] = []

    for h in altitudes:
        r = earth.radius + h
        pos = Vector2D(r, 0.0)

        # Rocket vertical speed v_vert = sqrt(2 * a * h)
        v_vert = math.sqrt(2.0 * net_acc * max(0.0, h))

        # Inertial velocity includes Earth surface rotational velocity so relative flow
        # is purely vertical
        vel = Vector2D(v_vert, omega * r)

        # Centerline heading pointing vertically upwards along +X axis
        aero = calculate_aero_forces(
            craft_pos=pos,
            craft_vel=vel,
            attitude_angle=0.0,
            body=earth,
            reference_area=1.0,
            cd_axial=0.3,
            cd_transverse=1.2,
        )
        q_profile.append(aero.dynamic_pressure)

    q_arr = np.array(q_profile)

    # 1. Lift-off dynamic pressure is zero
    assert math.isclose(q_arr[0], 0.0, abs_tol=1e-12)

    # 2. Maximum dynamic pressure occurs at realistic altitude (typically 8 km - 16 km)
    max_idx = int(np.argmax(q_arr))
    max_q = float(q_arr[max_idx])
    max_q_alt = float(altitudes[max_idx])

    assert max_q > 5000.0, f"Max Q {max_q} Pa is unexpectedly low"
    assert (
        8000.0 <= max_q_alt <= 18000.0
    ), f"Max Q altitude {max_q_alt/1000:.1f} km outside expected 8-18 km window"

    # 3. Post Max Q decay: dynamic pressure at 40 km is much less than Max Q
    idx_40k = int(np.searchsorted(altitudes, 40000.0))
    assert q_arr[idx_40k] < 0.15 * max_q

    # 4. At and above 100 km (Karman line), dynamic pressure is strictly 0.0
    cutoff_indices = np.where(altitudes >= 100000.0)[0]
    for idx in cutoff_indices:
        assert q_arr[idx] == 0.0, f"Dynamic pressure non-zero at alt {altitudes[idx]:.1f} m"


# ============================================================================
# 5. Aerodynamic Stability & Restoring Torque Tests
# ============================================================================


def test_aerodynamic_stability_and_torque():
    """Verify aerodynamic restoring and destabilizing torque around Center of Mass.

    Physics Convention:
        - When Center of Pressure (CoP) is behind Center of Mass (CoM),
          a positive perturbation angle of attack (+alpha) must produce a negative
          (clockwise) torque to restore attitude back toward alpha = 0.
        - When CoP is in front of CoM, positive alpha produces positive (counter-clockwise)
          torque, pitching the vehicle further away from zero (unstable).
    """
    earth = get_earth()

    # Place vehicle at 10 km altitude moving horizontally (+Y) at 300 m/s
    # Wind velocity at (radius + 10k, 0) is in +Y direction: (0, omega * r)
    pos = Vector2D(earth.radius + 10000.0, 0.0)
    wind_vel = earth.wind_velocity_at(pos)

    # Set craft velocity so relative velocity v_rel is strictly along +Y
    # v_rel = craft_vel - wind_vel = (0, 300)
    rel_speed = 300.0
    craft_vel = wind_vel + Vector2D(0.0, rel_speed)

    # Rocket geometry: length = 10 m
    # Stable configuration: CoM at 6.0 m, CoP at 4.0 m (CoP 2.0 m behind CoM)
    com_offset = 6.0
    cop_offset_stable = 4.0
    cop_offset_unstable = 7.0

    # Flight direction angle theta_vel = pi / 2 (along +Y)
    v_rel_hat = relative_velocity(craft_vel, wind_vel).normalized()
    assert math.isclose(v_rel_hat.x, 0.0, abs_tol=1e-12)
    assert math.isclose(v_rel_hat.y, 1.0, abs_tol=1e-12)

    # 1. Positive angle of attack: +5 degrees (pitching toward -X / counter-clockwise)
    # Attitude angle = pi / 2 + 5 deg
    alpha_pos = math.radians(5.0)
    theta_pos = math.pi / 2.0 + alpha_pos

    aero_stable_pos = calculate_aero_forces(
        craft_pos=pos,
        craft_vel=craft_vel,
        attitude_angle=theta_pos,
        body=earth,
        reference_area=1.0,
        length=10.0,
        cd_axial=0.3,
        cd_transverse=1.2,
        center_of_mass_offset=com_offset,
        center_of_pressure_offset=cop_offset_stable,
    )

    # Verify detected angle of attack
    assert math.isclose(aero_stable_pos.angle_of_attack, alpha_pos, rel_tol=1e-6)

    # RESTORING TORQUE CHECK:
    # Since alpha > 0, torque must be negative (clockwise) to drive alpha back to 0
    assert (
        aero_stable_pos.torque < 0.0
    ), f"Expected restoring torque < 0, got {aero_stable_pos.torque}"

    # 2. Negative angle of attack: -5 degrees
    alpha_neg = math.radians(-5.0)
    theta_neg = math.pi / 2.0 + alpha_neg

    aero_stable_neg = calculate_aero_forces(
        craft_pos=pos,
        craft_vel=craft_vel,
        attitude_angle=theta_neg,
        body=earth,
        reference_area=1.0,
        length=10.0,
        cd_axial=0.3,
        cd_transverse=1.2,
        center_of_mass_offset=com_offset,
        center_of_pressure_offset=cop_offset_stable,
    )

    assert math.isclose(aero_stable_neg.angle_of_attack, alpha_neg, rel_tol=1e-6)

    # RESTORING TORQUE CHECK:
    # Since alpha < 0, torque must be positive (counter-clockwise) to drive alpha back to 0
    assert (
        aero_stable_neg.torque > 0.0
    ), f"Expected restoring torque > 0, got {aero_stable_neg.torque}"
    # Torque magnitude should be symmetric
    assert math.isclose(aero_stable_neg.torque, -aero_stable_pos.torque, rel_tol=1e-6)

    # 3. Unstable configuration: CoP in front of CoM (cop_offset = 7.0 > com_offset = 6.0)
    aero_unstable = calculate_aero_forces(
        craft_pos=pos,
        craft_vel=craft_vel,
        attitude_angle=theta_pos,
        body=earth,
        reference_area=1.0,
        length=10.0,
        cd_axial=0.3,
        cd_transverse=1.2,
        center_of_mass_offset=com_offset,
        center_of_pressure_offset=cop_offset_unstable,
    )

    # DESTABILIZING TORQUE: torque has same sign as alpha (+), pitching vehicle further away
    assert (
        aero_unstable.torque > 0.0
    ), f"Expected destabilizing torque > 0, got {aero_unstable.torque}"

    # 4. Zero angle of attack (alpha = 0)
    aero_zero = calculate_aero_forces(
        craft_pos=pos,
        craft_vel=craft_vel,
        attitude_angle=math.pi / 2.0,
        body=earth,
        reference_area=1.0,
        length=10.0,
        cd_axial=0.3,
        cd_transverse=1.2,
        center_of_mass_offset=com_offset,
        center_of_pressure_offset=cop_offset_stable,
    )

    assert math.isclose(aero_zero.angle_of_attack, 0.0, abs_tol=1e-12)
    assert math.isclose(aero_zero.torque, 0.0, abs_tol=1e-12)
    # Drag coefficient is purely axial
    expected_f_drag = (
        aero_zero.dynamic_pressure * 1.0 * 0.3
    )  # q * A_ref * cd_axial
    assert math.isclose(aero_zero.force.magnitude(), expected_f_drag, rel_tol=1e-6)


# ============================================================================
# 6. Orbital Decay in Upper Atmosphere
# ============================================================================


def test_orbital_decay_upper_atmosphere():
    """Simulate Low Earth Orbit satellite decay at 80 km altitude using RK4 integration.

    Requirements:
        - Compare satellite in vacuum vs satellite in upper atmosphere (h = 80 km).
        - Propagate over two full revolutions (t_end ~= 10,312 s).
        - Verify progressive loss of specific orbital energy in the atmosphere:
          E_drag < E_vacuum and E_drag < E_initial.
        - Verify reduction of periapsis altitude: h_p_drag < h_p_vacuum.
        - Verify reduction of semi-major axis: a_drag < a_vacuum.
        - Vacuum orbit conserves energy and semi-major axis.
    """
    earth = get_earth()
    h0 = 80000.0  # 80 km initial circular altitude
    r0 = earth.radius + h0
    v_circ = math.sqrt(earth.mu / r0)
    period = 2.0 * math.pi * math.sqrt(r0**3 / earth.mu)

    # Satellite physical parameters
    mass = 1000.0  # 1000 kg
    area = 0.00005  # small cross-section for controlled multi-orbit decay
    cd = 1.0

    state0 = make_state(x=r0, y=0.0, vx=0.0, vy=v_circ, theta=math.pi / 2.0, omega=0.0)

    # 1. Vacuum Derivative Function (Pure Gravity)
    def vacuum_derivatives(_t: float, s: np.ndarray) -> np.ndarray:
        current_pos = Vector2D(s[IDX_X], s[IDX_Y])
        g_acc = earth.gravitational_acceleration_at(current_pos)
        return np.array(
            [s[IDX_VX], s[IDX_VY], g_acc.x, g_acc.y, 0.0, 0.0],
            dtype=np.float64,
        )

    # 2. Atmospheric Derivative Function (Gravity + Aerodynamic Drag)
    def aero_derivatives(_t: float, s: np.ndarray) -> np.ndarray:
        current_pos = Vector2D(s[IDX_X], s[IDX_Y])
        current_vel = Vector2D(s[IDX_VX], s[IDX_VY])
        g_acc = earth.gravitational_acceleration_at(current_pos)

        # Spacecraft flies prograde along its velocity vector
        aero = calculate_aero_forces(
            craft_pos=current_pos,
            craft_vel=current_vel,
            attitude_angle=current_vel.angle(),
            body=earth,
            reference_area=area,
            cd_axial=cd,
            cd_transverse=cd,
        )
        drag_acc = aero.force / mass
        total_acc = g_acc + drag_acc
        return np.array(
            [s[IDX_VX], s[IDX_VY], total_acc.x, total_acc.y, 0.0, 0.0],
            dtype=np.float64,
        )

    # Integrate both cases over 2 full orbital periods with dt = 5.0 s
    t_end = 2.0 * period
    dt = 5.0

    _times_vac, states_vac = integrate_trajectory(
        rk4_step,
        vacuum_derivatives,
        state0,
        (0.0, t_end),
        dt=dt,
    )

    _times_aero, states_aero = integrate_trajectory(
        rk4_step,
        aero_derivatives,
        state0,
        (0.0, t_end),
        dt=dt,
    )

    # Orbital elements analysis
    pos_vac = Vector2D(states_vac[-1, IDX_X], states_vac[-1, IDX_Y])
    vel_vac = Vector2D(states_vac[-1, IDX_VX], states_vac[-1, IDX_VY])
    orbit_vac = calculate_orbit(pos_vac, vel_vac, earth)

    pos_aero = Vector2D(states_aero[-1, IDX_X], states_aero[-1, IDX_Y])
    vel_aero = Vector2D(states_aero[-1, IDX_VX], states_aero[-1, IDX_VY])
    orbit_aero = calculate_orbit(pos_aero, vel_aero, earth)

    initial_energy = -earth.mu / (2.0 * r0)

    # Vacuum conservation
    assert math.isclose(orbit_vac.specific_orbital_energy, initial_energy, rel_tol=1e-5)
    assert math.isclose(orbit_vac.periapsis_altitude, h0, rel_tol=1e-4)
    assert math.isclose(orbit_vac.semi_major_axis, r0, rel_tol=1e-4)

    # Atmospheric orbital decay assertions:
    # 1. Total specific orbital energy decreases (becomes more negative)
    assert orbit_aero.specific_orbital_energy < initial_energy
    assert orbit_aero.specific_orbital_energy < orbit_vac.specific_orbital_energy
    energy_loss = initial_energy - orbit_aero.specific_orbital_energy
    assert energy_loss > 5000.0, f"Orbital energy loss {energy_loss:.1f} J/kg is too small"

    # 2. Periapsis altitude decreases compared to initial and vacuum
    assert orbit_aero.periapsis_altitude < h0
    assert orbit_aero.periapsis_altitude < orbit_vac.periapsis_altitude
    delta_hp = h0 - orbit_aero.periapsis_altitude
    assert delta_hp > 1000.0, f"Periapsis drop {delta_hp:.1f} m is too small"

    # 3. Semi-major axis decreases
    assert orbit_aero.semi_major_axis < orbit_vac.semi_major_axis
    delta_a = r0 - orbit_aero.semi_major_axis
    assert delta_a > 1000.0, f"Semi-major axis drop {delta_a:.1f} m is too small"


# ============================================================================
# 7. Helper Functions & Edge Cases
# ============================================================================


def test_cross_flow_drag_scaling():
    """Verify trigonometric drag coefficient scaling between axial and transverse values."""
    cd_ax = 0.3
    cd_tr = 1.2
    ref_area = 2.0
    q = 1000.0
    v_rel = Vector2D(100.0, 0.0)

    earth = get_earth()
    pos = Vector2D(earth.radius + 1000.0, 0.0)
    wind_vel = earth.wind_velocity_at(pos)
    craft_vel = wind_vel + v_rel

    # alpha = 0 -> cd = cd_ax
    aero_0 = calculate_aero_forces(
        craft_pos=pos,
        craft_vel=craft_vel,
        attitude_angle=0.0,
        body=earth,
        reference_area=ref_area,
        cd_axial=cd_ax,
        cd_transverse=cd_tr,
    )
    expected_f0 = -v_rel.normalized() * (aero_0.dynamic_pressure * ref_area * cd_ax)
    assert aero_0.force.is_close(expected_f0, rtol=1e-6)

    # alpha = 90 deg -> cd = cd_tr
    aero_90 = calculate_aero_forces(
        craft_pos=pos,
        craft_vel=craft_vel,
        attitude_angle=math.pi / 2.0,
        body=earth,
        reference_area=ref_area,
        cd_axial=cd_ax,
        cd_transverse=cd_tr,
    )
    expected_f90 = -v_rel.normalized() * (aero_90.dynamic_pressure * ref_area * cd_tr)
    assert aero_90.force.is_close(expected_f90, rtol=1e-6)

    # alpha = 45 deg -> axial force + cross-flow normal force (Allen-Perkins)
    aero_45 = calculate_aero_forces(
        craft_pos=pos,
        craft_vel=craft_vel,
        attitude_angle=math.pi / 4.0,
        body=earth,
        reference_area=ref_area,
        cd_axial=cd_ax,
        cd_transverse=cd_tr,
    )
    u_45 = Vector2D(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))
    n_45 = Vector2D(-math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))
    q_45 = aero_45.dynamic_pressure
    f_ax = -u_45 * (q_45 * ref_area * cd_ax * 0.5)
    f_tr = n_45 * (q_45 * ref_area * cd_tr * 0.5)
    expected_f45 = f_ax + f_tr
    assert aero_45.force.is_close(expected_f45, rtol=1e-6)


def test_lift_force_and_coefficient():
    """Verify lift force and lift coefficient functions."""
    # Lift coefficient: C_L = 0.5 * cl_alpha * sin(2*alpha), dC_L/d(alpha)|_0 = cl_alpha
    cl_slope = 2.0 * math.pi
    assert math.isclose(lift_coefficient(0.0, cl_slope), 0.0, abs_tol=1e-12)
    assert math.isclose(lift_coefficient(math.pi / 4.0, cl_slope), 0.5 * cl_slope, rel_tol=1e-12)
    assert math.isclose(lift_coefficient(-math.pi / 4.0, cl_slope), -0.5 * cl_slope, rel_tol=1e-12)
    assert math.isclose(lift_coefficient(math.pi / 2.0, cl_slope), 0.0, abs_tol=1e-12)

    # Numerical slope at alpha = 0 matches cl_slope
    d_alpha = 1e-6
    num_slope = (lift_coefficient(d_alpha, cl_slope) - lift_coefficient(-d_alpha, cl_slope)) / (2.0 * d_alpha)
    assert math.isclose(num_slope, cl_slope, rel_tol=1e-6)

    # Lift force perpendicular to flow
    q = 500.0
    area = 2.0
    cl = 0.5
    v_rel = Vector2D(200.0, 0.0)
    f_lift = lift_force(q, area, cl, v_rel)

    # Flow is +X, normal perpendicular is +Y
    assert math.isclose(f_lift.x, 0.0, abs_tol=1e-12)
    assert math.isclose(f_lift.y, q * area * cl, rel_tol=1e-12)

    # Zero velocity safety
    assert lift_force(q, area, cl, Vector2D(0.0, 0.0)) == Vector2D(0.0, 0.0)
    assert drag_force(q, area, 0.3, Vector2D(0.0, 0.0)) == Vector2D(0.0, 0.0)


def test_aerodynamic_vacuum_and_moon():
    """Verify that aerodynamic forces are identically zero in vacuum and on airless bodies."""
    earth = get_earth()
    moon = get_moon()

    pos_space = Vector2D(earth.radius + 200000.0, 0.0)
    vel = Vector2D(0.0, 7500.0)

    # 1. Earth outer space (200 km)
    aero_space = calculate_aero_forces(
        craft_pos=pos_space,
        craft_vel=vel,
        attitude_angle=0.0,
        body=earth,
        reference_area=5.0,
    )
    assert aero_space.force == Vector2D(0.0, 0.0)
    assert aero_space.torque == 0.0
    assert aero_space.dynamic_pressure == 0.0
    assert aero_space.mach == 0.0
    assert aero_space.angle_of_attack == 0.0

    # 2. Moon (no atmosphere)
    pos_moon = Vector2D(moon.radius + 1000.0, 0.0)
    aero_moon = calculate_aero_forces(
        craft_pos=pos_moon,
        craft_vel=vel,
        attitude_angle=0.0,
        body=moon,
        reference_area=5.0,
    )
    assert aero_moon.force == Vector2D(0.0, 0.0)
    assert aero_moon.torque == 0.0
    assert aero_moon.dynamic_pressure == 0.0


def test_celestial_presets_with_atmosphere():
    """Verify preset integrity and independence with added atmosphere models."""
    earth1 = get_earth()
    earth2 = get_earth()
    assert earth1.atmosphere is not None
    assert earth2.atmosphere is not None
    assert earth1.atmosphere is not earth2.atmosphere  # fresh independent instance

    assert EARTH.atmosphere is not None
    assert MOON.atmosphere is None

    moon1 = get_moon()
    assert moon1.atmosphere is None
