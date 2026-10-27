"""Unit test suite verifying celestial mechanics, gravitation, and Keplerian orbital extraction."""

from __future__ import annotations

import math
import warnings
import numpy as np
import pytest

from core.constants import G, g0
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
from physics.celestial import (
    EARTH,
    MOON,
    CelestialBody,
    CollisionWarning,
    get_earth,
    get_moon,
)
from physics.orbit import Orbit, calculate_orbit


# ============================================================================
# 1. Circular Orbit Test
# ============================================================================

def test_circular_orbit_leo_200km():
    """Verify Keplerian orbital parameter extraction for a 200 km circular LEO orbit.

    Requirements:
        - v_circ = sqrt(mu / r)
        - e ~= 0.0
        - r_a ~= r_p ~= r
        - flight_path_angle gamma ~= 0.0
    """
    alt = 200_000.0  # 200 km
    r_target = EARTH.radius + alt
    v_circ = math.sqrt(EARTH.mu / r_target)

    # Spacecraft at (r, 0) flying tangentially in +Y direction
    pos = Vector2D(r_target, 0.0)
    vel = Vector2D(0.0, v_circ)

    orbit: Orbit = calculate_orbit(pos, vel, EARTH)

    # Eccentricity should be zero within machine precision
    assert math.isclose(orbit.eccentricity, 0.0, abs_tol=1e-7)
    assert orbit.is_circular

    # Semi-major axis equals orbital radius
    assert math.isclose(orbit.semi_major_axis, r_target, rel_tol=1e-7)

    # Periapsis and apoapsis radii equal orbital radius
    assert math.isclose(orbit.periapsis_radius, r_target, rel_tol=1e-7)
    assert math.isclose(orbit.apoapsis_radius, r_target, rel_tol=1e-7)

    # Altitudes equal 200 km
    assert math.isclose(orbit.periapsis_altitude, alt, rel_tol=1e-7)
    assert math.isclose(orbit.apoapsis_altitude, alt, rel_tol=1e-7)

    # Flight path angle is strictly zero for circular motion
    assert math.isclose(orbit.flight_path_angle, 0.0, abs_tol=1e-7)

    # Orbital period matches T = 2*pi*sqrt(r^3 / mu)
    expected_period = 2.0 * math.pi * math.sqrt(r_target**3 / EARTH.mu)
    assert orbit.orbital_period is not None
    assert math.isclose(orbit.orbital_period, expected_period, rel_tol=1e-7)

    # Specific orbital energy epsilon = -mu / (2*a)
    expected_energy = -EARTH.mu / (2.0 * r_target)
    assert math.isclose(orbit.specific_orbital_energy, expected_energy, rel_tol=1e-7)


# ============================================================================
# 2. Closed Orbit Propagation Test
# ============================================================================

def test_closed_orbit_propagation_rk4():
    """Integrate a circular LEO orbit over one full period T using RK4.

    Requirements:
        - Final position and velocity return to initial conditions with < 0.05% error.
        - Total specific mechanical energy is strictly conserved.
    """
    alt = 200_000.0
    r0 = EARTH.radius + alt
    v_circ = math.sqrt(EARTH.mu / r0)

    pos0 = Vector2D(r0, 0.0)
    vel0 = Vector2D(0.0, v_circ)

    initial_orbit = calculate_orbit(pos0, vel0, EARTH)
    period = initial_orbit.orbital_period
    assert period is not None

    # State vector: [x, y, vx, vy, theta, omega]
    state0 = make_state(x=r0, y=0.0, vx=0.0, vy=v_circ, theta=0.0, omega=0.0)

    def gravitational_derivatives(_t: float, s: np.ndarray) -> np.ndarray:
        current_pos = Vector2D(s[IDX_X], s[IDX_Y])
        acc = EARTH.gravitational_acceleration_at(current_pos)
        return np.array([s[IDX_VX], s[IDX_VY], acc.x, acc.y, 0.0, 0.0], dtype=np.float64)

    # Integrate across one full revolution with dt = 2.0 s
    times, states = integrate_trajectory(
        rk4_step,
        gravitational_derivatives,
        state0,
        (0.0, period),
        dt=2.0,
    )

    final_pos = Vector2D(states[-1, IDX_X], states[-1, IDX_Y])
    final_vel = Vector2D(states[-1, IDX_VX], states[-1, IDX_VY])

    # Position relative error: ||r_final - r_initial|| / ||r_initial||
    pos_error_rel = (final_pos - pos0).magnitude() / r0
    assert pos_error_rel < 0.0005, f"Position error {pos_error_rel * 100:.4f}% exceeds 0.05%"

    # Velocity relative error: ||v_final - v_initial|| / ||v_initial||
    vel_error_rel = (final_vel - vel0).magnitude() / v_circ
    assert vel_error_rel < 0.0005, f"Velocity error {vel_error_rel * 100:.4f}% exceeds 0.05%"

    # Energy conservation check
    e_initial = 0.5 * (v_circ**2) - EARTH.mu / r0
    e_final = 0.5 * final_vel.magnitude_squared() - EARTH.mu / final_pos.magnitude()
    energy_error_rel = abs(e_final - e_initial) / abs(e_initial)
    assert energy_error_rel < 1e-5, f"Energy drift {energy_error_rel * 100:.6f}% is too high"


# ============================================================================
# 3. Kepler's 3rd Law Test
# ============================================================================

def test_keplers_third_law():
    """Verify Kepler's 3rd Law (T^2 / a^3 = 4*pi^2 / mu) for two distinct elliptical orbits.

    Orbit 1: Low-altitude elliptical orbit (hp = 200 km, ha = 1000 km)
    Orbit 2: High-altitude eccentric orbit (hp = 500 km, ha = 10,000 km)
    """
    # Orbit 1 setup
    rp1 = EARTH.radius + 200_000.0
    ra1 = EARTH.radius + 1_000_000.0
    a1 = (rp1 + ra1) / 2.0
    vp1 = math.sqrt(EARTH.mu * (2.0 / rp1 - 1.0 / a1))
    orb1 = calculate_orbit(Vector2D(rp1, 0.0), Vector2D(0.0, vp1), EARTH)

    # Orbit 2 setup
    rp2 = EARTH.radius + 500_000.0
    ra2 = EARTH.radius + 10_000_000.0
    a2 = (rp2 + ra2) / 2.0
    vp2 = math.sqrt(EARTH.mu * (2.0 / rp2 - 1.0 / a2))
    orb2 = calculate_orbit(Vector2D(rp2, 0.0), Vector2D(0.0, vp2), EARTH)

    assert orb1.orbital_period is not None
    assert orb2.orbital_period is not None

    ratio1 = (orb1.orbital_period**2) / (orb1.semi_major_axis**3)
    ratio2 = (orb2.orbital_period**2) / (orb2.semi_major_axis**3)
    expected_ratio = (4.0 * (math.pi**2)) / EARTH.mu

    # Both ratios must be identical and match the theoretical central body constant
    assert math.isclose(ratio1, ratio2, rel_tol=1e-11)
    assert math.isclose(ratio1, expected_ratio, rel_tol=1e-11)
    assert math.isclose(ratio2, expected_ratio, rel_tol=1e-11)


# ============================================================================
# 4. Hyperbolic Trajectory Test
# ============================================================================

def test_hyperbolic_trajectory():
    """Verify hyperbolic orbit parameters when velocity exceeds escape velocity.

    Requirements:
        - v > v_esc = sqrt(2*mu / r)
        - e > 1.0
        - a < 0.0
        - apoapsis altitude is safely handled as float('inf') without raising exceptions
        - orbital_period is None
    """
    alt = 500_000.0
    r = EARTH.radius + alt
    v_esc = math.sqrt(2.0 * EARTH.mu / r)
    v_hyp = 1.25 * v_esc  # 25% above escape speed

    pos = Vector2D(r, 0.0)
    vel = Vector2D(0.0, v_hyp)

    orb = calculate_orbit(pos, vel, EARTH)

    # Eccentricity must be strictly greater than 1
    assert orb.eccentricity > 1.0
    assert orb.is_hyperbolic

    # Semi-major axis is negative for hyperbolic trajectories
    assert orb.semi_major_axis < 0.0

    # Specific orbital energy is positive
    assert orb.specific_orbital_energy > 0.0

    # Periapsis radius is finite and positive
    assert orb.periapsis_radius > 0.0
    assert math.isclose(orb.periapsis_radius, r, rel_tol=1e-7)

    # Apoapsis radius and altitude are infinite
    assert math.isinf(orb.apoapsis_radius)
    assert orb.apoapsis_radius == float("inf")
    assert math.isinf(orb.apoapsis_altitude)
    assert orb.apoapsis_altitude == float("inf")

    # Unbound orbit has no period
    assert orb.orbital_period is None


# ============================================================================
# 5. Surface Collision Check
# ============================================================================

def test_surface_collision_check():
    """Verify that positions below planetary radius trigger collision warnings and collision flags."""
    # 1. Position strictly below surface (depth = 50 km)
    subsurface_pos = Vector2D(EARTH.radius - 50_000.0, 0.0)

    # Verify is_colliding triggers a CollisionWarning (subclass of UserWarning)
    with pytest.warns(CollisionWarning) as warning_records:
        is_hit = EARTH.is_colliding(subsurface_pos)

    assert is_hit is True
    assert len(warning_records) == 1
    assert "below planetary radius" in str(warning_records[0].message)

    # Altitude is negative
    assert math.isclose(EARTH.altitude(subsurface_pos), -50_000.0, abs_tol=1e-6)

    # calculate_orbit on a subsurface state also issues a CollisionWarning
    with pytest.warns(UserWarning):
        sub_orbit = calculate_orbit(subsurface_pos, Vector2D(0.0, 7000.0), EARTH)
    assert sub_orbit.periapsis_altitude < 0.0

    # 2. Position strictly above surface (altitude = 100 km)
    above_surface_pos = Vector2D(EARTH.radius + 100_000.0, 0.0)

    # Must NOT emit any warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert EARTH.is_colliding(above_surface_pos) is False

    assert math.isclose(EARTH.altitude(above_surface_pos), 100_000.0, abs_tol=1e-6)


# ============================================================================
# 6. Physical Constants & Celestial Body Presets Tests
# ============================================================================

def test_constants_and_celestial_presets():
    """Verify core physical constants and standard presets for Earth and Moon."""
    # Gravitational constant
    assert math.isclose(G, 6.67430e-11, rel_tol=1e-14)
    assert math.isclose(g0, 9.80665, rel_tol=1e-14)

    # Earth Preset properties
    earth = get_earth()
    assert earth.name == "Earth"
    assert math.isclose(earth.mass, 5.9722e24, rel_tol=1e-12)
    assert math.isclose(earth.radius, 6.371e6, rel_tol=1e-12)
    assert math.isclose(earth.rotation_period, 86164.0, rel_tol=1e-12)

    # Standard gravitational parameter: mu = G * M
    expected_earth_mu = G * 5.9722e24
    assert math.isclose(earth.mu, expected_earth_mu, rel_tol=1e-14)
    assert math.isclose(earth.standard_gravitational_parameter, expected_earth_mu, rel_tol=1e-14)

    # Surface gravity: g_surf = mu / R^2
    expected_earth_g = expected_earth_mu / (6.371e6**2)
    assert math.isclose(earth.surface_gravity, expected_earth_g, rel_tol=1e-14)
    assert math.isclose(earth.g_surface, expected_earth_g, rel_tol=1e-14)

    # Moon Preset properties
    moon = get_moon()
    assert moon.name == "Moon"
    assert math.isclose(moon.mass, 7.342e22, rel_tol=1e-12)
    assert math.isclose(moon.radius, 1.7374e6, rel_tol=1e-12)
    assert moon.rotation_period == 0.0

    expected_moon_mu = G * 7.342e22
    assert math.isclose(moon.mu, expected_moon_mu, rel_tol=1e-14)
    expected_moon_g = expected_moon_mu / (1.7374e6**2)
    assert math.isclose(moon.surface_gravity, expected_moon_g, rel_tol=1e-14)


def test_sphere_of_influence():
    """Verify Moon Sphere of Influence (SOI) relative to Earth.

    Equation:
        r_SOI = a * (m_moon / M_earth)^(2/5)
    For a = 384,400 km, r_SOI is approx 66,169 km.
    """
    moon = get_moon()
    earth = get_earth()
    a_moon = 384_400_000.0  # 384,400 km

    r_soi = moon.soi_radius(earth, a_moon)
    expected_soi = a_moon * ((moon.mass / earth.mass) ** 0.4)

    assert math.isclose(r_soi, expected_soi, rel_tol=1e-12)
    assert math.isclose(r_soi, 66_169_125.3, abs_tol=1e4)

    # Test SOI aliases and keyword arguments
    assert moon.sphere_of_influence(parent=earth, semi_major_axis=a_moon) == r_soi
    assert moon.sphere_of_influence_radius(parent_body=earth, a=a_moon) == r_soi
    assert moon.soi(parent=earth.mass, semi_major_axis=a_moon) == r_soi


def test_gravitational_acceleration_field():
    """Verify Newtonian inverse-square acceleration field and zero-division handling."""
    earth = get_earth()
    r1 = 7_000_000.0
    r2 = 14_000_000.0  # Double the distance

    pos1 = Vector2D(r1, 0.0)
    pos2 = Vector2D(r2, 0.0)

    acc1 = earth.gravitational_acceleration_at(pos1)
    acc2 = earth.gravitational_acceleration_at(pos2)

    # Acceleration points toward the planet center (-X direction)
    assert acc1.x < 0.0
    assert acc1.y == 0.0
    assert math.isclose(acc1.magnitude(), earth.mu / (r1**2), rel_tol=1e-12)

    # Inverse-square dropoff: acc(2*r) = acc(r) / 4
    assert math.isclose(acc1.magnitude() / acc2.magnitude(), 4.0, rel_tol=1e-12)

    # Center singularity
    with pytest.raises(ZeroDivisionError):
        earth.gravitational_acceleration_at(Vector2D(0.0, 0.0))


# ============================================================================
# 7. Parabolic Orbit & True Anomaly Verification
# ============================================================================

def test_parabolic_orbit():
    """Verify parabolic trajectory at exact escape speed: v = sqrt(2*mu / r)."""
    r0 = EARTH.radius + 400_000.0
    v_esc = math.sqrt(2.0 * EARTH.mu / r0)

    pos = Vector2D(r0, 0.0)
    vel = Vector2D(0.0, v_esc)

    orbit = calculate_orbit(pos, vel, EARTH)

    assert math.isclose(orbit.eccentricity, 1.0, abs_tol=1e-7)
    assert orbit.is_parabolic
    assert math.isinf(orbit.semi_major_axis)
    assert math.isinf(orbit.apoapsis_radius)
    assert math.isinf(orbit.apoapsis_altitude)
    assert math.isclose(orbit.periapsis_radius, r0, rel_tol=1e-7)
    assert orbit.orbital_period is None


def test_elliptic_orbit_quadrature_anomalies():
    """Verify true anomaly at periapsis (0), semi-latus rectum (pi/2), and apoapsis (pi)."""
    rp = EARTH.radius + 300_000.0
    ra = EARTH.radius + 1_500_000.0
    a = (rp + ra) / 2.0
    e = (ra - rp) / (ra + rp)
    p = a * (1.0 - e**2)
    h = math.sqrt(EARTH.mu * p)

    # 1. At periapsis
    vp = math.sqrt(EARTH.mu * (2.0 / rp - 1.0 / a))
    orb_peri = calculate_orbit(Vector2D(rp, 0.0), Vector2D(0.0, vp), EARTH)
    assert math.isclose(orb_peri.true_anomaly, 0.0, abs_tol=1e-7)
    assert math.isclose(orb_peri.eccentricity, e, rel_tol=1e-7)

    # 2. At true anomaly nu = pi / 2 (semi-latus rectum p)
    # Unit radial is +Y, transverse velocity is in -X
    vr = (EARTH.mu / h) * e
    vt = h / p
    orb_quad = calculate_orbit(Vector2D(0.0, p), Vector2D(-vt, vr), EARTH)
    assert math.isclose(orb_quad.true_anomaly, math.pi / 2.0, abs_tol=1e-7)

    # 3. At apoapsis
    va = math.sqrt(EARTH.mu * (2.0 / ra - 1.0 / a))
    orb_apo = calculate_orbit(Vector2D(-ra, 0.0), Vector2D(0.0, -va), EARTH)
    assert math.isclose(orb_apo.true_anomaly, math.pi, abs_tol=1e-7)


def test_retrograde_and_moving_primary_body():
    """Verify retrograde (clockwise) orbit and orbit around a moving celestial body."""
    # Moving planet at pos=(1e6, 2e6) with vel=(500, -200)
    planet_pos = Vector2D(1_000_000.0, 2_000_000.0)
    planet_vel = Vector2D(500.0, -200.0)
    moving_earth = CelestialBody(
        name="MovingEarth",
        mass=5.9722e24,
        radius=6.371e6,
        position=planet_pos,
        velocity=planet_vel,
    )

    r_orbit = moving_earth.radius + 300_000.0
    v_circ = math.sqrt(moving_earth.mu / r_orbit)

    # Retrograde (CW): relative velocity is (0, -v_circ)
    craft_pos = planet_pos + Vector2D(r_orbit, 0.0)
    craft_vel = planet_vel + Vector2D(0.0, -v_circ)

    orb = calculate_orbit(craft_pos, craft_vel, moving_earth)

    # Angular momentum should be negative (CW)
    assert orb.specific_angular_momentum < 0.0
    assert math.isclose(orb.eccentricity, 0.0, abs_tol=1e-7)
    assert math.isclose(orb.semi_major_axis, r_orbit, rel_tol=1e-7)
    assert math.isclose(orb.flight_path_angle, 0.0, abs_tol=1e-7)


def test_rectilinear_bound_trajectories():
    """Verify that radial/rectilinear trajectories (h = 0) with energy < 0 are classified as bound ellipses."""
    r0 = EARTH.radius + 1_000_000.0  # 1000 km altitude

    # Case A: Dropped from rest (v = 0)
    pos_rest = Vector2D(0.0, r0)
    vel_rest = Vector2D(0.0, 0.0)
    orb_rest = calculate_orbit(pos_rest, vel_rest, EARTH)

    assert math.isclose(orb_rest.eccentricity, 1.0, abs_tol=1e-7)
    assert orb_rest.is_closed is True
    assert orb_rest.is_parabolic is False
    assert not math.isinf(orb_rest.semi_major_axis)
    assert math.isclose(orb_rest.semi_major_axis, r0 / 2.0, rel_tol=1e-7)
    assert math.isclose(orb_rest.apoapsis_radius, r0, rel_tol=1e-7)
    assert math.isclose(orb_rest.periapsis_radius, 0.0, abs_tol=1e-7)
    assert orb_rest.orbital_period is not None
    assert orb_rest.orbital_period > 0.0

    # Case B: Vertical ascent at 1000 m/s
    v_vert = 1000.0
    pos_ascent = Vector2D(0.0, r0)
    vel_ascent = Vector2D(0.0, v_vert)
    orb_ascent = calculate_orbit(pos_ascent, vel_ascent, EARTH)

    assert math.isclose(orb_ascent.eccentricity, 1.0, abs_tol=1e-7)
    assert orb_ascent.is_closed is True
    assert orb_ascent.is_parabolic is False
    assert not math.isinf(orb_ascent.semi_major_axis)
    assert orb_ascent.apoapsis_radius > r0
    assert math.isclose(orb_ascent.periapsis_radius, 0.0, abs_tol=1e-7)
    assert orb_ascent.orbital_period is not None


def test_propagate_kepler_parabolic_no_teleportation():
    """Verify propagate_kepler does not crash with ZeroDivisionError or teleport on parabolic orbits."""
    from physics.orbit import propagate_kepler

    r0 = EARTH.radius + 500_000.0
    v_esc = math.sqrt(2.0 * EARTH.mu / r0)
    pos = Vector2D(r0, 0.0)
    vel = Vector2D(0.0, v_esc)  # Exact escape speed

    # Propagate for 100 seconds
    pos_next, vel_next = propagate_kepler(pos, vel, EARTH, dt=100.0)

    # Spacecraft should move a reasonable distance (around ~1100 km), NOT 1300 AU (~2e14 m)!
    dist = pos_next.magnitude()
    assert r0 < dist < r0 + 2_000_000.0, f"Teleportation detected: dist={dist}"

    # Velocity magnitude should decrease according to 2-body gravity
    v_next = vel_next.magnitude()
    assert v_next < v_esc
    assert math.isclose(0.5 * v_next**2, EARTH.mu / dist, rel_tol=1e-6)


def test_propagate_kepler_rectilinear_elliptical():
    """Verify propagate_kepler preserves velocity and position for vertical rectilinear drop/climb."""
    from physics.orbit import propagate_kepler

    r0 = EARTH.radius + 1_000_000.0
    pos = Vector2D(0.0, r0)
    vel = Vector2D(0.0, 500.0)  # Climbing vertically

    pos_next, vel_next = propagate_kepler(pos, vel, EARTH, dt=10.0)

    # Should still be strictly along Y-axis with non-zero vertical velocity
    assert math.isclose(pos_next.x, 0.0, abs_tol=1e-7)
    assert math.isclose(vel_next.x, 0.0, abs_tol=1e-7)
    assert pos_next.y > r0
    assert 0.0 < vel_next.y < 500.0  # Decelerated by gravity


