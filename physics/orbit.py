"""Keplerian orbital mechanics and state vector decomposition in 2D space.

Extracts classical orbital elements (Apoapsis, Periapsis, Semi-Major Axis,
Eccentricity, True Anomaly, Orbital Period, Specific Energy, Flight Path Angle)
from Cartesian relative position and velocity state vectors (x, y, vx, vy).
All calculations enforce 64-bit floating-point precision (np.float64).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.vector2d import Vector2D
from physics.celestial import CelestialBody


@dataclass(frozen=True, slots=True)
class Orbit:
    """Keplerian orbital elements and conservation invariants for planar 2D orbits.

    Attributes
    ----------
    semi_major_axis : float
        Semi-major axis a [m]. Positive for elliptic orbits, negative for hyperbolic,
        inf for parabolic.
    eccentricity : float
        Orbital eccentricity e (scalar magnitude of the eccentricity vector).
    periapsis_radius : float
        Distance from primary focal center to periapsis r_p [m].
    apoapsis_radius : float
        Distance from primary focal center to apoapsis r_a [m] (float('inf') if parabolic or hyperbolic).
    periapsis_altitude : float
        Geometric altitude of periapsis above the primary body surface h_p = r_p - R_body [m].
    apoapsis_altitude : float
        Geometric altitude of apoapsis above the primary body surface h_a = r_a - R_body [m]
        (float('inf') if parabolic or hyperbolic).
    specific_orbital_energy : float
        Specific mechanical orbital energy epsilon = v^2/2 - mu/r [J/kg or m^2/s^2].
    specific_angular_momentum : float
        Specific angular momentum scalar h = r_x * v_y - r_y * v_x [m^2/s].
    true_anomaly : float
        True anomaly angle nu in radians, defined in [0, 2*pi).
    orbital_period : Optional[float]
        Orbital period T [s] for closed orbits (e < 1.0); None for open orbits (e >= 1.0).
    flight_path_angle : float
        Angle gamma between the velocity vector and the local horizontal in radians [-pi/2, +pi/2].
    eccentricity_vector : Vector2D
        Laplace-Runge-Lenz eccentricity vector pointing toward periapsis.
    """

    semi_major_axis: float
    eccentricity: float
    periapsis_radius: float
    apoapsis_radius: float
    periapsis_altitude: float
    apoapsis_altitude: float
    specific_orbital_energy: float
    specific_angular_momentum: float
    true_anomaly: float
    orbital_period: Optional[float]
    flight_path_angle: float
    eccentricity_vector: Vector2D

    # --- Property Aliases ---

    @property
    def a(self) -> float:
        """Semi-major axis [m]."""
        return self.semi_major_axis

    @property
    def e(self) -> float:
        """Eccentricity scalar."""
        return self.eccentricity

    @property
    def r_p(self) -> float:
        """Periapsis radius [m]."""
        return self.periapsis_radius

    @property
    def r_a(self) -> float:
        """Apoapsis radius [m]."""
        return self.apoapsis_radius

    @property
    def h_p(self) -> float:
        """Periapsis altitude [m]."""
        return self.periapsis_altitude

    @property
    def h_a(self) -> float:
        """Apoapsis altitude [m]."""
        return self.apoapsis_altitude

    @property
    def energy(self) -> float:
        """Specific orbital energy [J/kg]."""
        return self.specific_orbital_energy

    @property
    def h(self) -> float:
        """Specific angular momentum [m^2/s]."""
        return self.specific_angular_momentum

    @property
    def nu(self) -> float:
        """True anomaly [rad]."""
        return self.true_anomaly

    @property
    def gamma(self) -> float:
        """Flight path angle [rad]."""
        return self.flight_path_angle

    @property
    def period(self) -> Optional[float]:
        """Orbital period [s]."""
        return self.orbital_period

    @property
    def is_closed(self) -> bool:
        """Determine if orbit is bound/closed (finite orbital period)."""
        return self.orbital_period is not None

    @property
    def is_circular(self) -> bool:
        """Determine if orbit is approximately circular (e < 1e-4)."""
        return self.eccentricity < 1e-4

    @property
    def is_hyperbolic(self) -> bool:
        """Determine if orbit is hyperbolic (energy > 0 and not parabolic)."""
        return self.specific_orbital_energy > 0.0 and not self.is_parabolic

    @property
    def is_parabolic(self) -> bool:
        """Determine if orbit is parabolic (infinite semi-major axis)."""
        return math.isinf(self.semi_major_axis)


def calculate_orbit(pos: Vector2D, vel: Vector2D, body: CelestialBody) -> Orbit:
    """Calculate 2D Keplerian orbital elements from state vectors relative to a celestial body.

    Computes relative position r = pos - body.position and relative velocity
    v = vel - body.velocity, then applies standard two-body orbital mechanics:
        epsilon = v^2 / 2 - mu / r
        a = - mu / (2 * epsilon)
        e_vec = (1 / mu) * [ (v^2 - mu / r) * r - (r . v) * v ]
        e = ||e_vec||
        h = r_x * v_y - r_y * v_x

    Parameters
    ----------
    pos : Vector2D
        Inertial position vector of the orbiting spacecraft [m].
    vel : Vector2D
        Inertial velocity vector of the orbiting spacecraft [m/s].
    body : CelestialBody
        Primary celestial body providing the central gravitational field.

    Returns
    -------
    Orbit
        Immutable dataclass containing all derived Keplerian orbital parameters.

    Raises
    ------
    ValueError
        If pos coincides with the body center (r = 0) or body mu is non-positive.
    """
    if not isinstance(pos, Vector2D):
        pos = Vector2D.from_tuple(pos)
    if not isinstance(vel, Vector2D):
        vel = Vector2D.from_tuple(vel)

    # 1. Relative state vectors
    r_vec = pos - body.position
    v_vec = vel - body.velocity
    r = r_vec.magnitude()

    if math.isclose(r, 0.0, abs_tol=1e-15):
        raise ValueError(
            f"Orbit undefined: spacecraft position coincides with {body.name} center of mass (r=0)."
        )

    # Check for surface collision and trigger warning if below surface
    body.is_colliding(pos)

    # 2. Gravitational parameter
    mu = float(np.float64(body.mu))
    if mu <= 0.0:
        raise ValueError(f"Body gravitational parameter mu must be positive, got {mu}")

    # 3. Specific angular momentum (2D scalar cross product: r_x * v_y - r_y * v_x)
    h = float(np.float64(r_vec.cross(v_vec)))

    # 4. Specific orbital energy: epsilon = v^2 / 2 - mu / r
    v_sq = float(np.float64(v_vec.magnitude_squared()))
    energy = float(np.float64(0.5 * v_sq - mu / r))

    # 5. Eccentricity vector (Laplace-Runge-Lenz vector)
    # e_vec = ((v^2 - mu/r) * r - (r . v) * v) / mu
    r_dot_v = float(np.float64(r_vec.dot(v_vec)))
    coeff_r = v_sq - (mu / r)
    term_r = r_vec * coeff_r
    term_v = v_vec * r_dot_v
    e_vec = (term_r - term_v) / mu
    e = float(np.float64(e_vec.magnitude()))

    # Robust handling of circular orbits without numerical jitter
    if e < 1e-14:
        e = 0.0
        e_vec = Vector2D(0.0, 0.0)

    # 6. Conic section classification and apsis calculation
    # Parabolic orbit: specific kinetic energy ~= specific gravitational potential energy
    # Dimensionless criterion: |epsilon| / (mu / r) < 1e-7 <=> math.isclose(0.5 * v_sq, mu / r, rel_tol=1e-7)
    u_pot = mu / r
    if math.isclose(0.5 * v_sq, u_pot, rel_tol=1e-7):
        e = 1.0
        semi_major_axis = float("inf")
        # Semi-latus rectum p = h^2 / mu, r_p = p / 2 = h^2 / (2 * mu)
        # For radial parabolic escape (h = 0), r_p = 0.0 naturally
        r_p = float(np.float64((h * h) / (2.0 * mu))) if mu > 0.0 else 0.0
        r_a = float("inf")
        h_p = float(np.float64(r_p - body.radius))
        h_a = float("inf")
        orbital_period = None

    # Hyperbolic trajectory: energy > 0 (v > v_esc)
    elif energy > 0.0:
        # Standard convention: semi-major axis for hyperbolic trajectory is negative
        semi_major_axis = float(np.float64(-mu / (2.0 * energy)))
        # Periapsis distance r_p = |a| * (e - 1) = a * (1 - e)
        # For radial hyperbolic trajectory (h = 0, e = 1.0), max(0.0, e - 1.0) ensures r_p = 0.0
        r_p = float(np.float64(abs(semi_major_axis) * max(0.0, e - 1.0)))
        r_a = float("inf")
        h_p = float(np.float64(r_p - body.radius))
        h_a = float("inf")
        orbital_period = None

    # Elliptic / Circular orbit: energy < 0 (v < v_esc)
    # Includes bound rectilinear trajectories (h = 0, e = 1.0):
    #   a = -mu / (2 * energy) > 0
    #   r_p = a * (1 - e) = 0.0
    #   r_a = a * (1 + e) = 2 * a
    #   T = 2 * pi * sqrt(a^3 / mu)
    else:
        semi_major_axis = float(np.float64(-mu / (2.0 * energy)))
        # Guard against 1-ULP floating-point overshoot where e could be 1.0000000000000002
        r_p = float(np.float64(semi_major_axis * max(0.0, 1.0 - e)))
        r_a = float(np.float64(semi_major_axis * (1.0 + e)))
        h_p = float(np.float64(r_p - body.radius))
        h_a = float(np.float64(r_a - body.radius))
        # Kepler's third law: T = 2 * pi * sqrt(a^3 / mu)
        orbital_period = float(np.float64(2.0 * math.pi * math.sqrt(semi_major_axis**3 / mu)))

    # 7. True anomaly nu in [0, 2*pi)
    if e < 1e-12:
        # For circular orbits, periapsis is degenerate; measure from reference direction (+X axis)
        nu = float(np.float64(math.atan2(r_vec.y, r_vec.x) % (2.0 * math.pi)))
    else:
        # cos(nu) = (e_vec . r_vec) / (e * r)
        cos_nu = e_vec.dot(r_vec) / (e * r)
        cos_nu_clamped = max(-1.0, min(1.0, cos_nu))
        nu_val = math.acos(cos_nu_clamped)
        # If r . v < 0, spacecraft is moving from apoapsis toward periapsis (inbound)
        if r_dot_v < 0.0:
            nu_val = 2.0 * math.pi - nu_val
        nu = float(np.float64(nu_val % (2.0 * math.pi)))

    # 8. Flight path angle gamma: angle between velocity and local horizontal
    # tan(gamma) = (r . v) / |h|  -> gamma = atan2(r . v, |h|)
    abs_h = abs(h)
    if math.isclose(abs_h, 0.0, abs_tol=1e-15) and math.isclose(r_dot_v, 0.0, abs_tol=1e-15):
        flight_path_angle = 0.0
    else:
        flight_path_angle = float(np.float64(math.atan2(r_dot_v, abs_h)))

    return Orbit(
        semi_major_axis=semi_major_axis,
        eccentricity=e,
        periapsis_radius=r_p,
        apoapsis_radius=r_a,
        periapsis_altitude=h_p,
        apoapsis_altitude=h_a,
        specific_orbital_energy=energy,
        specific_angular_momentum=h,
        true_anomaly=nu,
        orbital_period=orbital_period,
        flight_path_angle=flight_path_angle,
        eccentricity_vector=e_vec,
    )


def solve_kepler_elliptic(
    mean_anomaly: float,
    eccentricity: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Solve Kepler's equation for elliptic/circular orbits (e < 1.0) via Newton-Raphson.

    Equation:
        f(E) = E - e * sin(E) - M = 0

    Parameters
    ----------
    mean_anomaly : float
        Mean anomaly M in radians.
    eccentricity : float
        Orbital eccentricity (0 <= e < 1.0).
    tol : float, default=1e-12
        Convergence tolerance on residual |f(E)|.
    max_iter : int, default=100
        Maximum Newton-Raphson iterations.

    Returns
    -------
    float
        Eccentric anomaly E in radians, wrapped into [0, 2*pi).
    """
    e = float(np.float64(eccentricity))
    if not (0.0 <= e < 1.0):
        raise ValueError(
            f"Eccentricity must satisfy 0 <= e < 1 for elliptic Kepler solver, got {e}"
        )

    M = float(np.float64(mean_anomaly)) % (2.0 * math.pi)

    if e < 1e-12:
        return M

    # Initial guess
    if e < 0.8:
        E = M
    else:
        sin_m = math.sin(M)
        E = math.pi if abs(M - math.pi) < 0.5 else M + e * sin_m

    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        if abs(f) < tol:
            break
        f_prime = 1.0 - e * math.cos(E)
        if abs(f_prime) < 1e-15:
            break
        delta = f / f_prime
        E -= delta
        if abs(delta) < tol and abs(f) < tol:
            break

    return float(np.float64(E % (2.0 * math.pi)))


def solve_kepler_hyperbolic(
    mean_anomaly: float,
    eccentricity: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Solve Kepler's equation for hyperbolic trajectories (e > 1.0) via Newton-Raphson.

    Equation:
        f(H) = e * sinh(H) - H - M_h = 0

    Parameters
    ----------
    mean_anomaly : float
        Hyperbolic mean anomaly M_h in radians.
    eccentricity : float
        Orbital eccentricity (e > 1.0).
    tol : float, default=1e-12
        Convergence tolerance on residual |f(H)|.
    max_iter : int, default=100
        Maximum Newton-Raphson iterations.

    Returns
    -------
    float
        Hyperbolic anomaly H in radians.
    """
    e = float(np.float64(eccentricity))
    if e <= 1.0:
        raise ValueError(
            f"Eccentricity must be strictly greater than 1 for hyperbolic solver, got {e}"
        )

    M_h = float(np.float64(mean_anomaly))

    # Initial guess
    if abs(M_h) < 1.0:
        H = M_h / (e - 1.0)
    elif M_h > 0.0:
        H = math.log(2.0 * M_h / e + 1.8)
    else:
        H = -math.log(-2.0 * M_h / e + 1.8)

    for _ in range(max_iter):
        f = e * math.sinh(H) - H - M_h
        if abs(f) < tol:
            break
        f_prime = e * math.cosh(H) - 1.0
        delta = f / f_prime
        H -= delta
        if abs(delta) < tol and abs(f) < tol:
            break

    return float(np.float64(H))


def propagate_kepler(
    pos: Vector2D,
    vel: Vector2D,
    body: CelestialBody,
    dt: float,
) -> tuple[Vector2D, Vector2D]:
    """Propagate Cartesian state vector (r, v) forward by dt under 2-body Newtonian gravity.

    Solves Kepler's equation analytically for elliptic and hyperbolic trajectories.
    Preserves specific orbital energy and specific angular momentum to floating-point
    precision.

    Parameters
    ----------
    pos : Vector2D
        Inertial position vector of the spacecraft [m].
    vel : Vector2D
        Inertial velocity vector of the spacecraft [m/s].
    body : CelestialBody
        Central celestial body providing the central gravitational field.
    dt : float
        Time step to propagate forward in seconds [s].

    Returns
    -------
    tuple[Vector2D, Vector2D]
        Propagated (position, velocity) vectors in the inertial frame [m], [m/s].

    Raises
    ------
    ValueError
        If pos coincides with body center or parabolic orbit e == 1.0 cannot be propagated.
    """
    if not isinstance(pos, Vector2D):
        pos = Vector2D.from_tuple(pos)
    if not isinstance(vel, Vector2D):
        vel = Vector2D.from_tuple(vel)

    dt_val = float(np.float64(dt))
    if math.isclose(dt_val, 0.0, abs_tol=1e-15):
        return (pos, vel)

    # 1. Relative state vectors
    r_vec = pos - body.position
    v_vec = vel - body.velocity
    r0 = r_vec.magnitude()

    if math.isclose(r0, 0.0, abs_tol=1e-15):
        raise ValueError("Spacecraft position coincides with central body center of mass (r=0).")

    mu = float(np.float64(body.mu))
    if mu <= 0.0:
        raise ValueError(f"Central body mu must be positive, got {mu}")

    # Specific angular momentum scalar h = r_x * v_y - r_y * v_x
    h_scalar = float(np.float64(r_vec.cross(v_vec)))
    sign_h = 1.0 if h_scalar >= 0.0 else -1.0

    # Specific mechanical energy: epsilon = v^2 / 2 - mu / r
    v_sq = float(np.float64(v_vec.magnitude_squared()))
    energy0 = float(np.float64(0.5 * v_sq - mu / r0))

    # Eccentricity vector
    r_dot_v = float(np.float64(r_vec.dot(v_vec)))
    e_vec = ((v_sq - mu / r0) * r_vec - r_dot_v * v_vec) / mu
    e = float(np.float64(e_vec.magnitude()))

    # Check for parabolic trajectory
    if math.isclose(energy0, 0.0, abs_tol=1e-12) or math.isclose(e, 1.0, abs_tol=1e-7):
        # Slightly bias to avoid singularity
        if energy0 < 0.0:
            e = 1.0 - 1e-8
        else:
            e = 1.0 + 1e-8

    # 2. Elliptic / Circular Trajectories (0 <= e < 1.0)
    if e < 1.0:
        semi_major_axis = float(np.float64(-mu / (2.0 * energy0)))
        p = semi_major_axis * (1.0 - e * e)
        n = math.sqrt(mu / (semi_major_axis**3))

        if e < 1e-12:
            e = 0.0
            p_hat = r_vec / r0
            q_hat = Vector2D(-p_hat.y, p_hat.x) * sign_h
            nu0 = 0.0
            e0 = 0.0
        else:
            p_hat = e_vec / e
            q_hat = Vector2D(-p_hat.y, p_hat.x) * sign_h
            cos_nu0 = max(-1.0, min(1.0, p_hat.dot(r_vec) / r0))
            nu0_val = math.acos(cos_nu0)
            if r_dot_v < 0.0:
                nu0_val = 2.0 * math.pi - nu0_val
            nu0 = nu0_val % (2.0 * math.pi)
            e0 = math.atan2(
                math.sqrt(max(0.0, 1.0 - e * e)) * math.sin(nu0),
                e + math.cos(nu0),
            ) % (2.0 * math.pi)

        # Initial and target mean anomalies
        m0 = e0 - e * math.sin(e0)
        m_target = (m0 + n * dt_val) % (2.0 * math.pi)

        # Solve Kepler's equation for target eccentric anomaly E
        e_target = solve_kepler_elliptic(m_target, e, tol=1e-13)

        # New true anomaly nu
        nu_target = math.atan2(
            math.sqrt(max(0.0, 1.0 - e * e)) * math.sin(e_target),
            math.cos(e_target) - e,
        ) % (2.0 * math.pi)

        # New orbital radius
        r_target = semi_major_axis * (1.0 - e * math.cos(e_target))

        # Reconstruct Cartesian state vectors in orbital plane
        r_rel = p_hat * (r_target * math.cos(nu_target)) + q_hat * (r_target * math.sin(nu_target))
        sqrt_mu_p = math.sqrt(mu / p) if p > 0.0 else 0.0
        v_rel = (p_hat * (-math.sin(nu_target)) + q_hat * (e + math.cos(nu_target))) * sqrt_mu_p

    # 3. Hyperbolic Trajectories (e > 1.0)
    else:
        semi_major_axis = float(np.float64(-mu / (2.0 * energy0)))
        abs_a = abs(semi_major_axis)
        p = abs_a * (e * e - 1.0)
        n_h = math.sqrt(mu / (abs_a**3))

        p_hat = e_vec / e
        q_hat = Vector2D(-p_hat.y, p_hat.x) * sign_h

        cos_nu0 = max(-1.0, min(1.0, p_hat.dot(r_vec) / r0))
        nu0_val = math.acos(cos_nu0)
        if r_dot_v < 0.0:
            nu0_val = -nu0_val

        # tanh(H_0 / 2) = sqrt((e - 1) / (e + 1)) * tan(nu_0 / 2)
        x = math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu0_val / 2.0)
        x_clamped = max(-0.999999999999, min(0.999999999999, x))
        h0 = 2.0 * math.atanh(x_clamped)

        # Initial and target hyperbolic mean anomalies
        m_h0 = e * math.sinh(h0) - h0
        m_h_target = m_h0 + n_h * dt_val

        # Solve Kepler's equation for hyperbolic anomaly H
        h_target = solve_kepler_hyperbolic(m_h_target, e, tol=1e-13)

        # New true anomaly nu
        y = math.sqrt((e + 1.0) / (e - 1.0)) * math.tanh(h_target / 2.0)
        nu_signed = 2.0 * math.atan(y)
        nu_target = nu_signed % (2.0 * math.pi)

        # New radius
        r_target = abs_a * (e * math.cosh(h_target) - 1.0)

        # Reconstruct Cartesian state vectors
        r_rel = p_hat * (r_target * math.cos(nu_target)) + q_hat * (r_target * math.sin(nu_target))
        sqrt_mu_p = math.sqrt(mu / p) if p > 0.0 else 0.0
        v_rel = (p_hat * (-math.sin(nu_target)) + q_hat * (e + math.cos(nu_target))) * sqrt_mu_p

    pos_next = body.position + r_rel
    vel_next = body.velocity + v_rel

    return pos_next, vel_next


__all__ = [
    "Orbit",
    "calculate_orbit",
    "propagate_kepler",
    "solve_kepler_elliptic",
    "solve_kepler_hyperbolic",
]
