"""Aerodynamic force, torque, and flow state calculations for planar rocket dynamics.

This module implements:
    - Relative flow velocity with planetary atmospheric co-rotation.
    - Dynamic pressure and Mach number computations.
    - Cross-flow / angle-of-attack dependent drag coefficients.
    - Aerodynamic restoring/destabilizing torque around the Center of Mass.
    - Slender-body lift forces and moments.

All calculations enforce 64-bit floating-point precision (np.float64 / IEEE 754 double).
SI Units:
    - Forces: Newtons [N]
    - Torques: Newton-meters [N m]
    - Dynamic Pressure: Pascals [Pa]
    - Angles: radians [rad]
    - Velocities: meters per second [m/s]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

from core.vector2d import Vector2D

if TYPE_CHECKING:
    from physics.celestial import CelestialBody

Scalar = Union[int, float, np.floating, np.integer]


@dataclass(frozen=True, slots=True)
class AeroState:
    """Aerodynamic forces, moments, and atmospheric flow state.

    Attributes
    ----------
    force : Vector2D
        Total aerodynamic force vector in the inertial frame [N].
    torque : float
        Aerodynamic torque around the center of mass [N m].
        Positive for counter-clockwise rotation, negative for clockwise.
    dynamic_pressure : float
        Dynamic pressure q = 0.5 * rho * v_rel^2 [Pa].
    mach : float
        Mach number M = v_rel / a.
    angle_of_attack : float
        Angle of attack alpha in radians [-pi, pi], measured from relative
        velocity vector to rocket heading vector.
    """

    force: Vector2D
    torque: float
    dynamic_pressure: float
    mach: float
    angle_of_attack: float

    def __iter__(self):
        """Allow sequence unpacking: force, torque, q, mach, alpha = aero_state."""
        yield self.force
        yield self.torque
        yield self.dynamic_pressure
        yield self.mach
        yield self.angle_of_attack

    def __getitem__(self, index: int):
        """Enable indexed access aero_state[0] -> force, etc."""
        return (
            self.force,
            self.torque,
            self.dynamic_pressure,
            self.mach,
            self.angle_of_attack,
        )[index]

    @property
    def q(self) -> float:
        """Alias for dynamic_pressure [Pa]."""
        return self.dynamic_pressure

    @property
    def aoa(self) -> float:
        """Alias for angle_of_attack [rad]."""
        return self.angle_of_attack

    @property
    def alpha(self) -> float:
        """Alias for angle_of_attack [rad]."""
        return self.angle_of_attack


def dynamic_pressure(density: float, rel_speed: float) -> float:
    """Compute dynamic pressure: q = 0.5 * rho * v_rel^2.

    Parameters
    ----------
    density : float
        Atmospheric density rho in kg/m^3.
    rel_speed : float
        Relative speed v_rel = ||v_craft - v_wind|| in m/s.

    Returns
    -------
    float
        Dynamic pressure q in Pascals [Pa] (64-bit float).
    """
    rho = float(np.float64(density))
    v = float(np.float64(rel_speed))
    if rho <= 0.0 or v <= 0.0:
        return 0.0
    return float(np.float64(0.5 * rho * v * v))


def mach_number(rel_speed: float, speed_of_sound: float) -> float:
    """Compute Mach number: M = v_rel / a.

    Parameters
    ----------
    rel_speed : float
        Relative flow speed v_rel in m/s.
    speed_of_sound : float
        Local speed of sound a in m/s.

    Returns
    -------
    float
        Dimensionless Mach number. Returns 0.0 if speed_of_sound <= 0.0.
    """
    v = float(np.float64(rel_speed))
    a = float(np.float64(speed_of_sound))
    if a <= 0.0 or v <= 0.0:
        return 0.0
    return float(np.float64(v / a))


def relative_velocity(craft_vel: Vector2D, wind_vel: Vector2D) -> Vector2D:
    """Compute flow velocity of the atmosphere relative to the spacecraft.

    Equation:
        v_rel = v_craft - v_wind

    Parameters
    ----------
    craft_vel : Vector2D
        Inertial velocity vector of the spacecraft [m/s].
    wind_vel : Vector2D
        Inertial velocity vector of the atmospheric wind [m/s].

    Returns
    -------
    Vector2D
        Relative velocity vector [m/s].
    """
    if not isinstance(craft_vel, Vector2D):
        craft_vel = Vector2D.from_tuple(craft_vel)
    if not isinstance(wind_vel, Vector2D):
        wind_vel = Vector2D.from_tuple(wind_vel)
    return craft_vel - wind_vel


def lift_coefficient(angle_of_attack: float, cl_alpha: float = 2.0 * math.pi) -> float:
    """Compute slender-body / thin-airfoil lift coefficient.

    Equation:
        C_L(alpha) = 0.5 * cl_alpha * sin(2 * alpha)
    Satisfies dC_L / d(alpha) = cl_alpha at alpha = 0.

    Parameters
    ----------
    angle_of_attack : float
        Angle of attack in radians [rad].
    cl_alpha : float, default=2*pi
        Lift curve slope dC_L / d(alpha) at alpha = 0 [1/rad].

    Returns
    -------
    float
        Lift coefficient C_L.
    """
    alpha = float(np.float64(angle_of_attack))
    slope = float(np.float64(cl_alpha))
    return float(np.float64(0.5 * slope * math.sin(2.0 * alpha)))


def drag_force(
    dyn_press: float,
    reference_area: float,
    cd: float,
    rel_vel: Vector2D,
) -> Vector2D:
    """Compute aerodynamic drag force vector opposing relative velocity.

    Equation:
        F_drag = -q * A_ref * C_d * v_rel_hat

    Parameters
    ----------
    dyn_press : float
        Dynamic pressure q [Pa].
    reference_area : float
        Reference aerodynamic frontal cross-section A_ref [m^2].
    cd : float
        Drag coefficient C_d.
    rel_vel : Vector2D
        Relative velocity vector v_rel [m/s].

    Returns
    -------
    Vector2D
        Drag force vector in inertial coordinates [N].
    """
    v_mag = rel_vel.magnitude()
    if math.isclose(v_mag, 0.0, abs_tol=1e-15) or dyn_press <= 0.0 or reference_area <= 0.0:
        return Vector2D(0.0, 0.0)
    v_hat = rel_vel / v_mag
    f_mag = float(np.float64(dyn_press * reference_area * cd))
    return -v_hat * f_mag


def lift_force(
    dyn_press: float,
    reference_area: float,
    cl: float,
    rel_vel: Vector2D,
) -> Vector2D:
    """Compute aerodynamic lift force vector perpendicular to relative velocity.

    Equation:
        F_lift = q * A_ref * C_L * v_rel_perp_hat
    Where in 2D, v_rel_perp_hat = (-v_y, v_x) / ||v_rel||.

    Parameters
    ----------
    dyn_press : float
        Dynamic pressure q [Pa].
    reference_area : float
        Reference aerodynamic area A_ref [m^2].
    cl : float
        Lift coefficient C_L.
    rel_vel : Vector2D
        Relative velocity vector v_rel [m/s].

    Returns
    -------
    Vector2D
        Lift force vector in inertial coordinates [N].
    """
    v_mag = rel_vel.magnitude()
    if math.isclose(v_mag, 0.0, abs_tol=1e-15) or dyn_press <= 0.0 or reference_area <= 0.0:
        return Vector2D(0.0, 0.0)
    v_hat = rel_vel / v_mag
    v_perp = Vector2D(-v_hat.y, v_hat.x)
    f_mag = float(np.float64(dyn_press * reference_area * cl))
    return v_perp * f_mag


def calculate_aero_forces(
    craft_pos: Vector2D,
    craft_vel: Vector2D,
    attitude_angle: float,
    body: CelestialBody,
    reference_area: float,
    length: float = 1.0,
    cd_axial: float = 0.3,
    cd_transverse: float = 1.2,
    center_of_mass_offset: float = 0.0,
    center_of_pressure_offset: float = 0.0,
    *,
    cl_alpha: float = 0.0,
    transverse_area: Optional[float] = None,
    angular_velocity: float = 0.0,
) -> AeroState:
    """Compute comprehensive aerodynamic state, forces, and torque on a rocket body.

    Calculates:
        1. Planetary altitude and atmospheric presence.
        2. Relative flow velocity including planetary co-rotation wind.
        3. Heading unit vector u_hat = (cos theta, sin theta).
        4. Angle of attack alpha = atan2(v_rel_hat x u_hat, v_rel_hat . u_hat).
        5. Effective drag coefficient C_d(alpha) = cd_axial * cos^2(alpha) + cd_transverse * sin^2(alpha).
        6. Total aerodynamic drag force F_drag = -q * A_ref * C_d(alpha) * v_rel_hat (plus optional lift).
        7. Aerodynamic restoring/destabilizing torque around CoM:
           tau = (r_CoP - r_CoM) x F_drag = (CoP_offset - CoM_offset) * u_hat x F_drag.

    Parameters
    ----------
    craft_pos : Vector2D
        Inertial position coordinates of the spacecraft [m].
    craft_vel : Vector2D
        Inertial velocity vector of the spacecraft [m/s].
    attitude_angle : float
        Spacecraft centerline heading angle theta [rad] relative to +X axis.
    body : CelestialBody
        Primary celestial body with atmospheric properties and rotation.
    reference_area : float
        Aerodynamic reference cross-sectional area A_ref [m^2].
    length : float, default=1.0
        Total length of the rocket body from tail to nose [m].
    cd_axial : float, default=0.3
        Axial drag coefficient (zero angle of attack).
    cd_transverse : float, default=1.2
        Transverse / broadside drag coefficient (90 degree angle of attack).
    center_of_mass_offset : float, default=0.0
        Distance along the rocket centerline from tail to Center of Mass [m].
    center_of_pressure_offset : float, default=0.0
        Distance along the rocket centerline from tail to Center of Pressure [m].
    cl_alpha : float, default=0.0
        Lift curve slope dC_L / d(alpha). Set to 0.0 for pure drag modeling.

    Returns
    -------
    AeroState
        Dataclass containing total force vector, torque scalar, dynamic pressure,
        Mach number, and angle of attack.
    """
    if not isinstance(craft_pos, Vector2D):
        craft_pos = Vector2D.from_tuple(craft_pos)
    if not isinstance(craft_vel, Vector2D):
        craft_vel = Vector2D.from_tuple(craft_vel)

    zero_state = AeroState(
        force=Vector2D(0.0, 0.0),
        torque=0.0,
        dynamic_pressure=0.0,
        mach=0.0,
        angle_of_attack=0.0,
    )

    # 1. Atmospheric presence and altitude check
    if body.atmosphere is None:
        return zero_state

    alt = body.altitude(craft_pos)
    if alt >= body.atmosphere.cutoff_altitude or alt < 0.0:
        return zero_state

    # 2. Relative velocity
    wind_vel = body.wind_velocity_at(craft_pos)
    v_rel = relative_velocity(craft_vel, wind_vel)
    v_rel_mag = v_rel.magnitude()

    if math.isclose(v_rel_mag, 0.0, abs_tol=1e-12):
        return zero_state

    # Atmospheric thermodynamic properties
    rho = body.atmosphere.density(alt)
    a_sound = body.atmosphere.speed_of_sound(alt)
    q = dynamic_pressure(rho, v_rel_mag)
    mach = mach_number(v_rel_mag, a_sound)

    # 3. Heading and lateral normal unit vectors
    theta = float(np.float64(attitude_angle))
    u_hat = Vector2D(math.cos(theta), math.sin(theta))
    n_hat = Vector2D(-math.sin(theta), math.cos(theta))

    # 4. Angle of attack alpha: signed angle from v_rel to u_hat
    v_rel_hat = v_rel / v_rel_mag
    v_rel_perp = Vector2D(-v_rel_hat.y, v_rel_hat.x)
    sin_alpha = v_rel_hat.cross(u_hat)
    cos_alpha = v_rel_hat.dot(u_hat)
    alpha = float(np.float64(math.atan2(sin_alpha, cos_alpha)))

    a_trans = transverse_area if transverse_area is not None else reference_area

    # 5. Axial force along longitudinal body axis -u_hat
    f_axial_mag = q * reference_area * cd_axial * cos_alpha * abs(cos_alpha)
    f_axial = -u_hat * float(np.float64(f_axial_mag))

    # 6. Cross-flow normal force: Allen-Perkins viscous cross-flow along body normal n_hat
    f_cross_mag = q * a_trans * cd_transverse * abs(sin_alpha) * sin_alpha
    f_crossflow = n_hat * float(np.float64(f_cross_mag))

    # 7. Potential slender-body lift: along v_rel_perp
    if not math.isclose(cl_alpha, 0.0, abs_tol=1e-15):
        cl = lift_coefficient(alpha, cl_alpha)
        f_potential = v_rel_perp * float(np.float64(q * reference_area * cl))
    else:
        f_potential = Vector2D(0.0, 0.0)

    total_force = f_axial + f_crossflow + f_potential

    # 8. Aerodynamic torque around Center of Mass:
    # r_CoP - r_CoM = (center_of_pressure_offset - center_of_mass_offset) * u_hat
    cop_com_delta = float(np.float64(center_of_pressure_offset - center_of_mass_offset))
    r_arm = u_hat * cop_com_delta
    static_torque = float(np.float64(r_arm.cross(total_force)))

    # Rotational pitch damping torque opposing angular velocity omega:
    # tau_damp = - 0.5 * rho * v_rel * Cd_transverse * A_transverse * (L^2 / 12) * omega
    tau_damp = 0.0
    if abs(angular_velocity) > 1e-12:
        tau_damp = -float(
            np.float64(
                0.5 * rho * v_rel_mag * cd_transverse * a_trans * ((length * length) / 12.0) * angular_velocity
            )
        )

    torque = float(np.float64(static_torque + tau_damp))

    return AeroState(
        force=total_force,
        torque=torque,
        dynamic_pressure=q,
        mach=mach,
        angle_of_attack=alpha,
    )


__all__ = [
    "AeroState",
    "calculate_aero_forces",
    "drag_force",
    "dynamic_pressure",
    "lift_coefficient",
    "lift_force",
    "mach_number",
    "relative_velocity",
]
