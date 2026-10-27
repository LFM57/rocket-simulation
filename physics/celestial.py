"""Celestial body modeling and Newtonian inverse-square gravitation.

This module defines planetary and lunar bodies, their physical properties,
and gravitational acceleration field computations in 2D inertial space.
All calculations enforce 64-bit floating-point precision (np.float64).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from core.constants import G
from core.vector2d import Vector2D
from physics.atmosphere import AtmosphereModel


class CollisionWarning(UserWarning):
    """Warning emitted when an entity is at or below the surface of a celestial body."""

    pass


@dataclass
class CelestialBody:
    """Planetary or celestial body representation for 2D Newtonian gravitation.

    Parameters
    ----------
    name : str
        Common name of the celestial body (e.g., 'Earth', 'Moon').
    mass : float
        Total mass of the body in kilograms [kg].
    radius : float
        Mean volumetric radius of the body in meters [m].
    position : Vector2D, default=(0.0, 0.0)
        Position of the body center in the inertial frame [m].
    velocity : Vector2D, default=(0.0, 0.0)
        Velocity of the body center in the inertial frame [m/s].
    rotation_period : float, default=0.0
        Sidereal rotation period around its center in seconds [s] (0 for non-rotating).
    atmosphere : Optional[AtmosphereModel], default=None
        Atmospheric model for planetary density, pressure, and temperature.
    """

    name: str
    mass: float
    radius: float
    position: Vector2D = field(default_factory=lambda: Vector2D(0.0, 0.0))
    velocity: Vector2D = field(default_factory=lambda: Vector2D(0.0, 0.0))
    rotation_period: float = 0.0
    atmosphere: Optional[AtmosphereModel] = None

    def __post_init__(self) -> None:
        """Enforce 64-bit floating point precision on numerical attributes."""
        self.mass = float(np.float64(self.mass))
        self.radius = float(np.float64(self.radius))
        self.rotation_period = float(np.float64(self.rotation_period))

        if not isinstance(self.position, Vector2D):
            self.position = Vector2D.from_tuple(self.position)
        if not isinstance(self.velocity, Vector2D):
            self.velocity = Vector2D.from_tuple(self.velocity)

        if self.mass <= 0.0:
            raise ValueError(f"CelestialBody mass must be positive, got {self.mass}")
        if self.radius <= 0.0:
            raise ValueError(f"CelestialBody radius must be positive, got {self.radius}")

    # --- Derived Properties ---

    @property
    def mu(self) -> float:
        """Standard gravitational parameter: mu = G * M [m^3 s^-2]."""
        return float(np.float64(G) * np.float64(self.mass))

    @property
    def standard_gravitational_parameter(self) -> float:
        """Alias for mu."""
        return self.mu

    @property
    def gravitational_parameter(self) -> float:
        """Alias for mu."""
        return self.mu

    @property
    def surface_gravity(self) -> float:
        """Nominal acceleration of gravity at the body surface: g_surf = mu / R^2 [m/s^2]."""
        r_64 = np.float64(self.radius)
        return float(np.float64(self.mu) / (r_64 * r_64))

    @property
    def g_surface(self) -> float:
        """Alias for surface_gravity."""
        return self.surface_gravity

    def soi_radius(
        self,
        parent: Optional[Union[CelestialBody, float]] = None,
        semi_major_axis: Optional[float] = None,
        *,
        parent_body: Optional[Union[CelestialBody, float]] = None,
        a: Optional[float] = None,
    ) -> float:
        """Calculate the Laplace Sphere of Influence (SOI) radius relative to a parent body.

        Equation:
            r_SOI = a * (m_body / M_parent)^(2/5)

        Parameters
        ----------
        parent : CelestialBody or float, optional
            Parent body instance or parent body mass in kg.
        semi_major_axis : float, optional
            Semi-major axis of this body's orbit around the parent [m].
            If omitted and parent is a CelestialBody, relative distance is used.
        parent_body : CelestialBody or float, optional
            Keyword alias for parent.
        a : float, optional
            Keyword alias for semi_major_axis.

        Returns
        -------
        float
            Sphere of Influence radius in meters [m] (64-bit float).
        """
        p = parent if parent is not None else parent_body
        if p is None:
            raise ValueError("A parent body or parent mass must be specified to calculate Sphere of Influence.")

        sma = semi_major_axis if semi_major_axis is not None else a

        if isinstance(p, CelestialBody):
            m_parent = p.mass
            if sma is None:
                sma = (self.position - p.position).magnitude()
        else:
            m_parent = float(p)

        if sma is None or sma <= 0.0:
            raise ValueError(
                f"Semi-major axis must be positive to calculate Sphere of Influence, got {sma}"
            )
        if m_parent <= 0.0:
            raise ValueError(f"Parent mass must be positive, got {m_parent}")

        ratio = np.float64(self.mass) / np.float64(m_parent)
        r_soi = np.float64(sma) * (ratio ** np.float64(0.4))
        return float(r_soi)

    def sphere_of_influence(
        self,
        parent: Optional[Union[CelestialBody, float]] = None,
        semi_major_axis: Optional[float] = None,
        *,
        parent_body: Optional[Union[CelestialBody, float]] = None,
        a: Optional[float] = None,
    ) -> float:
        """Alias for soi_radius."""
        return self.soi_radius(parent, semi_major_axis, parent_body=parent_body, a=a)

    def sphere_of_influence_radius(
        self,
        parent: Optional[Union[CelestialBody, float]] = None,
        semi_major_axis: Optional[float] = None,
        *,
        parent_body: Optional[Union[CelestialBody, float]] = None,
        a: Optional[float] = None,
    ) -> float:
        """Alias for soi_radius."""
        return self.soi_radius(parent, semi_major_axis, parent_body=parent_body, a=a)

    def soi(
        self,
        parent: Optional[Union[CelestialBody, float]] = None,
        semi_major_axis: Optional[float] = None,
        *,
        parent_body: Optional[Union[CelestialBody, float]] = None,
        a: Optional[float] = None,
    ) -> float:
        """Alias for soi_radius."""
        return self.soi_radius(parent, semi_major_axis, parent_body=parent_body, a=a)

    # --- Kinematic & Gravitational Methods ---

    def gravitational_acceleration_at(self, pos: Vector2D) -> Vector2D:
        """Calculate Newtonian gravitational acceleration vector at an arbitrary position.

        Equation:
            a = - (mu / ||r||^3) * r
        Where:
            r = pos - body.position

        Parameters
        ----------
        pos : Vector2D
            Inertial coordinates [m] where acceleration is evaluated.

        Returns
        -------
        Vector2D
            Gravitational acceleration vector in m/s^2.

        Raises
        ------
        ZeroDivisionError
            If pos coincides with the body center of mass.
        """
        r_vec = pos - self.position
        r = r_vec.magnitude()
        if math.isclose(r, 0.0, abs_tol=1e-15):
            raise ZeroDivisionError(
                f"Gravitational acceleration undefined at center of mass of {self.name} (r=0)."
            )

        # a = - (mu / r^3) * r_vec
        factor = -self.mu / (r * r * r)
        return r_vec * factor

    def is_colliding(self, pos: Vector2D, warn: bool = True) -> bool:
        """Check if a position vector is inside the celestial body.

        Checks whether Euclidean distance from the body center is strictly less than radius:
            ||pos - position|| < radius

        Emits a CollisionWarning if collision is detected and warn=True.

        Parameters
        ----------
        pos : Vector2D
            Inertial coordinates [m] to evaluate.
        warn : bool, default=True
            Whether to emit a CollisionWarning when colliding.

        Returns
        -------
        bool
            True if position is inside the body radius, False otherwise.
        """
        dist = (pos - self.position).magnitude()
        colliding = dist < self.radius
        if colliding and warn:
            warnings.warn(
                f"Collision detected with {self.name}: position {pos} is below planetary radius "
                f"{self.radius:.1f} m (altitude {dist - self.radius:.1f} m).",
                CollisionWarning,
                stacklevel=2,
            )
        return colliding

    def altitude(self, pos: Vector2D) -> float:
        """Calculate geometric altitude above the mean spherical surface.

        Equation:
            altitude = ||pos - position|| - radius

        Parameters
        ----------
        pos : Vector2D
            Inertial coordinates [m].

        Returns
        -------
        float
            Altitude in meters [m]. Negative values denote positions inside the body.
        """
        dist = (pos - self.position).magnitude()
        return float(np.float64(dist) - np.float64(self.radius))

    def wind_velocity_at(self, pos: Vector2D) -> Vector2D:
        """Calculate atmospheric wind velocity due to planetary co-rotation.

        Models atmospheric co-rotation with the planet's rotation period:
            omega_planet = (2 * pi / rotation_period) * k_hat
            v_wind = omega_planet x (pos - body.position)
            In 2D: v_wind = Vector2D(-omega * r_y, omega * r_x)

        Parameters
        ----------
        pos : Vector2D
            Inertial coordinates [m].

        Returns
        -------
        Vector2D
            Atmospheric wind velocity vector in inertial frame [m/s].
            Returns Vector2D(0.0, 0.0) if rotation_period == 0 or if no atmosphere exists,
            or if position is at or above the atmospheric cutoff altitude.
        """
        if self.atmosphere is None or math.isclose(self.rotation_period, 0.0, abs_tol=1e-15):
            return Vector2D(0.0, 0.0)

        if not isinstance(pos, Vector2D):
            pos = Vector2D.from_tuple(pos)

        if self.altitude(pos) >= self.atmosphere.cutoff_altitude:
            return Vector2D(0.0, 0.0)

        r_vec = pos - self.position
        omega = float(np.float64(2.0 * math.pi / self.rotation_period))
        return Vector2D(-omega * r_vec.y, omega * r_vec.x)


# --- Standard Presets ---

EARTH: CelestialBody = CelestialBody(
    name="Earth",
    mass=5.9722e24,
    radius=6.371e6,
    position=Vector2D(0.0, 0.0),
    velocity=Vector2D(0.0, 0.0),
    rotation_period=86164.0,
    atmosphere=AtmosphereModel(),
)

MOON: CelestialBody = CelestialBody(
    name="Moon",
    mass=7.342e22,
    radius=1.7374e6,
    position=Vector2D(0.0, 0.0),
    velocity=Vector2D(0.0, 0.0),
    rotation_period=0.0,
    atmosphere=None,
)


def get_earth() -> CelestialBody:
    """Return a fresh independent instance of the standard Earth preset."""
    return CelestialBody(
        name="Earth",
        mass=5.9722e24,
        radius=6.371e6,
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(0.0, 0.0),
        rotation_period=86164.0,
        atmosphere=AtmosphereModel(),
    )


def get_moon() -> CelestialBody:
    """Return a fresh independent instance of the standard Moon preset."""
    return CelestialBody(
        name="Moon",
        mass=7.342e22,
        radius=1.7374e6,
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(0.0, 0.0),
        rotation_period=0.0,
        atmosphere=None,
    )


__all__ = [
    "CelestialBody",
    "CollisionWarning",
    "EARTH",
    "MOON",
    "get_earth",
    "get_moon",
]
