"""Aerospace physics, orbital mechanics, planetary atmospheres, and aerodynamics package."""

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
from physics.atmosphere import (
    EARTH_ATMOSPHERE,
    AtmosphereModel,
)
from physics.celestial import (
    EARTH,
    MOON,
    CelestialBody,
    CollisionWarning,
    get_earth,
    get_moon,
)
from physics.orbit import (
    Orbit,
    calculate_orbit,
    propagate_kepler,
    solve_kepler_elliptic,
    solve_kepler_hyperbolic,
)

__all__ = [
    # Celestial
    "CelestialBody",
    "CollisionWarning",
    "EARTH",
    "MOON",
    "get_earth",
    "get_moon",
    # Orbit
    "Orbit",
    "calculate_orbit",
    "propagate_kepler",
    "solve_kepler_elliptic",
    "solve_kepler_hyperbolic",
    # Atmosphere
    "AtmosphereModel",
    "EARTH_ATMOSPHERE",
    # Aerodynamics
    "AeroState",
    "calculate_aero_forces",
    "drag_force",
    "dynamic_pressure",
    "lift_coefficient",
    "lift_force",
    "mach_number",
    "relative_velocity",
]
