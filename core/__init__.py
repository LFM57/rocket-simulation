"""Core mathematical foundations and numerical integrators for 2D rocket simulation."""

from core.constants import (
    G,
    G0,
    GRAVITATIONAL_CONSTANT,
    STANDARD_GRAVITY,
    g0,
)
from core.integrator import (
    IDX_OMEGA,
    IDX_THETA,
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    STATE_DIM,
    DerivativeFunc,
    StateIndex,
    integrate_trajectory,
    make_state,
    rk4_step,
    velocity_verlet_step,
    wrap_angle_pi,
)
from core.vector2d import Vector2D

__all__ = [
    "Vector2D",
    "STATE_DIM",
    "IDX_X",
    "IDX_Y",
    "IDX_VX",
    "IDX_VY",
    "IDX_THETA",
    "IDX_OMEGA",
    "StateIndex",
    "DerivativeFunc",
    "make_state",
    "rk4_step",
    "velocity_verlet_step",
    "integrate_trajectory",
    "wrap_angle_pi",
    "G",
    "GRAVITATIONAL_CONSTANT",
    "g0",
    "G0",
    "STANDARD_GRAVITY",
]