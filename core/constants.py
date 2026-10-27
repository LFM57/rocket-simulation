"""Fundamental physical constants for orbital mechanics and aerospace dynamics.

All values adhere strictly to CODATA 2018 recommendations and standard SI units.
Values are represented with 64-bit floating-point precision (np.float64 / IEEE 754 double).
"""

from __future__ import annotations

import numpy as np

# Universal Gravitational Constant [m^3 kg^-1 s^-2] (CODATA 2018)
G: float = float(np.float64(6.67430e-11))
GRAVITATIONAL_CONSTANT: float = G

# Standard Earth sea-level acceleration of gravity [m/s^2] (ISO 80000-3)
g0: float = float(np.float64(9.80665))
G0: float = g0
STANDARD_GRAVITY: float = g0

__all__ = [
    "G",
    "GRAVITATIONAL_CONSTANT",
    "g0",
    "G0",
    "STANDARD_GRAVITY",
]
