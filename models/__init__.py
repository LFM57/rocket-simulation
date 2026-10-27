"""Modular rocket model package for planar aerospace simulation.

This package provides:
    - Component dataclasses (Part, FuelTank, Engine, Decoupler, Payload).
    - Multi-stage vehicle architecture (Stage, Rocket).
    - Realistic rocket presets (Falcon 9 / two-stage launch vehicles).
"""

from __future__ import annotations

from models.parts import Decoupler, Engine, FuelTank, Part, Payload
from models.presets import FALCON_9_CONFIG, create_falcon_9, get_falcon_9, make_falcon_9
from models.rocket import Rocket, Stage

__all__ = [
    "Decoupler",
    "Engine",
    "FALCON_9_CONFIG",
    "FuelTank",
    "Part",
    "Payload",
    "Rocket",
    "Stage",
    "create_falcon_9",
    "get_falcon_9",
    "make_falcon_9",
]
