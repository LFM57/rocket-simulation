"""Realistic rocket configuration presets.

This module provides standard multi-stage orbital rocket configurations,
most notably a two-stage orbital rocket modeled on the SpaceX Falcon 9 launch vehicle.

Specifications:
    - Stage 1:
        - 9x Merlin 1D sea-level engines
        - Large LOX/RP-1 propellant tank (410,000 kg capacity)
        - Pneumatic interstage decoupler
    - Stage 2:
        - 1x Merlin Vacuum engine (MVac)
        - Upper stage LOX/RP-1 propellant tank (107,500 kg capacity)
        - Dragon capsule / fairing payload (5,000 kg dry mass)
"""

from __future__ import annotations

from typing import Any

from models.parts import Decoupler, Engine, FuelTank, Payload
from models.rocket import Rocket, Stage


def create_falcon_9() -> Rocket:
    """Create a realistic two-stage Falcon 9 orbital rocket model.

    Returns
    -------
    Rocket
        Fully configured two-stage Rocket instance.
    """
    gimbal_5deg = 0.08726646259971647  # 5 degrees in radians

    # Stage 1: 9x Merlin 1D Engines
    s1_engines: list[Engine] = []
    for i in range(9):
        eng = Engine(
            id=f"merlin_1d_{i}",
            name=f"Merlin 1D #{i + 1}",
            dry_mass=470.0,
            length=2.7,
            radius=0.6,
            position_offset=1.35,
            thrust_vac=914000.0,
            thrust_sl=845000.0,
            isp_vac=311.0,
            isp_sl=282.0,
            gimbal_limit=gimbal_5deg,
            current_gimbal=0.0,
            throttle=1.0,
            is_ignited=False,
        )
        s1_engines.append(eng)

    # Stage 1 Propellant Tank (LOX/RP-1)
    s1_tank = FuelTank(
        id="s1_tank",
        name="Stage 1 Propellant Tank",
        dry_mass=20000.0,
        length=40.0,
        radius=1.85,
        position_offset=22.7,
        propellant_mass=410000.0,
        propellant_capacity=410000.0,
    )

    # Interstage Decoupler
    interstage = Decoupler(
        id="interstage_decoupler",
        name="Pneumatic Interstage Decoupler",
        dry_mass=1000.0,
        length=3.0,
        radius=1.85,
        position_offset=43.5,
        separation_impulse=5000.0,
    )

    stage_1 = Stage(
        stage_id="stage_1",
        name="Stage 1 Booster",
        parts=[*s1_engines, s1_tank, interstage],
        staging_order=0,
        is_active=True,
        is_separated=False,
        position_offset=0.0,
    )

    # Stage 2: 1x Merlin Vacuum Engine
    mvac = Engine(
        id="merlin_vac",
        name="Merlin 1D Vacuum",
        dry_mass=900.0,
        length=3.0,
        radius=1.2,
        position_offset=1.5,
        thrust_vac=981000.0,
        thrust_sl=400000.0,
        isp_vac=348.0,
        isp_sl=200.0,
        gimbal_limit=gimbal_5deg,
        current_gimbal=0.0,
        throttle=1.0,
        is_ignited=False,
    )

    # Stage 2 Propellant Tank
    s2_tank = FuelTank(
        id="s2_tank",
        name="Stage 2 Propellant Tank",
        dry_mass=4000.0,
        length=10.0,
        radius=1.85,
        position_offset=7.0,
        propellant_mass=107500.0,
        propellant_capacity=107500.0,
    )

    # Payload Capsule
    payload = Payload(
        id="payload_capsule",
        name="Dragon Payload Capsule",
        dry_mass=5000.0,
        length=8.0,
        radius=1.85,
        position_offset=16.0,
    )

    stage_2 = Stage(
        stage_id="stage_2",
        name="Stage 2 Upper Stage",
        parts=[mvac, s2_tank, payload],
        staging_order=1,
        is_active=False,
        is_separated=False,
        position_offset=45.0,
    )

    return Rocket(
        name="Falcon 9",
        stages=[stage_1, stage_2],
        active_stage_index=0,
    )


# Convenient aliases
make_falcon_9 = create_falcon_9
get_falcon_9 = create_falcon_9
create_two_stage_rocket = create_falcon_9

# Pre-computed dictionary specification for JSON serialization tests
FALCON_9_CONFIG: dict[str, Any] = create_falcon_9().to_dict()

__all__ = [
    "FALCON_9_CONFIG",
    "create_falcon_9",
    "create_two_stage_rocket",
    "get_falcon_9",
    "make_falcon_9",
]
