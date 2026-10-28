"""Modular multi-stage rocket vehicle representation.

This module implements:
    - Stage: Collection of parts with separation state and mass aggregations.
    - Rocket: Ordered stack of stages with dynamic Center of Mass (CoM),
      dynamic Moment of Inertia (I_zz via Parallel Axis Theorem), dynamic
      Center of Pressure (CoP), pressure-dependent propellant consumption,
      gimbal torque, and staging separation.

All calculations enforce 64-bit floating-point precision (np.float64 / IEEE 754 double).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from core.vector2d import Vector2D
from models.parts import (
    Decoupler,
    Engine,
    FuelTank,
    Part,
    Payload,
)

Scalar = Union[int, float, np.floating, np.integer]


@dataclass(slots=True)
class Stage:
    """A single rocket stage containing component parts.

    Parameters
    ----------
    stage_id : str
        Unique identifier for this stage.
    name : str
        Human-readable name.
    parts : list[Part]
        Ordered collection of parts in this stage.
    staging_order : int, default=0
        Sequence index for staging (0 = first/bottom stage to ignite).
    is_active : bool, default=False
        Whether this stage is currently active/firing.
    is_separated : bool, default=False
        Whether this stage has been decoupled and discarded.
    position_offset : float, default=0.0
        Distance of stage base along the rocket longitudinal axis relative
        to the rocket base [m].
    """

    stage_id: str
    name: str
    parts: list[Part] = field(default_factory=list)
    staging_order: int = 0
    is_active: bool = False
    is_separated: bool = False
    position_offset: float = 0.0

    def __post_init__(self) -> None:
        """Enforce 64-bit precision and types."""
        self.stage_id = str(self.stage_id)
        self.name = str(self.name)
        self.staging_order = int(self.staging_order)
        self.is_active = bool(self.is_active)
        self.is_separated = bool(self.is_separated)
        self.position_offset = float(np.float64(self.position_offset))

    @property
    def id(self) -> str:
        """Alias for stage_id."""
        return self.stage_id

    def dry_mass(self) -> float:
        """Calculate aggregate dry structural mass of the stage [kg]."""
        return float(np.float64(sum(p.dry_mass for p in self.parts)))

    def propellant_mass(self) -> float:
        """Calculate aggregate remaining propellant mass across all fuel tanks [kg]."""
        return float(
            np.float64(
                sum(p.propellant_mass for p in self.parts if isinstance(p, FuelTank))
            )
        )

    def total_mass(self) -> float:
        """Calculate aggregate current total mass of the stage [kg]."""
        return float(np.float64(sum(p.current_mass for p in self.parts)))

    @property
    def engines(self) -> list[Engine]:
        """List of all Engine parts in this stage."""
        return [p for p in self.parts if isinstance(p, Engine)]

    @property
    def fuel_tanks(self) -> list[FuelTank]:
        """List of all FuelTank parts in this stage."""
        return [p for p in self.parts if isinstance(p, FuelTank)]

    @property
    def decouplers(self) -> list[Decoupler]:
        """List of all Decoupler parts in this stage."""
        return [p for p in self.parts if isinstance(p, Decoupler)]

    @property
    def payloads(self) -> list[Payload]:
        """List of all Payload parts in this stage."""
        return [p for p in self.parts if isinstance(p, Payload)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize stage to dictionary representation."""
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "staging_order": self.staging_order,
            "is_active": self.is_active,
            "is_separated": self.is_separated,
            "position_offset": self.position_offset,
            "parts": [p.to_dict() for p in self.parts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stage:
        """Construct Stage from dictionary representation."""
        parts_data = data.get("parts", [])
        parts = [Part.from_dict(p) for p in parts_data]
        stage_id = data.get("stage_id", data.get("id", ""))
        return cls(
            stage_id=str(stage_id),
            name=str(data.get("name", "")),
            parts=parts,
            staging_order=int(data.get("staging_order", 0)),
            is_active=bool(data.get("is_active", False)),
            is_separated=bool(data.get("is_separated", False)),
            position_offset=float(data.get("position_offset", 0.0)),
        )


@dataclass(slots=True)
class Rocket:
    """Modular multi-stage rocket vehicle representation.

    Parameters
    ----------
    name : str, default="Rocket"
        Human-readable name of the vehicle.
    stages : list[Stage]
        Ordered stack of stages.
    active_stage_index : int, default=0
        Index of the currently active stage in stages list.
    """

    name: str = "Rocket"
    stages: list[Stage] = field(default_factory=list)
    active_stage_index: int = 0
    last_separation_impulse: float = 0.0

    def __post_init__(self) -> None:
        """Ensure active stage pointers and types."""
        self.name = str(self.name)
        self.active_stage_index = int(self.active_stage_index)

        # If stages exist but none are active, activate the specified stage
        if self.stages and not any(s.is_active for s in self.stages if not s.is_separated):
            if 0 <= self.active_stage_index < len(self.stages):
                if not self.stages[self.active_stage_index].is_separated:
                    self.stages[self.active_stage_index].is_active = True

    @property
    def active_stage(self) -> Optional[Stage]:
        """Return the currently active non-separated Stage, or None."""
        if 0 <= self.active_stage_index < len(self.stages):
            candidate = self.stages[self.active_stage_index]
            if candidate.is_active and not candidate.is_separated:
                return candidate

        # Fallback: find any active non-separated stage
        for s in self.stages:
            if s.is_active and not s.is_separated:
                return s
        return None

    def total_mass(self) -> float:
        """Compute sum of all masses in non-separated stages [kg]."""
        total = 0.0
        for stage in self.stages:
            if not stage.is_separated:
                total += stage.total_mass()
        return float(np.float64(total))

    def dry_mass(self) -> float:
        """Compute sum of dry masses in non-separated stages [kg]."""
        total = 0.0
        for stage in self.stages:
            if not stage.is_separated:
                total += stage.dry_mass()
        return float(np.float64(total))

    def propellant_mass(self) -> float:
        """Compute sum of propellant mass across non-separated stages [kg]."""
        total = 0.0
        for stage in self.stages:
            if not stage.is_separated:
                total += stage.propellant_mass()
        return float(np.float64(total))

    def center_of_mass(self) -> float:
        """Compute 1D coordinate of Center of Mass (CoM) relative to rocket base [m].

        Equation:
            y_CoM = sum(m_i * y_i) / sum(m_i)
        where y_i = stage.position_offset + part.position_offset.

        Returns
        -------
        float
            CoM distance from rocket base along the centerline [m].
        """
        weighted_sum = 0.0
        total_m = 0.0

        for stage in self.stages:
            if stage.is_separated:
                continue
            stage_base = stage.position_offset
            for part in stage.parts:
                m_i = part.current_mass
                y_i = stage_base + part.position_offset
                weighted_sum += m_i * y_i
                total_m += m_i

        if total_m <= 0.0:
            return 0.0
        return float(np.float64(weighted_sum / total_m))

    def moment_of_inertia(self) -> float:
        """Calculate rotational inertia I_zz about dynamic CoM using Parallel Axis Theorem.

        Equation:
            I_zz = sum_i (I_local,i + m_i * (y_i - y_CoM)^2)

        Returns
        -------
        float
            Moment of inertia [kg m^2].
        """
        y_com = self.center_of_mass()
        total_i = 0.0

        for stage in self.stages:
            if stage.is_separated:
                continue
            stage_base = stage.position_offset
            for part in stage.parts:
                m_i = part.current_mass
                y_i = stage_base + part.position_offset
                i_local = part.local_moment_of_inertia()
                d_i = y_i - y_com
                total_i += i_local + m_i * (d_i * d_i)

        return float(np.float64(total_i))

    def center_of_pressure(self) -> float:
        """Compute area-weighted aerodynamic center of pressure along centerline [m].

        Clamps parallel clusters sharing the same axial station to stage outer diameter,
        and excludes enclosed internal components.

        Equation:
            y_CoP = sum(A_i * y_i) / sum(A_i)

        Returns
        -------
        float
            CoP distance from rocket base along the centerline [m].
        """
        weighted_sum = 0.0
        total_area = 0.0

        for stage in self.stages:
            if stage.is_separated:
                continue
            stage_base = stage.position_offset
            stage_radius = max((p.radius for p in stage.parts if not getattr(p, "is_enclosed", False)), default=0.0)
            stage_diameter = 2.0 * stage_radius

            # Group exposed parts by axial position to detect lateral clusters
            axial_groups: dict[float, list[Part]] = {}
            for part in stage.parts:
                if getattr(part, "is_enclosed", False) or part.projected_area <= 0.0:
                    continue
                pos = round(part.position_offset, 3)
                axial_groups.setdefault(pos, []).append(part)

            for pos, parts in axial_groups.items():
                y_i = stage_base + pos
                if len(parts) > 1 and stage_diameter > 0.0:
                    # Parallel cluster: clamp combined width to stage outer diameter
                    max_len = max(p.length for p in parts)
                    sum_width = sum(2.0 * p.radius for p in parts)
                    eff_width = min(sum_width, stage_diameter)
                    cluster_area = eff_width * max_len
                    weighted_sum += cluster_area * y_i
                    total_area += cluster_area
                else:
                    for part in parts:
                        a_i = part.projected_area
                        weighted_sum += a_i * y_i
                        total_area += a_i

        if total_area <= 0.0:
            return 0.0
        return float(np.float64(weighted_sum / total_area))

    def burn_fuel(self, dt: float, ambient_pressure: float) -> tuple[float, float, float]:
        """Consume propellant from active stage fuel tanks according to engine mass flow.

        Handles propellant starvation: if fuel tanks are exhausted, thrust cuts off.

        Parameters
        ----------
        dt : float
            Simulation time increment [s].
        ambient_pressure : float
            Static atmospheric pressure [Pa].

        Returns
        -------
        tuple[float, float, float]
            (total_thrust_magnitude [N], gimbal_angle [rad], mass_flow_rate [kg/s]).
        """
        dt_val = float(np.float64(dt))
        if dt_val <= 0.0:
            return 0.0, 0.0, 0.0

        stage = self.active_stage
        if stage is None or stage.is_separated:
            return 0.0, 0.0, 0.0

        tanks = [p for p in stage.parts if isinstance(p, FuelTank)]
        avail_prop = sum(t.propellant_mass for t in tanks)

        ignited_engines = [
            p for p in stage.parts if isinstance(p, Engine) and p.is_ignited and p.throttle > 0.0
        ]
        if not ignited_engines:
            return 0.0, 0.0, 0.0

        # Propellant starvation check
        if avail_prop <= 0.0:
            for eng in ignited_engines:
                eng.shutdown()
            return 0.0, 0.0, 0.0

        # Calculate nominal thrust and mass flow for each active engine
        engine_thrusts: list[float] = []
        engine_flows: list[float] = []
        for eng in ignited_engines:
            t_val, m_dot = eng.get_thrust_and_mass_flow(ambient_pressure)
            engine_thrusts.append(t_val)
            engine_flows.append(m_dot)

        total_thrust_nom = sum(engine_thrusts)
        total_flow_nom = sum(engine_flows)
        prop_needed = total_flow_nom * dt_val

        if prop_needed <= 0.0:
            return 0.0, 0.0, 0.0

        # Determine actual burn based on fuel availability
        if prop_needed > avail_prop:
            # Tank emptied during this time step (starvation)
            burn_fraction = avail_prop / prop_needed
            actual_burned = avail_prop
            actual_thrust = total_thrust_nom * burn_fraction
            actual_flow = actual_burned / dt_val

            # Drain all remaining propellant
            for tank in tanks:
                tank.drain(tank.propellant_mass)

            # Auto shutdown engines due to fuel exhaustion
            for eng in ignited_engines:
                eng.shutdown()
        else:
            actual_burned = prop_needed
            actual_thrust = total_thrust_nom
            actual_flow = total_flow_nom

            # Drain proportionally from available tanks
            for tank in tanks:
                if avail_prop > 0.0:
                    drain_share = prop_needed * (tank.propellant_mass / avail_prop)
                    tank.drain(drain_share)

        # Compute thrust-weighted average gimbal angle
        if total_thrust_nom > 0.0:
            weighted_gimbal = sum(
                t * eng.current_gimbal for eng, t in zip(ignited_engines, engine_thrusts)
            ) / total_thrust_nom
        else:
            weighted_gimbal = (
                ignited_engines[0].current_gimbal if ignited_engines else 0.0
            )

        return (
            float(np.float64(actual_thrust)),
            float(np.float64(weighted_gimbal)),
            float(np.float64(actual_flow)),
        )

    def thrust_force_and_torque(self, ambient_pressure: float) -> tuple[Vector2D, float]:
        """Compute net thrust force vector and rotational torque about dynamic CoM.

        Equations:
            F_thrust,body = Vector2D(T * sin(delta), T * cos(delta))
            tau_thrust = -(y_engine - y_CoM) * (T * sin(delta))

        Parameters
        ----------
        ambient_pressure : float
            Static atmospheric pressure [Pa].

        Returns
        -------
        tuple[Vector2D, float]
            (thrust_force_body [N], thrust_torque [N m]).
        """
        stage = self.active_stage
        if stage is None or stage.is_separated:
            return Vector2D(0.0, 0.0), 0.0

        # Starvation check: no thrust if active stage fuel is empty
        if stage.propellant_mass() <= 0.0:
            return Vector2D(0.0, 0.0), 0.0

        ignited_engines = [
            p for p in stage.parts if isinstance(p, Engine) and p.is_ignited and p.throttle > 0.0
        ]
        if not ignited_engines:
            return Vector2D(0.0, 0.0), 0.0

        y_com = self.center_of_mass()
        total_fx = 0.0
        total_fy = 0.0
        total_tau = 0.0
        stage_base = stage.position_offset

        for eng in ignited_engines:
            t_val, _ = eng.get_thrust_and_mass_flow(ambient_pressure)
            if t_val <= 0.0:
                continue

            delta = eng.current_gimbal
            # Nozzle deflects by delta; reaction thrust has transverse component in +X_body
            fx = t_val * math.sin(delta)
            fy = t_val * math.cos(delta)
            total_fx += fx
            total_fy += fy

            # Rigorous 2D cross product: tau = r_x * F_y - r_y * F_x = 0 - arm * fx
            y_eng = stage_base + eng.position_offset
            arm = y_eng - y_com
            total_tau += -arm * fx

        return Vector2D(float(np.float64(total_fx)), float(np.float64(total_fy))), float(
            np.float64(total_tau)
        )

    def stage(self) -> Optional[Stage]:
        """Decouple the currently active lowest stage and activate the next stage.

        Drops the separated stage from the active rocket mass/inertia stack,
        activates the next stage in staging sequence, and returns the decoupled stage.

        Returns
        -------
        Optional[Stage]
            The decoupled stage instance, or None if no active stage could be decoupled.
        """
        current = self.active_stage
        if current is None:
            return None

        # Decouple the current active stage
        current.is_active = False
        current.is_separated = True
        for eng in current.engines:
            eng.shutdown()

        # Search for the next stage in staging order
        next_stage: Optional[Stage] = None
        next_idx: int = -1

        for idx, s in enumerate(self.stages):
            if not s.is_separated and s.staging_order > current.staging_order:
                if next_stage is None or s.staging_order < next_stage.staging_order:
                    next_stage = s
                    next_idx = idx

        if next_stage is not None:
            next_stage.is_active = True
            for p in next_stage.parts:
                p.is_enclosed = False
            self.active_stage_index = next_idx
        else:
            # Advance past stages
            self.active_stage_index = len(self.stages)

        # Record separation impulse from decouplers on separating interface
        impulse = sum(d.separation_impulse for d in current.decouplers)
        if impulse == 0.0 and next_stage is not None:
            impulse = sum(d.separation_impulse for d in next_stage.decouplers)
        self.last_separation_impulse = float(np.float64(impulse))

        return current

    def to_dict(self) -> dict[str, Any]:
        """Serialize full rocket specification to a dictionary."""
        return {
            "name": self.name,
            "active_stage_index": self.active_stage_index,
            "stages": [s.to_dict() for s in self.stages],
        }

    def to_json(self, indent: Optional[int] = None) -> Union[dict[str, Any], str]:
        """Serialize rocket specification.

        Parameters
        ----------
        indent : Optional[int], default=None
            If None, returns a dictionary representation as per requirement.
            If an integer, returns a formatted JSON string.

        Returns
        -------
        Union[dict[str, Any], str]
            Dictionary or JSON string.
        """
        data = self.to_dict()
        if indent is not None:
            return json.dumps(data, indent=indent)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rocket:
        """Construct Rocket from dictionary representation."""
        stages_data = data.get("stages", [])
        stages = [Stage.from_dict(s) for s in stages_data]
        return cls(
            name=str(data.get("name", "Rocket")),
            stages=stages,
            active_stage_index=int(data.get("active_stage_index", 0)),
        )

    @classmethod
    def from_json(cls, json_str_or_dict: Union[str, dict[str, Any]]) -> Rocket:
        """Construct Rocket from a JSON string or dictionary representation.

        Parameters
        ----------
        json_str_or_dict : Union[str, dict[str, Any]]
            JSON string or dictionary containing rocket definition.

        Returns
        -------
        Rocket
            Deserialized Rocket instance.
        """
        if isinstance(json_str_or_dict, str):
            data = json.loads(json_str_or_dict)
        elif isinstance(json_str_or_dict, dict):
            data = json_str_or_dict
        else:
            raise TypeError(
                f"Expected str or dict for from_json, got {type(json_str_or_dict).__name__}"
            )
        return cls.from_dict(data)


__all__ = [
    "Rocket",
    "Stage",
]
