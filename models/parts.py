"""Modular rocket component parts oriented along the longitudinal axis.

Components are modeled along the vertical/longitudinal axis y, measuring:
    - dry_mass [kg]
    - length [m]
    - radius [m]
    - position_offset [m] (distance of part center from the base of its stage)

All calculations enforce 64-bit floating-point precision (np.float64 / IEEE 754 double).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union

import numpy as np

from core.constants import g0

# Standard sea-level atmospheric pressure [Pa] (ISO 2533)
STANDARD_SEA_LEVEL_PRESSURE: float = 101325.0
P0: float = STANDARD_SEA_LEVEL_PRESSURE

Scalar = Union[int, float, np.floating, np.integer]


@dataclass(slots=True)
class Part:
    """Base class for all rocket components.

    Parameters
    ----------
    id : str
        Unique identifier for the part.
    name : str
        Human-readable name.
    dry_mass : float
        Dry structural mass of the component [kg].
    length : float
        Longitudinal length along the y-axis [m].
    radius : float
        Outer radius perpendicular to centerline [m].
    position_offset : float, default=0.0
        Distance of part center from the base of its stage [m].
    """

    id: str
    name: str
    dry_mass: float
    length: float
    radius: float
    position_offset: float = 0.0

    def __post_init__(self) -> None:
        """Enforce 64-bit floating-point precision and physical bounds."""
        self.dry_mass = float(np.float64(self.dry_mass))
        self.length = float(np.float64(self.length))
        self.radius = float(np.float64(self.radius))
        self.position_offset = float(np.float64(self.position_offset))

        if self.dry_mass < 0.0:
            raise ValueError(f"Dry mass cannot be negative, got {self.dry_mass}")
        if self.length < 0.0:
            raise ValueError(f"Length cannot be negative, got {self.length}")
        if self.radius < 0.0:
            raise ValueError(f"Radius cannot be negative, got {self.radius}")

    @property
    def current_mass(self) -> float:
        """Current total mass of the part [kg]."""
        return self.dry_mass

    @property
    def projected_area(self) -> float:
        """Projected lateral aerodynamic area A = 2 * r * L [m^2]."""
        return float(np.float64(2.0 * self.radius * self.length))

    def local_moment_of_inertia(self) -> float:
        """Calculate dynamic local moment of inertia about transverse central axis.

        Uses the solid/hollow cylinder transverse inertia approximation:
            I_local = (1/12) * m * (3 * r^2 + L^2)

        Returns
        -------
        float
            Local moment of inertia [kg m^2].
        """
        m = self.current_mass
        r = self.radius
        l_val = self.length
        return float(np.float64((1.0 / 12.0) * m * (3.0 * r * r + l_val * l_val)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize part to a clean dictionary representation."""
        return {
            "type": self.__class__.__name__,
            "id": self.id,
            "name": self.name,
            "dry_mass": self.dry_mass,
            "length": self.length,
            "radius": self.radius,
            "position_offset": self.position_offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Part:
        """Construct a Part (or subclass) from a dictionary representation."""
        part_type = data.get("type", cls.__name__)
        target_cls = PART_REGISTRY.get(part_type, cls)
        if target_cls is not cls and issubclass(target_cls, Part):
            return target_cls.from_dict(data)

        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            dry_mass=float(data["dry_mass"]),
            length=float(data["length"]),
            radius=float(data["radius"]),
            position_offset=float(data.get("position_offset", 0.0)),
        )


@dataclass(slots=True)
class FuelTank(Part):
    """Propellant tank component holding consumable fuel/oxidizer.

    Parameters
    ----------
    propellant_mass : float, default=0.0
        Current consumable propellant mass [kg].
    propellant_capacity : float, default=0.0
        Maximum consumable propellant mass capacity [kg].
    """

    propellant_mass: float = 0.0
    propellant_capacity: float = 0.0

    def __post_init__(self) -> None:
        """Enforce 64-bit precision and propellant bounds."""
        super().__post_init__()
        self.propellant_mass = float(np.float64(self.propellant_mass))
        self.propellant_capacity = float(np.float64(self.propellant_capacity))

        if self.propellant_capacity < 0.0:
            raise ValueError(
                f"Propellant capacity cannot be negative, got {self.propellant_capacity}"
            )
        if self.propellant_mass < 0.0:
            raise ValueError(f"Propellant mass cannot be negative, got {self.propellant_mass}")
        if self.propellant_mass > self.propellant_capacity:
            # Clamp to capacity if slightly over due to float imprecision
            self.propellant_mass = self.propellant_capacity

    @property
    def current_mass(self) -> float:
        """Total current mass: dry_mass + propellant_mass [kg]."""
        return float(np.float64(self.dry_mass + self.propellant_mass))

    def drain(self, amount: float) -> float:
        """Drain propellant from the tank, clamped at 0.

        Parameters
        ----------
        amount : float
            Requested propellant amount to drain [kg].

        Returns
        -------
        float
            Actual drained propellant mass [kg].
        """
        amt = float(np.float64(amount))
        if amt <= 0.0:
            return 0.0
        actual = min(self.propellant_mass, amt)
        self.propellant_mass = float(np.float64(self.propellant_mass - actual))
        return actual

    def local_moment_of_inertia(self) -> float:
        """Dynamic local moment of inertia using current mass.

        Equation:
            I_local = (1/12) * m * (3 * r^2 + L^2)
        where m = dry_mass + propellant_mass.
        """
        m = self.current_mass
        r = self.radius
        l_val = self.length
        return float(np.float64((1.0 / 12.0) * m * (3.0 * r * r + l_val * l_val)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize fuel tank to dictionary."""
        d = super().to_dict()
        d.update(
            {
                "propellant_mass": self.propellant_mass,
                "propellant_capacity": self.propellant_capacity,
            }
        )
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FuelTank:
        """Construct FuelTank from dictionary."""
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            dry_mass=float(data["dry_mass"]),
            length=float(data["length"]),
            radius=float(data["radius"]),
            position_offset=float(data.get("position_offset", 0.0)),
            propellant_mass=float(data.get("propellant_mass", 0.0)),
            propellant_capacity=float(data.get("propellant_capacity", 0.0)),
        )


@dataclass(slots=True)
class Engine(Part):
    """Rocket propulsion engine with pressure-dependent Isp, thrust, and gimbaling.

    Parameters
    ----------
    thrust_vac : float, default=0.0
        Thrust in vacuum [N].
    thrust_sl : float, default=0.0
        Thrust at standard sea-level pressure (101325 Pa) [N].
    isp_vac : float, default=0.0
        Specific impulse in vacuum [s].
    isp_sl : float, default=0.0
        Specific impulse at standard sea-level pressure [s].
    gimbal_limit : float, default=0.08726646259971647
        Maximum nozzle gimbal deflection limit in radians (~5 degrees).
    current_gimbal : float, default=0.0
        Current deflection angle delta [rad]. Clamped to [-gimbal_limit, +gimbal_limit].
    throttle : float, default=1.0
        Throttle setting clamped between [0.0, 1.0].
    is_ignited : bool, default=False
        Whether the engine is currently ignited and firing.
    """

    thrust_vac: float = 0.0
    thrust_sl: float = 0.0
    isp_vac: float = 0.0
    isp_sl: float = 0.0
    gimbal_limit: float = 0.08726646259971647
    current_gimbal: float = 0.0
    throttle: float = 1.0
    is_ignited: bool = False

    def __post_init__(self) -> None:
        """Enforce 64-bit precision and valid bounds."""
        super().__post_init__()
        self.thrust_vac = float(np.float64(self.thrust_vac))
        self.thrust_sl = float(np.float64(self.thrust_sl))
        self.isp_vac = float(np.float64(self.isp_vac))
        self.isp_sl = float(np.float64(self.isp_sl))
        self.gimbal_limit = float(np.float64(abs(self.gimbal_limit)))
        self.current_gimbal = float(
            np.float64(max(-self.gimbal_limit, min(self.gimbal_limit, self.current_gimbal)))
        )
        self.throttle = float(np.float64(max(0.0, min(1.0, self.throttle))))
        self.is_ignited = bool(self.is_ignited)

        if self.thrust_vac < 0.0 or self.thrust_sl < 0.0:
            raise ValueError("Engine thrust values cannot be negative")
        if self.isp_vac < 0.0 or self.isp_sl < 0.0:
            raise ValueError("Specific impulse values cannot be negative")

    def set_throttle(self, throttle: float) -> None:
        """Set engine throttle level, clamped to [0.0, 1.0]."""
        self.throttle = float(np.float64(max(0.0, min(1.0, float(throttle)))))

    def set_gimbal(self, angle: float) -> None:
        """Set engine nozzle gimbal deflection angle [rad], clamped to gimbal limit."""
        ang = float(np.float64(angle))
        self.current_gimbal = float(
            np.float64(max(-self.gimbal_limit, min(self.gimbal_limit, ang)))
        )

    def ignite(self) -> None:
        """Ignite the engine."""
        self.is_ignited = True

    def shutdown(self) -> None:
        """Shut down the engine."""
        self.is_ignited = False

    def get_isp(self, ambient_pressure: float) -> float:
        """Calculate specific impulse interpolated between sea-level and vacuum.

        Equation:
            I_sp(P) = I_sp,vac - (I_sp,vac - I_sp,sl) * min(1.0, max(0.0, P / P_0))
        where P_0 = 101325 Pa.

        Parameters
        ----------
        ambient_pressure : float
            Local atmospheric static pressure [Pa].

        Returns
        -------
        float
            Specific impulse [s].
        """
        p = float(np.float64(max(0.0, ambient_pressure)))
        p_ratio = min(1.0, p / STANDARD_SEA_LEVEL_PRESSURE)
        isp = self.isp_vac - (self.isp_vac - self.isp_sl) * p_ratio
        return float(np.float64(max(0.0, isp)))

    def get_thrust_and_mass_flow(
        self, ambient_pressure: float, ignore_ignited: bool = False
    ) -> tuple[float, float]:
        """Compute effective thrust and propellant mass flow rate at ambient pressure.

        Equations:
            T = throttle * (T_vac - (T_vac - T_sl) * min(1.0, max(0.0, P / P_0)))
            m_dot = T / (I_sp * g_0)

        Parameters
        ----------
        ambient_pressure : float
            Local atmospheric static pressure [Pa].
        ignore_ignited : bool, default=False
            If True, evaluate thrust ignoring engine ignition state.

        Returns
        -------
        tuple[float, float]
            (thrust [N], mass_flow_rate [kg/s]). Returns (0.0, 0.0) if not ignited or throttle == 0.
        """
        if (not self.is_ignited and not ignore_ignited) or self.throttle <= 0.0:
            return 0.0, 0.0

        p = float(np.float64(max(0.0, ambient_pressure)))
        p_ratio = min(1.0, p / STANDARD_SEA_LEVEL_PRESSURE)
        t_base = self.thrust_vac - (self.thrust_vac - self.thrust_sl) * p_ratio
        thrust = float(np.float64(self.throttle * max(0.0, t_base)))

        isp = self.get_isp(ambient_pressure)
        m_dot = float(np.float64(thrust / (isp * g0))) if (isp > 0.0 and g0 > 0.0) else 0.0
        return thrust, m_dot

    def to_dict(self) -> dict[str, Any]:
        """Serialize engine to dictionary."""
        d = super().to_dict()
        d.update(
            {
                "thrust_vac": self.thrust_vac,
                "thrust_sl": self.thrust_sl,
                "isp_vac": self.isp_vac,
                "isp_sl": self.isp_sl,
                "gimbal_limit": self.gimbal_limit,
                "current_gimbal": self.current_gimbal,
                "throttle": self.throttle,
                "is_ignited": self.is_ignited,
            }
        )
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Engine:
        """Construct Engine from dictionary."""
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            dry_mass=float(data["dry_mass"]),
            length=float(data["length"]),
            radius=float(data["radius"]),
            position_offset=float(data.get("position_offset", 0.0)),
            thrust_vac=float(data.get("thrust_vac", 0.0)),
            thrust_sl=float(data.get("thrust_sl", 0.0)),
            isp_vac=float(data.get("isp_vac", 0.0)),
            isp_sl=float(data.get("isp_sl", 0.0)),
            gimbal_limit=float(data.get("gimbal_limit", 0.08726646259971647)),
            current_gimbal=float(data.get("current_gimbal", 0.0)),
            throttle=float(data.get("throttle", 1.0)),
            is_ignited=bool(data.get("is_ignited", False)),
        )


@dataclass(slots=True)
class Decoupler(Part):
    """Interstage separation decoupler.

    Parameters
    ----------
    separation_impulse : float, default=0.0
        Separation impulse J in [N s] delivered during stage separation.
    """

    separation_impulse: float = 0.0

    def __post_init__(self) -> None:
        """Enforce 64-bit precision."""
        super().__post_init__()
        self.separation_impulse = float(np.float64(self.separation_impulse))

    def to_dict(self) -> dict[str, Any]:
        """Serialize decoupler to dictionary."""
        d = super().to_dict()
        d.update(
            {
                "separation_impulse": self.separation_impulse,
            }
        )
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decoupler:
        """Construct Decoupler from dictionary."""
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            dry_mass=float(data["dry_mass"]),
            length=float(data["length"]),
            radius=float(data["radius"]),
            position_offset=float(data.get("position_offset", 0.0)),
            separation_impulse=float(data.get("separation_impulse", 0.0)),
        )


@dataclass(slots=True)
class Payload(Part):
    """Payload fairing, scientific satellite, or crew capsule component."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize payload to dictionary."""
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Payload:
        """Construct Payload from dictionary."""
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            dry_mass=float(data["dry_mass"]),
            length=float(data["length"]),
            radius=float(data["radius"]),
            position_offset=float(data.get("position_offset", 0.0)),
        )


PART_REGISTRY: dict[str, type[Part]] = {
    "Part": Part,
    "FuelTank": FuelTank,
    "Engine": Engine,
    "Decoupler": Decoupler,
    "Payload": Payload,
}

__all__ = [
    "Decoupler",
    "Engine",
    "FuelTank",
    "P0",
    "Part",
    "Payload",
    "STANDARD_SEA_LEVEL_PRESSURE",
]
