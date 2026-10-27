"""Planetary atmospheric modeling and barometric properties.

This module implements:
    - Exponential barometric density profiles with smooth cubic tapering to Karman line.
    - Tropospheric lapse rate temperature models with configurable minimum temperature caps.
    - Speed of sound and static pressure derived via the ideal gas law and adiabatic relations.

All calculations enforce 64-bit floating-point precision (np.float64 / IEEE 754 double).
SI Units:
    - Altitudes / Distances: meters [m]
    - Density: kilograms per cubic meter [kg/m^3]
    - Pressure: Pascals [Pa]
    - Temperature: Kelvin [K]
    - Speed of Sound: meters per second [m/s]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np

Scalar = Union[int, float, np.floating, np.integer]


@dataclass(slots=True)
class AtmosphereModel:
    """Planetary atmospheric model supporting barometric density and thermodynamic state.

    Parameters
    ----------
    sea_level_density : float, default=1.225
        Atmospheric density at mean sea level rho_0 [kg/m^3].
    scale_height : float, default=8500.0
        Atmospheric scale height H [m].
    cutoff_altitude : float, default=100000.0
        Maximum altitude (Karman line) h_max [m] above which density is identically 0.0.
    sea_level_temperature : float, default=288.15
        Standard sea-level temperature T_0 [K].
    lapse_rate : float, default=0.0065
        Tropospheric temperature lapse rate L [K/m].
    specific_gas_constant : float, default=287.05
        Specific gas constant R_spec [J/(kg K)] for the atmospheric composition.
    adiabatic_index : float, default=1.4
        Ratio of specific heats gamma (Cp / Cv).
    min_temperature : float, default=200.0
        Lower temperature cap T_min [K] for upper troposphere/mesosphere.
    taper_start_ratio : float, default=0.9
        Fraction of cutoff_altitude where cubic polynomial tapering begins.
    """

    sea_level_density: float = 1.225
    scale_height: float = 8500.0
    cutoff_altitude: float = 100000.0
    sea_level_temperature: float = 288.15
    lapse_rate: float = 0.0065
    specific_gas_constant: float = 287.05
    adiabatic_index: float = 1.4
    min_temperature: float = 200.0
    taper_start_ratio: float = 0.9

    def __post_init__(self) -> None:
        """Enforce 64-bit floating point precision and validate parameters."""
        self.sea_level_density = float(np.float64(self.sea_level_density))
        self.scale_height = float(np.float64(self.scale_height))
        self.cutoff_altitude = float(np.float64(self.cutoff_altitude))
        self.sea_level_temperature = float(np.float64(self.sea_level_temperature))
        self.lapse_rate = float(np.float64(self.lapse_rate))
        self.specific_gas_constant = float(np.float64(self.specific_gas_constant))
        self.adiabatic_index = float(np.float64(self.adiabatic_index))
        self.min_temperature = float(np.float64(self.min_temperature))
        self.taper_start_ratio = float(np.float64(self.taper_start_ratio))

        if self.sea_level_density < 0.0:
            raise ValueError(
                f"Sea level density must be non-negative, got {self.sea_level_density}"
            )
        if self.scale_height <= 0.0:
            raise ValueError(f"Atmospheric scale height must be positive, got {self.scale_height}")
        if self.cutoff_altitude <= 0.0:
            raise ValueError(f"Cutoff altitude must be positive, got {self.cutoff_altitude}")
        if self.sea_level_temperature <= 0.0:
            raise ValueError(
                f"Sea level temperature must be positive, got {self.sea_level_temperature}"
            )
        if self.specific_gas_constant <= 0.0:
            raise ValueError(
                f"Specific gas constant must be positive, got {self.specific_gas_constant}"
            )
        if self.adiabatic_index <= 0.0:
            raise ValueError(f"Adiabatic index must be positive, got {self.adiabatic_index}")
        if self.min_temperature <= 0.0:
            raise ValueError(f"Minimum temperature must be positive, got {self.min_temperature}")
        if not (0.0 < self.taper_start_ratio < 1.0):
            raise ValueError(
                f"Taper start ratio must be strictly between 0 and 1, got {self.taper_start_ratio}"
            )

    def density(self, altitude: float) -> float:
        """Compute atmospheric density at a given geometric altitude.

        Calculates exponential barometric density with smooth cubic Hermite tapering:
            rho(h) = rho_0 * exp(-h / H)
        For h in [h_taper_start, h_max], the density is modulated by a smoothstep cubic
        spline S(u) = (1 - u)^2 * (1 + 2u) where u = (h - h_start) / (h_max - h_start).
        This guarantees C^1 continuity and ensures density and its derivative drop
        smoothly to exactly 0.0 at cutoff_altitude.

        Parameters
        ----------
        altitude : float
            Geometric altitude above the planetary surface [m].

        Returns
        -------
        float
            Atmospheric density in kg/m^3. Returns 0.0 if altitude >= cutoff_altitude
            or if altitude < 0.0.
        """
        alt = float(np.float64(altitude))
        if alt < 0.0 or alt >= self.cutoff_altitude:
            return 0.0

        rho_baro = self.sea_level_density * math.exp(-alt / self.scale_height)
        h_taper_start = self.taper_start_ratio * self.cutoff_altitude

        if alt <= h_taper_start:
            return float(np.float64(rho_baro))

        # Cubic smoothstep falloff: S(u) = (1 - u)^2 * (1 + 2u)
        # S(0) = 1, S'(0) = 0, S(1) = 0, S'(1) = 0
        u = (alt - h_taper_start) / (self.cutoff_altitude - h_taper_start)
        s = (1.0 - u) * (1.0 - u) * (1.0 + 2.0 * u)
        return float(np.float64(rho_baro * s))

    def temperature(self, altitude: float) -> float:
        """Compute atmospheric temperature at a given geometric altitude.

        Piecewise linear tropospheric lapse rate model capped at min_temperature:
            T(h) = max(T_0 - L * max(0.0, h), T_min)

        Parameters
        ----------
        altitude : float
            Geometric altitude above the planetary surface [m].

        Returns
        -------
        float
            Atmospheric temperature in Kelvin [K].
        """
        alt = float(np.float64(altitude))
        effective_alt = max(0.0, alt)
        t_lapse = self.sea_level_temperature - self.lapse_rate * effective_alt
        return float(np.float64(max(t_lapse, self.min_temperature)))

    def speed_of_sound(self, altitude: float) -> float:
        """Compute local speed of sound at a given geometric altitude.

        Equation:
            a = sqrt(gamma * R_spec * T(h))

        Parameters
        ----------
        altitude : float
            Geometric altitude above the planetary surface [m].

        Returns
        -------
        float
            Speed of sound in meters per second [m/s].
        """
        temp = self.temperature(altitude)
        a_sq = self.adiabatic_index * self.specific_gas_constant * temp
        return float(np.float64(math.sqrt(max(0.0, a_sq))))

    def pressure(self, altitude: float) -> float:
        """Compute atmospheric static pressure using the ideal gas law.

        Equation:
            P = rho(h) * R_spec * T(h)

        Parameters
        ----------
        altitude : float
            Geometric altitude above the planetary surface [m].

        Returns
        -------
        float
            Static atmospheric pressure in Pascals [Pa]. Identically 0.0 at or above cutoff.
        """
        rho = self.density(altitude)
        if rho <= 0.0:
            return 0.0
        temp = self.temperature(altitude)
        return float(np.float64(rho * self.specific_gas_constant * temp))


# Standard Earth atmospheric model preset
EARTH_ATMOSPHERE: AtmosphereModel = AtmosphereModel()

__all__ = [
    "AtmosphereModel",
    "EARTH_ATMOSPHERE",
]
