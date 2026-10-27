"""Planar 3-DOF flight dynamics solver and FlightVehicle simulation engine.

This module couples:
    - 2D Newtonian gravitation around rotating celestial bodies.
    - Pressure-dependent gimbaled rocket thrust and mass flow depletion.
    - Compressible aerodynamic drag, lift, and restoring torque.
    - Ground contact, surface co-rotation clamps, and crash impact detection.
    - Automated PID / SAS attitude control loop integration.
    - Keplerian orbital mechanics telemetry tracking.

All calculations enforce 64-bit floating point precision (np.float64 / IEEE 754 double).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Union

import numpy as np

from core.integrator import make_state, rk4_step, wrap_angle_pi
from core.vector2d import Vector2D
from models.parts import Engine
from models.rocket import Rocket, Stage
from physics.aerodynamics import AeroState, calculate_aero_forces
from physics.celestial import CelestialBody
from physics.orbit import calculate_orbit
from simulation.autopilot import AutopilotSAS, SASMode


class VehicleState(str, Enum):
    """Operational flight dynamics state of the vehicle."""

    ON_PAD = "ON_PAD"
    IN_FLIGHT = "IN_FLIGHT"
    LANDED = "LANDED"
    DESTROYED = "DESTROYED"


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Telemetry snapshot captured at a discrete simulation epoch.

    Attributes
    ----------
    time : float
        Elapsed simulation time in seconds [s].
    altitude : float
        Geometric altitude above mean spherical surface [m].
    speed : float
        Inertial speed magnitude ||v|| [m/s].
    mach : float
        Flight Mach number relative to local atmosphere.
    dynamic_pressure : float
        Dynamic pressure q = 0.5 * rho * v_rel^2 [Pa].
    thrust : float
        Instantaneous total thrust magnitude [N].
    mass : float
        Total vehicle mass [kg].
    pitch : float
        Body attitude angle theta relative to +X axis in radians [rad].
    angle_of_attack : float
        Angle of attack alpha in radians [rad].
    apoapsis : float
        Keplerian orbit apoapsis altitude above surface [m] (inf if parabolic/hyperbolic).
    periapsis : float
        Keplerian orbit periapsis altitude above surface [m].
    stage : int
        Currently active stage index.
    """

    time: float
    altitude: float
    speed: float
    mach: float
    dynamic_pressure: float
    thrust: float
    mass: float
    pitch: float
    angle_of_attack: float
    apoapsis: float
    periapsis: float
    stage: int
    vehicle_state: Optional[Union[str, VehicleState]] = None
    local_pitch: float = 0.0

    @property
    def state(self) -> Optional[Union[str, VehicleState]]:
        """Alias for vehicle_state."""
        return self.vehicle_state

    @property
    def attitude(self) -> float:
        """Alias for pitch in radians [rad]."""
        return self.pitch

    @property
    def pitch_deg(self) -> float:
        """Body attitude angle in degrees [deg]."""
        return float(np.degrees(self.pitch))

    @property
    def q(self) -> float:
        """Alias for dynamic_pressure [Pa]."""
        return self.dynamic_pressure

    @property
    def aoa(self) -> float:
        """Alias for angle_of_attack [rad]."""
        return self.angle_of_attack

    @property
    def aoa_deg(self) -> float:
        """Angle of attack in degrees [deg]."""
        return float(np.degrees(self.angle_of_attack))

    @property
    def pitch_local(self) -> float:
        """Alias for pitch angle relative to local horizon [rad]."""
        return self.local_pitch

    @property
    def pitch_local_deg(self) -> float:
        """Pitch angle relative to local horizontal in degrees."""
        return float(np.degrees(self.pitch_local))


class FlightVehicle:
    """Planar 3-DOF flight vehicle simulation solver.

    Coordinates:
        - Cartesian world frame (x, y) centered at the celestial body center.
        - Attitude angle theta (rad): theta=0 points along +X, theta=pi/2 points along +Y.
        - Heading unit vector: u_hat = (cos(theta), sin(theta)).
        - Transverse unit vector: n_hat = (-sin(theta), cos(theta)).
        - Tail is at -u_hat from CoM; Nose/payload is at +u_hat from CoM.

    Parameters
    ----------
    rocket : Rocket
        Multi-stage rocket configuration and propulsion stack.
    body : CelestialBody
        Primary celestial body providing gravity and atmosphere.
    position : Union[Vector2D, Sequence[float]]
        Initial inertial position coordinates [m].
    velocity : Union[Vector2D, Sequence[float]], default=(0.0, 0.0)
        Initial inertial velocity vector [m/s].
    attitude : float, default=0.0
        Initial body attitude angle theta [rad].
    angular_velocity : float, default=0.0
        Initial angular velocity omega [rad/s].
    autopilot : Optional[AutopilotSAS], default=None
        Attitude control autopilot SAS instance. If None, a default instance is created.
    state : Optional[VehicleState], default=None
        Initial vehicle state. If None, set to ON_PAD if altitude <= 0, else IN_FLIGHT.
    reference_area : Optional[float], default=None
        Aerodynamic cross-sectional reference area A_ref [m^2]. If None, derived from parts.
    cd_axial : float, default=0.3
        Axial aerodynamic drag coefficient (zero AoA).
    cd_transverse : float, default=1.2
        Transverse aerodynamic drag coefficient (90 deg AoA).
    cl_alpha : float, default=0.0
        Lift curve slope dC_L / d(alpha).
    crash_velocity_limit : float, default=12.0
        Maximum allowable relative impact velocity for safe touchdown [m/s].
    crash_attitude_limit_deg : float, default=10.0
        Maximum allowable attitude deviation from vertical for safe touchdown [deg].
    """

    def __init__(
        self,
        rocket: Rocket,
        body: CelestialBody,
        position: Union[Vector2D, Sequence[float]],
        velocity: Union[Vector2D, Sequence[float]] = (0.0, 0.0),
        attitude: float = 0.0,
        angular_velocity: float = 0.0,
        autopilot: Optional[AutopilotSAS] = None,
        state: Optional[VehicleState] = None,
        reference_area: Optional[float] = None,
        cd_axial: float = 0.3,
        cd_transverse: float = 1.2,
        cl_alpha: float = 0.0,
        crash_velocity_limit: float = 12.0,
        crash_attitude_limit_deg: float = 10.0,
    ) -> None:
        self.rocket: Rocket = rocket
        self.body: CelestialBody = body

        self.position: Vector2D = (
            position if isinstance(position, Vector2D) else Vector2D.from_tuple(position)
        )
        self.velocity: Vector2D = (
            velocity if isinstance(velocity, Vector2D) else Vector2D.from_tuple(velocity)
        )
        self.attitude: float = float(np.float64(attitude))
        self.angular_velocity: float = float(np.float64(angular_velocity))

        self.autopilot: AutopilotSAS = (
            autopilot if autopilot is not None else AutopilotSAS(target_attitude=self.attitude)
        )

        self.time: float = 0.0
        self.reference_area: Optional[float] = (
            float(np.float64(reference_area)) if reference_area is not None else None
        )
        self.cd_axial: float = float(np.float64(cd_axial))
        self.cd_transverse: float = float(np.float64(cd_transverse))
        self.cl_alpha: float = float(np.float64(cl_alpha))
        self.crash_velocity_limit: float = float(np.float64(crash_velocity_limit))
        self.crash_attitude_limit_deg: float = float(np.float64(crash_attitude_limit_deg))

        # Determine initial vehicle state
        init_alt = self.body.altitude(self.position)
        if state is not None:
            self.state: VehicleState = (
                VehicleState(state) if isinstance(state, str) else state
            )
        elif init_alt <= 1e-4:
            self.state = VehicleState.ON_PAD
        else:
            self.state = VehicleState.IN_FLIGHT

        # If on pad, enforce surface clamp initial conditions
        if self.state == VehicleState.ON_PAD:
            self._clamp_to_surface()
            if autopilot is None:
                self.autopilot.target_attitude = self.attitude

        self._step_thrust_scale: float = 1.0
        self._firing_engines: list[Engine] = []
        self._last_step_thrust: Optional[float] = None
        self._step_cache: Optional[tuple[float, float, float, float, float, float, float]] = None

    # --- Geometric Properties & Coordinate Conventions ---

    @property
    def heading_vector(self) -> Vector2D:
        """Longitudinal heading unit vector u_hat = (cos(theta), sin(theta))."""
        return Vector2D(math.cos(self.attitude), math.sin(self.attitude))

    @property
    def u_hat(self) -> Vector2D:
        """Alias for heading_vector."""
        return self.heading_vector

    @property
    def normal_vector(self) -> Vector2D:
        """Transverse unit vector n_hat = (-sin(theta), cos(theta))."""
        return Vector2D(-math.sin(self.attitude), math.cos(self.attitude))

    @property
    def n_hat(self) -> Vector2D:
        """Alias for normal_vector."""
        return self.normal_vector

    @property
    def altitude(self) -> float:
        """Geometric altitude above mean planetary radius [m]."""
        return float(np.float64(self.body.altitude(self.position)))

    @property
    def speed(self) -> float:
        """Inertial speed magnitude [m/s]."""
        return self.velocity.magnitude()

    @property
    def pitch(self) -> float:
        """Alias for attitude in radians [rad]."""
        return self.attitude

    @pitch.setter
    def pitch(self, value: float) -> None:
        self.attitude = float(np.float64(value))

    @property
    def omega(self) -> float:
        """Alias for angular_velocity [rad/s]."""
        return self.angular_velocity

    @omega.setter
    def omega(self, value: float) -> None:
        self.angular_velocity = float(np.float64(value))

    @property
    def state_vector(self) -> np.ndarray:
        """Dynamic planar state vector S = [x, y, v_x, v_y, theta, omega]^T."""
        return make_state(
            self.position.x,
            self.position.y,
            self.velocity.x,
            self.velocity.y,
            self.attitude,
            self.angular_velocity,
        )

    @property
    def local_pitch(self) -> float:
        """Pitch angle relative to the local horizon in radians [-pi, pi].

        +pi/2 rad (+90 deg) = vertical ascent away from planet center.
        0 rad (0 deg) = horizontal flight along local horizon.
        -pi/2 rad (-90 deg) = vertical descent toward planet center.
        """
        r_mag = self.position.magnitude()
        if r_mag <= 1e-12:
            return self.attitude
        r_hat = self.position / r_mag
        t_hat = Vector2D(-r_hat.y, r_hat.x)
        u_hat = self.heading_vector
        return float(np.float64(math.atan2(u_hat.dot(r_hat), u_hat.dot(t_hat))))

    @property
    def local_pitch_deg(self) -> float:
        """Pitch angle relative to the local horizon in degrees [-180, 180]."""
        return float(np.degrees(self.local_pitch))

    # --- Auxiliary Geometric Extractor Methods ---

    def _get_reference_area(self) -> float:
        """Compute or retrieve aerodynamic frontal reference area A_ref [m^2]."""
        if self.reference_area is not None and self.reference_area > 0.0:
            return self.reference_area
        radii = [
            p.radius
            for s in self.rocket.stages
            if not s.is_separated
            for p in s.parts
            if p.radius > 0.0
        ]
        if radii:
            r_max = max(radii)
            return float(np.float64(math.pi * r_max * r_max))
        return 1.0

    def _get_transverse_area(self) -> float:
        """Compute lateral planform aerodynamic area A_trans = sum(2 * r * L) [m^2]."""
        total_lat = sum(
            p.projected_area
            for s in self.rocket.stages
            if not s.is_separated
            for p in s.parts
        )
        return float(np.float64(total_lat if total_lat > 0.0 else self._get_reference_area()))

    def _get_total_length(self) -> float:
        """Compute total longitudinal length of non-separated stages [m]."""
        extents = [
            s.position_offset + p.position_offset + p.length * 0.5
            for s in self.rocket.stages
            if not s.is_separated
            for p in s.parts
        ]
        if extents:
            return float(np.float64(max(extents)))
        return 1.0

    def _get_planet_omega(self) -> float:
        """Calculate celestial body sidereal rotation rate [rad/s]."""
        if math.isclose(self.body.rotation_period, 0.0, abs_tol=1e-15):
            return 0.0
        return float(np.float64(2.0 * math.pi / self.body.rotation_period))

    def _clamp_to_surface(self) -> None:
        """Clamp position, velocity, and attitude to planetary surface co-rotation."""
        r_mag = self.position.magnitude()
        r_hat = self.position.normalized() if r_mag > 1e-6 else Vector2D(0.0, 1.0)

        # Place vehicle at planetary surface radius R_planet
        self.position = r_hat * self.body.radius

        # Attitude stands perpendicular to the local planetary surface: theta = atan2(y, x)
        self.attitude = float(np.float64(math.atan2(self.position.y, self.position.x)))

        # Velocity matches surface atmospheric co-rotation wind speed at position
        omega_p = self._get_planet_omega()
        self.velocity = Vector2D(-omega_p * self.position.y, omega_p * self.position.x)
        self.angular_velocity = omega_p

    # --- Propulsion & Staging Interfaces ---

    def ignite(self) -> None:
        """Ignite all engines in the currently active stage."""
        if self.rocket.active_stage is not None:
            for eng in self.rocket.active_stage.engines:
                eng.ignite()

    def shutdown(self) -> None:
        """Shut down all engines in the currently active stage."""
        if self.rocket.active_stage is not None:
            for eng in self.rocket.active_stage.engines:
                eng.shutdown()

    def set_throttle(self, throttle: float) -> None:
        """Set throttle for all engines in the active stage (auto-igniting if throttle > 0)."""
        th = float(np.float64(max(0.0, min(1.0, float(throttle)))))
        if self.rocket.active_stage is not None:
            for eng in self.rocket.active_stage.engines:
                if th > 0.0 and not eng.is_ignited:
                    eng.ignite()
                eng.set_throttle(th)

    def stage(self, ignite: bool = True) -> Optional[Stage]:
        """Decouple the current active stage and activate the next stage.

        Parameters
        ----------
        ignite : bool, default=True
            Whether to automatically ignite the newly activated stage's engines.

        Returns
        -------
        Optional[Stage]
            The separated stage, or None if no stage was decoupled.
        """
        decoupled = self.rocket.stage()
        if decoupled is not None and ignite:
            if self.rocket.active_stage is not None:
                for eng in self.rocket.active_stage.engines:
                    eng.ignite()
        return decoupled

    # --- Dynamics & Derivatives ---

    def compute_derivatives(
        self,
        state_vector: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute the dynamic derivative vector dS/dt for the planar 3-DOF vehicle.

        Parameters
        ----------
        state_vector : np.ndarray
            Planar state vector S = [x, y, v_x, v_y, theta, omega]^T.
        t : float, default=0.0
            Simulation time epoch [s].

        Returns
        -------
        np.ndarray
            Derivative vector dS/dt = [v_x, v_y, a_x, a_y, omega, alpha]^T (dtype np.float64).
        """
        # Defensive check against swapped parameter order f(t, s) vs f(s, t)
        if isinstance(state_vector, (int, float, np.floating, np.integer)) and isinstance(
            t, np.ndarray
        ):
            t, state_vector = float(state_vector), t

        s = np.asarray(state_vector, dtype=np.float64)
        pos = Vector2D(s[0], s[1])
        vel = Vector2D(s[2], s[3])
        theta = float(s[4])
        omega = float(s[5])

        if self._step_cache is not None:
            m, i_zz, y_com, y_cop, ref_area, tot_length, transverse_area = self._step_cache
        else:
            m = self.rocket.total_mass()
            i_zz = self.rocket.moment_of_inertia()
            y_com = self.rocket.center_of_mass()
            y_cop = self.rocket.center_of_pressure()
            ref_area = self._get_reference_area()
            tot_length = self._get_total_length()
            transverse_area = self._get_transverse_area()

        if m <= 0.0:
            return np.zeros(6, dtype=np.float64)
        if i_zz <= 0.0:
            i_zz = 1.0

        # 1. Newtonian Gravitational Force: F_g = m * a_g(r)
        a_grav = self.body.gravitational_acceleration_at(pos)
        f_grav = a_grav * m

        # 2. Ambient Pressure
        alt = max(0.0, self.body.altitude(pos))
        p_amb = self.body.atmosphere.pressure(alt) if self.body.atmosphere else 0.0

        # 3. Gimbaled Thrust Force & Torque:
        # F_t = Vector2D(T * cos(theta + delta), T * sin(theta + delta))
        # tau_t = (y_engine - y_CoM) * T * sin(-delta)
        f_thrust_x = 0.0
        f_thrust_y = 0.0
        tau_thrust = 0.0
        active_stage = self.rocket.active_stage

        if (
            active_stage is not None
            and not active_stage.is_separated
            and self._step_thrust_scale > 0.0
        ):
            stage_base = active_stage.position_offset
            engines_to_eval = (
                self._firing_engines
                if self._firing_engines
                else [
                    p for p in active_stage.parts
                    if isinstance(p, Engine) and p.is_ignited and p.throttle > 0.0
                ]
                if active_stage.propellant_mass() > 0.0
                else []
            )

            for eng in engines_to_eval:
                t_val, _ = eng.get_thrust_and_mass_flow(p_amb)
                t_val *= self._step_thrust_scale
                if t_val <= 0.0:
                    continue
                delta = eng.current_gimbal
                # Deflecting nozzle by +delta produces reaction thrust at angle (theta - delta)
                # Pushing the tail to the right (+X when vertical) pitches the nose counter-clockwise (+tau)
                ang = theta - delta
                f_x = t_val * math.cos(ang)
                f_y = t_val * math.sin(ang)
                f_thrust_x += f_x
                f_thrust_y += f_y

                # r x F = (r_x * F_y - r_y * F_x) where r = (y_eng - y_com) * u_hat
                y_eng = stage_base + eng.position_offset
                arm = y_eng - y_com
                tau_thrust += -arm * (t_val * math.sin(delta))

        f_thrust = Vector2D(f_thrust_x, f_thrust_y)

        # 4. Aerodynamic Force & Torque relative to co-rotating atmosphere
        aero = calculate_aero_forces(
            craft_pos=pos,
            craft_vel=vel,
            attitude_angle=theta,
            body=self.body,
            reference_area=ref_area,
            length=tot_length,
            cd_axial=self.cd_axial,
            cd_transverse=self.cd_transverse,
            center_of_mass_offset=y_com,
            center_of_pressure_offset=y_cop,
            cl_alpha=self.cl_alpha,
            transverse_area=transverse_area,
        )

        # 5. Net Force & Torque
        f_total = f_grav + f_thrust + aero.force
        tau_total = tau_thrust + aero.torque

        a_x = f_total.x / m
        a_y = f_total.y / m
        alpha = tau_total / i_zz

        return np.array([vel.x, vel.y, a_x, a_y, omega, alpha], dtype=np.float64)

    # --- Simulation Step ---

    def step(
        self,
        dt: float,
        target_pitch: Optional[float] = None,
        throttle: Optional[float] = None,
    ) -> TelemetrySnapshot:
        """Advance the flight simulation by one time step dt.

        Execution Pipeline:
            1. Update throttle and target pitch commands.
            2. Advance autopilot SAS and command engine gimbal angles.
            3. Evaluate ground / launchpad clamp equilibrium and liftoff conditions.
            4. Burn propellant over dt.
            5. Propagate dynamics via 4th-Order Runge-Kutta (RK4).
            6. Evaluate terrain collision / touchdown boundary conditions.
            7. Return detailed TelemetrySnapshot.

        Parameters
        ----------
        dt : float
            Simulation time increment in seconds [s] (dt > 0).
        target_pitch : Optional[float], default=None
            Commanded attitude pitch angle theta_target [rad].
        throttle : Optional[float], default=None
            Engine throttle setting [0.0, 1.0].

        Returns
        -------
        TelemetrySnapshot
            Complete telemetry recording at the end of the time step.
        """
        dt_val = float(np.float64(dt))
        if dt_val <= 0.0:
            raise ValueError(f"Simulation time step dt must be positive, got {dt}")

        # 1. Update guidance and throttle controls
        if target_pitch is not None:
            self.autopilot.target_attitude = float(np.float64(target_pitch))
            if self.autopilot.mode == SASMode.OFF:
                self.autopilot.mode = SASMode.HOLD_ATTITUDE

        if throttle is not None:
            self.set_throttle(throttle)

        # 2. Advance autopilot SAS (updates engine gimbal deflection)
        delta_cmd = self.autopilot.update(
            theta=self.attitude,
            omega=self.angular_velocity,
            dt=dt_val,
            velocity=self.velocity,
        )
        if self.rocket.active_stage is not None:
            for eng in self.rocket.active_stage.engines:
                eng.set_gimbal(delta_cmd)

        omega_p = self._get_planet_omega()

        # 3. State-specific propagation
        if self.state == VehicleState.ON_PAD:
            # Check liftoff condition: F_thrust . r_hat > m * g_surface
            r_mag = self.position.magnitude()
            r_hat = self.position / r_mag if r_mag > 0.0 else Vector2D(0.0, 1.0)
            p_sl = self.body.atmosphere.pressure(0.0) if self.body.atmosphere else 0.0

            # Calculate instantaneous net thrust vector
            f_thrust_vec = Vector2D(0.0, 0.0)
            if self.rocket.active_stage is not None:
                for eng in self.rocket.active_stage.engines:
                    if eng.is_ignited and eng.throttle > 0.0:
                        t_e, _ = eng.get_thrust_and_mass_flow(p_sl)
                        ang = self.attitude - eng.current_gimbal
                        f_thrust_vec += Vector2D(t_e * math.cos(ang), t_e * math.sin(ang))

            radial_thrust = f_thrust_vec.dot(r_hat)
            weight = self.rocket.total_mass() * self.body.surface_gravity

            if radial_thrust > weight:
                # Liftoff transition!
                self.state = VehicleState.IN_FLIGHT
            else:
                # Clamp remains active: co-rotate with planet
                self.rocket.burn_fuel(dt_val, p_sl)
                d_phi = omega_p * dt_val
                self.position = self.position.rotate(d_phi).normalized() * self.body.radius
                self.velocity = self.body.wind_velocity_at(self.position)
                self.attitude = float(np.float64(math.atan2(self.position.y, self.position.x)))
                self.angular_velocity = omega_p
                if target_pitch is None:
                    self.autopilot.target_attitude = self.attitude
                self.time += dt_val

        if self.state == VehicleState.IN_FLIGHT:
            alt_cur = max(0.0, self.body.altitude(self.position))
            p_amb = self.body.atmosphere.pressure(alt_cur) if self.body.atmosphere else 0.0

            # Calculate nominal thrust to determine fractional burn factor on burnout
            nom_thrust = 0.0
            firing_engines: list[Engine] = []
            if self.rocket.active_stage is not None and not self.rocket.active_stage.is_separated:
                for eng in self.rocket.active_stage.engines:
                    if eng.is_ignited and eng.throttle > 0.0:
                        t_e, _ = eng.get_thrust_and_mass_flow(p_amb)
                        nom_thrust += t_e
                        firing_engines.append(eng)

            actual_thrust, _, _ = self.rocket.burn_fuel(dt_val, p_amb)
            self._step_thrust_scale = (actual_thrust / nom_thrust) if nom_thrust > 0.0 else 0.0
            self._firing_engines = firing_engines if self._step_thrust_scale > 0.0 else []
            self._last_step_thrust = actual_thrust

            # Cache structural properties that are invariant over the 4 RK4 derivative evaluations
            m = self.rocket.total_mass()
            i_zz = self.rocket.moment_of_inertia()
            y_com = self.rocket.center_of_mass()
            y_cop = self.rocket.center_of_pressure()
            ref_area = self._get_reference_area()
            tot_length = self._get_total_length()
            transverse_area = self._get_transverse_area()
            self._step_cache = (m, i_zz, y_com, y_cop, ref_area, tot_length, transverse_area)

            # RK4 Integration of planar 3-DOF equations of motion
            next_s = rk4_step(
                lambda t_s, s_vec: self.compute_derivatives(s_vec, t_s),
                self.time,
                self.state_vector,
                dt_val,
            )

            self._step_thrust_scale = 1.0  # Reset scale
            self._firing_engines = []
            self._step_cache = None
            self.time += dt_val
            self.position = Vector2D(next_s[0], next_s[1])
            self.velocity = Vector2D(next_s[2], next_s[3])
            self.attitude = float(next_s[4])
            self.angular_velocity = float(next_s[5])

            # 4. Ground collision / touchdown check
            alt_new = self.altitude
            if alt_new <= 0.0:
                r_hat = self.position.normalized()
                self.position = r_hat * self.body.radius
                v_surface = self.body.wind_velocity_at(self.position)
                v_rel = (self.velocity - v_surface).magnitude()

                # Attitude deviation from surface vertical
                vert_angle = math.atan2(self.position.y, self.position.x)
                att_deviation = abs(wrap_angle_pi(self.attitude - vert_angle))

                if (
                    v_rel > self.crash_velocity_limit
                    or att_deviation > math.radians(self.crash_attitude_limit_deg)
                ):
                    self.state = VehicleState.DESTROYED
                    self.velocity = Vector2D(0.0, 0.0)
                    self.angular_velocity = 0.0
                    self.shutdown()
                else:
                    self.state = VehicleState.LANDED
                    self.velocity = v_surface
                    self.angular_velocity = omega_p
                    self.attitude = vert_angle
                    self.shutdown()

        elif self.state == VehicleState.LANDED:
            # Co-rotate with planetary surface
            d_phi = omega_p * dt_val
            self.position = self.position.rotate(d_phi).normalized() * self.body.radius
            self.velocity = self.body.wind_velocity_at(self.position)
            self.attitude = float(np.float64(math.atan2(self.position.y, self.position.x)))
            self.angular_velocity = omega_p
            self.time += dt_val

        elif self.state == VehicleState.DESTROYED:
            self.time += dt_val

        # 5. Compile and return TelemetrySnapshot
        return self.get_telemetry_snapshot()

    def get_telemetry_snapshot(self) -> TelemetrySnapshot:
        """Compile and return current TelemetrySnapshot.

        Returns
        -------
        TelemetrySnapshot
            Snapshot of the current flight state.
        """
        alt_report = self.body.altitude(self.position)
        if self.state in (VehicleState.ON_PAD, VehicleState.LANDED, VehicleState.DESTROYED):
            alt_report = max(0.0, alt_report)

        p_ambient = (
            self.body.atmosphere.pressure(max(0.0, alt_report))
            if self.body.atmosphere
            else 0.0
        )
        aero = calculate_aero_forces(
            craft_pos=self.position,
            craft_vel=self.velocity,
            attitude_angle=self.attitude,
            body=self.body,
            reference_area=self._get_reference_area(),
            length=self._get_total_length(),
            cd_axial=self.cd_axial,
            cd_transverse=self.cd_transverse,
            center_of_mass_offset=self.rocket.center_of_mass(),
            center_of_pressure_offset=self.rocket.center_of_pressure(),
            cl_alpha=self.cl_alpha,
            transverse_area=self._get_transverse_area(),
        )

        orbit = calculate_orbit(self.position, self.velocity, self.body)

        if self._last_step_thrust is not None:
            current_thrust = self._last_step_thrust
            self._last_step_thrust = None
        else:
            current_thrust = 0.0
            if (
                self.state in (VehicleState.ON_PAD, VehicleState.IN_FLIGHT)
                and self.rocket.active_stage is not None
                and not self.rocket.active_stage.is_separated
                and self.rocket.active_stage.propellant_mass() > 0.0
            ):
                for eng in self.rocket.active_stage.engines:
                    if eng.is_ignited and eng.throttle > 0.0:
                        t_val, _ = eng.get_thrust_and_mass_flow(p_ambient)
                        current_thrust += t_val

        return TelemetrySnapshot(
            time=float(np.float64(self.time)),
            altitude=float(np.float64(alt_report)),
            speed=float(np.float64(self.velocity.magnitude())),
            mach=float(np.float64(aero.mach)),
            dynamic_pressure=float(np.float64(aero.dynamic_pressure)),
            thrust=float(np.float64(current_thrust)),
            mass=float(np.float64(self.rocket.total_mass())),
            pitch=float(np.float64(self.attitude)),
            angle_of_attack=float(np.float64(aero.angle_of_attack)),
            apoapsis=float(np.float64(orbit.apoapsis_altitude)),
            periapsis=float(np.float64(orbit.periapsis_altitude)),
            stage=int(self.rocket.active_stage_index),
            vehicle_state=self.state,
            local_pitch=float(np.float64(self.local_pitch)),
        )


__all__ = [
    "FlightVehicle",
    "TelemetrySnapshot",
    "VehicleState",
]
