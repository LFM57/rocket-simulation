"""Procedural 2D graphics rendering engine for Spaceflight Simulator (SFS).

Draws rockets, parts, gimbals, multi-layer exhaust plumes, atmospheric sky,
starfields, curved planetary terrain, Keplerian conic trajectories, and HUD vectors
using pure Pygame primitives with no external image dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pygame

from core.constants import g0
from core.vector2d import Vector2D
from models.parts import Decoupler, Engine, FuelTank, Part, Payload
from models.rocket import Rocket, Stage
from physics.celestial import CelestialBody
from physics.orbit import Orbit, calculate_orbit
from simulation.flight_dynamics import FlightVehicle, VehicleState
from ui.camera import Camera, CameraMode
from ui.widgets import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_RED,
    ACCENT_YELLOW,
    PANEL_BORDER,
    TEXT_MUTED,
    TEXT_WHITE,
    draw_text,
    format_altitude,
)


@dataclass
class DebrisStage:
    """Separated stage coasting under Newtonian gravity."""

    stage: Stage
    position: Vector2D
    velocity: Vector2D
    attitude: float
    angular_velocity: float
    is_active: bool = True


class Renderer:
    """Procedural 2D vector renderer for aerospace flight and orbital mechanics."""

    def __init__(self, surface: pygame.Surface) -> None:
        self.surface: pygame.Surface = surface

        # Deterministic starfield for consistent sky rendering
        self._stars: list[tuple[float, float, int, float]] = []
        self._init_starfield(count=300)

        # Active stage debris list
        self.debris: list[DebrisStage] = []

    def _init_starfield(self, count: int = 300) -> None:
        """Generate static procedural stars distributed across unit space."""
        rng = random.Random(42)
        self._stars.clear()
        for _ in range(count):
            rx = rng.random()  # [0, 1]
            ry = rng.random()  # [0, 1]
            size = 1 if rng.random() < 0.85 else 2
            brightness = rng.uniform(0.3, 1.0)
            self._stars.append((rx, ry, size, brightness))

    def set_surface(self, surface: pygame.Surface) -> None:
        """Update active rendering target surface."""
        self.surface = surface

    # =========================================================================
    # 1. Environment & Sky Rendering
    # =========================================================================

    def render_sky_and_stars(
        self,
        camera: Camera,
        vehicle: FlightVehicle,
        body: CelestialBody,
    ) -> None:
        """Render dynamic atmospheric sky background and starfield."""
        sw = self.surface.get_width()
        sh = self.surface.get_height()

        alt = max(0.0, vehicle.altitude)
        h_cutoff = (
            float(body.atmosphere.cutoff_altitude)
            if body.atmosphere is not None
            else 100_000.0
        )

        # Daylight factor: 1.0 at ground, 0.0 above Karman line (100 km)
        daylight = max(0.0, min(1.0, 1.0 - (alt / h_cutoff) ** 0.6))

        # Ground sky blue: (85, 155, 230), Space black: (10, 12, 18)
        sky_r = int(10 + (85 - 10) * daylight)
        sky_g = int(12 + (155 - 12) * daylight)
        sky_b = int(18 + (230 - 18) * daylight)
        self.surface.fill((sky_r, sky_g, sky_b))

        # Render stars when daylight drops
        star_opacity = max(0.0, 1.0 - daylight * 1.5)
        if star_opacity > 0.05:
            # Shift starfield slightly with camera rotation for subtle parallax
            ang = camera.rotation_angle
            for rx, ry, size, br in self._stars:
                # Wrap coordinates
                sx = int((rx * sw + math.cos(ang) * 40.0) % sw)
                sy = int((ry * sh + math.sin(ang) * 40.0) % sh)
                alpha = int(255 * br * star_opacity)
                star_c = (alpha, alpha, min(255, int(alpha * 1.1)))
                if size == 1:
                    self.surface.set_at((sx, sy), star_c)
                else:
                    pygame.draw.circle(self.surface, star_c, (sx, sy), 1)

    def render_terrain(
        self,
        camera: Camera,
        body: CelestialBody,
    ) -> None:
        """Render curved planetary surface and ground features."""
        sw = self.surface.get_width()
        sh = self.surface.get_height()

        # Check if we are zoomed in (close-up flight view) or zoomed out (map view)
        r_planet = float(body.radius)
        screen_radius = r_planet * camera.scale

        if screen_radius > max(sw, sh) * 3.0:
            # High-zoom close-up flight mode: sample visible horizon arc
            arc_points = camera.get_visible_horizon_points(body, num_points=70)
            screen_pts: list[tuple[float, float]] = [
                camera.world_to_screen(pt) for pt in arc_points
            ]

            # Form closed terrain polygon beneath the horizon
            poly_pts = list(screen_pts)
            # Add bottom corners
            poly_pts.append((float(sw + 100), float(sh + 100)))
            poly_pts.append((-100.0, float(sh + 100)))

            # Fill terrain: earthy dark green/brown
            pygame.draw.polygon(self.surface, (38, 54, 42), poly_pts)

            # Draw surface horizon line
            if len(screen_pts) >= 2:
                pygame.draw.lines(self.surface, (70, 115, 80), False, screen_pts, width=3)

            # Draw ground distance markers (every 500m along surface near vehicle)
            ref = (
                camera.reference_position
                if camera.reference_position is not None
                else camera.position
            )
            theta_0 = math.atan2(ref.y, ref.x)
            # Distance span in meters
            span_m = (sw * 0.6) / max(1e-6, camera.scale)
            if span_m < 50_000.0:  # Only draw markers when close to ground
                tick_step = 500.0 if span_m > 4000.0 else (100.0 if span_m > 800.0 else 20.0)
                d_theta_step = tick_step / r_planet
                num_ticks = int(span_m / tick_step) + 2

                for i in range(-num_ticks, num_ticks + 1):
                    tick_ang = theta_0 + i * d_theta_step
                    p_ground = Vector2D(
                        r_planet * math.cos(tick_ang), r_planet * math.sin(tick_ang)
                    )
                    gx, gy = camera.world_to_screen(p_ground)

                    if -50 <= gx <= sw + 50:
                        # Draw small vertical tick
                        n_ground = p_ground.normalized()
                        p_top = p_ground + n_ground * 4.0
                        tx, ty = camera.world_to_screen(p_top)
                        pygame.draw.line(self.surface, (95, 145, 105), (gx, gy), (tx, ty), width=2)

                        # Label major ticks
                        dist_val = i * tick_step
                        if abs(dist_val) < 1e-4:
                            draw_text(self.surface, "PAD", gx, gy + 14, size=11, color=(140, 190, 150), bold=True, align="center")
                        elif abs(dist_val) % (tick_step * 2) == 0:
                            lbl = f"{dist_val/1000.0:+.1f}k" if abs(dist_val) >= 1000.0 else f"{dist_val:+.0f}m"
                            draw_text(self.surface, lbl, gx, gy + 14, size=10, color=(110, 160, 120), align="center")

        else:
            # Zoomed-out map mode: draw Earth disk & atmosphere ring
            bx, by = camera.world_to_screen(body.position)
            sr = int(screen_radius)

            if sr > 1:
                # 1. Translucent atmosphere outer glow
                if body.atmosphere is not None:
                    atmo_r = int((r_planet + body.atmosphere.cutoff_altitude) * camera.scale)
                    if atmo_r > sr:
                        atmo_surf = pygame.Surface((atmo_r * 2 + 4, atmo_r * 2 + 4), pygame.SRCALPHA)
                        pygame.draw.circle(
                            atmo_surf,
                            (70, 160, 240, 75),
                            (atmo_r + 2, atmo_r + 2),
                            atmo_r,
                        )
                        self.surface.blit(atmo_surf, (bx - atmo_r - 2, by - atmo_r - 2))

                # 2. Planet body disk
                pygame.draw.circle(self.surface, (32, 90, 175), (int(bx), int(by)), sr)
                # Planet outline
                pygame.draw.circle(self.surface, (60, 130, 220), (int(bx), int(by)), sr, width=1)

    # =========================================================================
    # 2. Procedural Vehicle & Parts Rendering
    # =========================================================================

    def render_rocket(
        self,
        camera: Camera,
        vehicle: FlightVehicle,
        render_plumes: bool = True,
    ) -> None:
        """Render all active stages, parts, gimbals, and exhaust plumes."""
        rocket = vehicle.rocket
        craft_pos = vehicle.position
        theta = vehicle.attitude

        # Longitudinal heading unit vector (tail to nose)
        u_hat = Vector2D(math.cos(theta), math.sin(theta))
        # Transverse unit vector (lateral right)
        n_hat = Vector2D(-math.sin(theta), math.cos(theta))

        y_com = rocket.center_of_mass()
        alt = vehicle.altitude
        p_amb = (
            vehicle.body.atmosphere.pressure(max(0.0, alt))
            if vehicle.body.atmosphere
            else 0.0
        )

        # Render non-separated stages from bottom to top
        for stage in rocket.stages:
            if stage.is_separated:
                continue

            stage_base = stage.position_offset

            # Separate parts by type for layer rendering
            fuel_tanks: list[FuelTank] = []
            engines: list[Engine] = []
            decouplers: list[Decoupler] = []
            payloads: list[Payload] = []

            for p in stage.parts:
                if isinstance(p, Engine):
                    engines.append(p)
                elif isinstance(p, FuelTank):
                    fuel_tanks.append(p)
                elif isinstance(p, Decoupler):
                    decouplers.append(p)
                elif isinstance(p, Payload):
                    payloads.append(p)
                else:
                    fuel_tanks.append(p)  # default

            # Detect engine cluster lateral offsets
            eng_offsets = self._compute_cluster_offsets(engines)

            # 1. Render Engines & Plumes
            for eng, x_lat in zip(engines, eng_offsets):
                d_y = (stage_base + eng.position_offset) - y_com
                self._render_engine(
                    camera=camera,
                    craft_pos=craft_pos,
                    u_hat=u_hat,
                    n_hat=n_hat,
                    d_y=d_y,
                    x_lat=x_lat,
                    engine=eng,
                    p_amb=p_amb,
                    render_plume=render_plumes,
                )

            # 2. Render Fuel Tanks
            for tank in fuel_tanks:
                d_y = (stage_base + tank.position_offset) - y_com
                self._render_fuel_tank(
                    camera=camera,
                    craft_pos=craft_pos,
                    u_hat=u_hat,
                    n_hat=n_hat,
                    d_y=d_y,
                    x_lat=0.0,
                    tank=tank,
                )

            # 3. Render Decouplers
            for dec in decouplers:
                d_y = (stage_base + dec.position_offset) - y_com
                self._render_decoupler(
                    camera=camera,
                    craft_pos=craft_pos,
                    u_hat=u_hat,
                    n_hat=n_hat,
                    d_y=d_y,
                    x_lat=0.0,
                    decoupler=dec,
                )

            # 4. Render Payloads & Fairings
            for pay in payloads:
                d_y = (stage_base + pay.position_offset) - y_com
                self._render_payload(
                    camera=camera,
                    craft_pos=craft_pos,
                    u_hat=u_hat,
                    n_hat=n_hat,
                    d_y=d_y,
                    x_lat=0.0,
                    payload=pay,
                )

        # 5. Render CoM and CoP indicators if zoomed in close
        if camera.scale > 2.0:
            self._render_markers(camera, craft_pos, u_hat, rocket, y_com)

    def _compute_cluster_offsets(self, engines: list[Engine]) -> list[float]:
        """Compute lateral cluster offsets x_lat for multi-engine stages."""
        n = len(engines)
        if n <= 1:
            return [0.0]

        # For 9 engines (Falcon 9 octagol cluster):
        # 8 around perimeter, 1 in center
        if n == 9:
            r_cluster = engines[0].radius * 2.2
            # 2D projection on lateral transverse axis: [-r, +r]
            angles = [i * (2.0 * math.pi / 8) for i in range(8)]
            offsets = [r_cluster * math.sin(a) for a in angles]
            offsets.append(0.0)  # Center engine
            return offsets

        # Standard linear distribution:
        spacing = engines[0].radius * 2.1
        total_span = (n - 1) * spacing
        return [-0.5 * total_span + i * spacing for i in range(n)]

    def _render_fuel_tank(
        self,
        camera: Camera,
        craft_pos: Vector2D,
        u_hat: Vector2D,
        n_hat: Vector2D,
        d_y: float,
        x_lat: float,
        tank: FuelTank,
    ) -> None:
        """Render cylindrical fuel tank with metallic seams and propellant fill level."""
        l_half = tank.length * 0.5
        r = tank.radius

        # 4 corners in craft coordinates: (axial, lateral)
        corners = [
            (d_y - l_half, x_lat - r),
            (d_y + l_half, x_lat - r),
            (d_y + l_half, x_lat + r),
            (d_y - l_half, x_lat + r),
        ]

        screen_pts = [
            camera.world_to_screen(craft_pos + u_hat * cy + n_hat * cx)
            for cy, cx in corners
        ]

        # Tank body: sleek light grey/white metallic
        pygame.draw.polygon(self.surface, (215, 222, 232), screen_pts)
        pygame.draw.polygon(self.surface, (120, 135, 155), screen_pts, width=1)

        # Internal propellant remaining overlay
        prop_frac = (
            tank.propellant_mass / max(1e-6, tank.propellant_capacity)
            if hasattr(tank, "propellant_capacity")
            else 0.0
        )
        if 0.0 < prop_frac < 0.999:
            fill_len = tank.length * prop_frac
            fill_bottom = d_y - l_half
            fill_top = fill_bottom + fill_len
            fill_corners = [
                (fill_bottom, x_lat - r + 0.1),
                (fill_top, x_lat - r + 0.1),
                (fill_top, x_lat + r - 0.1),
                (fill_bottom, x_lat + r - 0.1),
            ]
            fill_screen_pts = [
                camera.world_to_screen(craft_pos + u_hat * cy + n_hat * cx)
                for cy, cx in fill_corners
            ]
            # Cyan propellant fill
            pygame.draw.polygon(self.surface, (0, 180, 240, 120), fill_screen_pts)

        # Ribbed horizontal structural segment lines
        if tank.length > 5.0 and camera.scale > 1.5:
            num_ribs = min(8, int(tank.length // 4.0))
            for i in range(1, num_ribs):
                rib_y = d_y - l_half + i * (tank.length / num_ribs)
                p1 = camera.world_to_screen(craft_pos + u_hat * rib_y + n_hat * (x_lat - r))
                p2 = camera.world_to_screen(craft_pos + u_hat * rib_y + n_hat * (x_lat + r))
                pygame.draw.line(self.surface, (170, 180, 195), p1, p2, width=1)

    def _render_engine(
        self,
        camera: Camera,
        craft_pos: Vector2D,
        u_hat: Vector2D,
        n_hat: Vector2D,
        d_y: float,
        x_lat: float,
        engine: Engine,
        p_amb: float,
        render_plume: bool,
    ) -> None:
        """Render engine bell nozzle rotated by current gimbal and animated exhaust plume."""
        l_val = engine.length
        r_exit = engine.radius
        r_throat = r_exit * 0.35

        # Nozzle pivots at top of engine along current_gimbal
        delta = engine.current_gimbal
        # Nozzle axis points along -u_hat rotated by -delta
        nozzle_u = u_hat.rotate(-delta)
        nozzle_n = n_hat.rotate(-delta)

        # Bell center top pivot point
        pivot_pos = craft_pos + u_hat * (d_y + l_val * 0.5) + n_hat * x_lat

        # Bell polygon vertices: throat -> exit left -> exit right -> throat right
        bell_pts_world = [
            pivot_pos + nozzle_n * (-r_throat),
            pivot_pos - nozzle_u * l_val + nozzle_n * (-r_exit),
            pivot_pos - nozzle_u * l_val + nozzle_n * r_exit,
            pivot_pos + nozzle_n * r_throat,
        ]

        bell_screen = [camera.world_to_screen(p) for p in bell_pts_world]

        # Engine bell: graphite / titanium metallic dark grey
        pygame.draw.polygon(self.surface, (60, 65, 75), bell_screen)
        pygame.draw.polygon(self.surface, (140, 145, 160), bell_screen, width=1)

        # Render active exhaust plume if engine is firing
        if render_plume and engine.is_ignited and engine.throttle > 0.0:
            exit_center = pivot_pos - nozzle_u * l_val
            self._render_exhaust_plume(
                camera=camera,
                exit_pos=exit_center,
                nozzle_dir=-nozzle_u,
                transverse_dir=nozzle_n,
                r_exit=r_exit,
                length=l_val,
                throttle=engine.throttle,
                p_amb=p_amb,
            )

    def _render_exhaust_plume(
        self,
        camera: Camera,
        exit_pos: Vector2D,
        nozzle_dir: Vector2D,
        transverse_dir: Vector2D,
        r_exit: float,
        length: float,
        throttle: float,
        p_amb: float,
    ) -> None:
        """Render multi-layer animated flame plume with ambient pressure expansion."""
        # Plume length scales with throttle
        plume_len = length * (1.8 + 4.5 * throttle)

        # Ambient pressure expansion factor:
        # Sea-level (101325 Pa): tight column (exp = 1.0)
        # Vacuum (0 Pa): underexpanded wide plume (exp = 3.2)
        p_norm = max(0.0, min(1.0, p_amb / 101325.0))
        expansion = 1.0 + 2.2 * (1.0 - p_norm)

        # Flame tip flicker
        ticks = pygame.time.get_ticks()
        flicker = 1.0 + 0.08 * math.sin(ticks * 0.04)

        tip_pos = exit_pos + nozzle_dir * (plume_len * flicker)
        mid_pos = exit_pos + nozzle_dir * (plume_len * 0.45 * flicker)
        mid_r = r_exit * expansion * 1.5

        # 1. Outer translucent orange/yellow flame
        outer_pts = [
            camera.world_to_screen(exit_pos + transverse_dir * (-r_exit)),
            camera.world_to_screen(mid_pos + transverse_dir * (-mid_r)),
            camera.world_to_screen(tip_pos),
            camera.world_to_screen(mid_pos + transverse_dir * mid_r),
            camera.world_to_screen(exit_pos + transverse_dir * r_exit),
        ]
        pygame.draw.polygon(self.surface, (255, 130, 20), outer_pts)

        # 2. Mid high-energy bright orange/yellow flame
        mid_core_len = plume_len * 0.75 * flicker
        mid_core_r = r_exit * expansion * 0.9
        mid_pts = [
            camera.world_to_screen(exit_pos + transverse_dir * (-r_exit * 0.8)),
            camera.world_to_screen(exit_pos + nozzle_dir * (mid_core_len * 0.4) + transverse_dir * (-mid_core_r)),
            camera.world_to_screen(exit_pos + nozzle_dir * mid_core_len),
            camera.world_to_screen(exit_pos + nozzle_dir * (mid_core_len * 0.4) + transverse_dir * mid_core_r),
            camera.world_to_screen(exit_pos + transverse_dir * (r_exit * 0.8)),
        ]
        pygame.draw.polygon(self.surface, (255, 210, 40), mid_pts)

        # 3. Inner supersonic white shock core
        inner_core_len = plume_len * 0.35 * flicker
        inner_pts = [
            camera.world_to_screen(exit_pos + transverse_dir * (-r_exit * 0.45)),
            camera.world_to_screen(exit_pos + nozzle_dir * inner_core_len),
            camera.world_to_screen(exit_pos + transverse_dir * (r_exit * 0.45)),
        ]
        pygame.draw.polygon(self.surface, (245, 252, 255), inner_pts)

    def _render_decoupler(
        self,
        camera: Camera,
        craft_pos: Vector2D,
        u_hat: Vector2D,
        n_hat: Vector2D,
        d_y: float,
        x_lat: float,
        decoupler: Decoupler,
    ) -> None:
        """Render interstage decoupler band with hazard stripes."""
        l_half = decoupler.length * 0.5
        r = decoupler.radius

        corners = [
            (d_y - l_half, x_lat - r),
            (d_y + l_half, x_lat - r),
            (d_y + l_half, x_lat + r),
            (d_y - l_half, x_lat + r),
        ]
        screen_pts = [
            camera.world_to_screen(craft_pos + u_hat * cy + n_hat * cx)
            for cy, cx in corners
        ]

        # Dark interstage base
        pygame.draw.polygon(self.surface, (50, 56, 68), screen_pts)
        pygame.draw.polygon(self.surface, (90, 100, 120), screen_pts, width=1)

        # Yellow/black hazard stripe overlay if visible
        if camera.scale > 2.0:
            mid_y = d_y
            p1 = camera.world_to_screen(craft_pos + u_hat * mid_y + n_hat * (x_lat - r))
            p2 = camera.world_to_screen(craft_pos + u_hat * mid_y + n_hat * (x_lat + r))
            pygame.draw.line(self.surface, ACCENT_YELLOW, p1, p2, width=2)

    def _render_payload(
        self,
        camera: Camera,
        craft_pos: Vector2D,
        u_hat: Vector2D,
        n_hat: Vector2D,
        d_y: float,
        x_lat: float,
        payload: Payload,
    ) -> None:
        """Render payload fairing / crew capsule with aerodynamic nose cone."""
        l_val = payload.length
        l_half = l_val * 0.5
        r = payload.radius

        # Aerodynamic ogive nose cone polygon
        # Base left -> Cylindrical mid left -> Nose tip -> Cylindrical mid right -> Base right
        y_base = d_y - l_half
        y_shoulder = d_y + l_half * 0.2
        y_tip = d_y + l_half

        pts = [
            camera.world_to_screen(craft_pos + u_hat * y_base + n_hat * (x_lat - r)),
            camera.world_to_screen(craft_pos + u_hat * y_shoulder + n_hat * (x_lat - r)),
            camera.world_to_screen(craft_pos + u_hat * y_tip + n_hat * x_lat),
            camera.world_to_screen(craft_pos + u_hat * y_shoulder + n_hat * (x_lat + r)),
            camera.world_to_screen(craft_pos + u_hat * y_base + n_hat * (x_lat + r)),
        ]

        pygame.draw.polygon(self.surface, (230, 236, 245), pts)
        pygame.draw.polygon(self.surface, (130, 140, 160), pts, width=1)

        # Window dot on capsule
        if camera.scale > 2.5:
            win_pos = camera.world_to_screen(
                craft_pos + u_hat * (d_y + l_half * 0.1) + n_hat * x_lat
            )
            pygame.draw.circle(self.surface, (30, 40, 55), (int(win_pos[0]), int(win_pos[1])), 3)

    def _render_markers(
        self,
        camera: Camera,
        craft_pos: Vector2D,
        u_hat: Vector2D,
        rocket: Rocket,
        y_com: float,
    ) -> None:
        """Render Center of Mass (yellow/black) and Center of Pressure (blue/white) markers."""
        # CoM marker is at dynamic y_com (offset 0 from craft_pos)
        com_screen = camera.world_to_screen(craft_pos)
        cx, cy = int(com_screen[0]), int(com_screen[1])

        # Draw CoM crosshair circle (yellow / black)
        r_m = 6
        pygame.draw.circle(self.surface, (20, 20, 20), (cx, cy), r_m)
        pygame.draw.circle(self.surface, ACCENT_YELLOW, (cx, cy), r_m, width=2)
        pygame.draw.line(self.surface, ACCENT_YELLOW, (cx - r_m, cy), (cx + r_m, cy), width=1)
        pygame.draw.line(self.surface, ACCENT_YELLOW, (cx, cy - r_m), (cx, cy + r_m), width=1)

        # CoP marker
        y_cop = rocket.center_of_pressure()
        d_cop = y_cop - y_com
        cop_screen = camera.world_to_screen(craft_pos + u_hat * d_cop)
        px, py = int(cop_screen[0]), int(cop_screen[1])

        # Draw CoP bullseye circle (cyan / white)
        pygame.draw.circle(self.surface, ACCENT_CYAN, (px, py), r_m, width=2)
        pygame.draw.circle(self.surface, (255, 255, 255), (px, py), 2)

    # =========================================================================
    # 3. Separated Debris Simulation & Rendering
    # =========================================================================

    def add_debris(
        self,
        stage: Stage,
        position: Vector2D,
        velocity: Vector2D,
        attitude: float,
        omega: float,
    ) -> None:
        """Track discarded stage as coasting visual debris."""
        debris_item = DebrisStage(
            stage=stage,
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=omega,
        )
        self.debris.append(debris_item)

    def update_debris(self, dt: float, body: CelestialBody) -> None:
        """Advance discarded stage debris under 2-body Newtonian gravity."""
        dt_val = float(dt)
        active_list: list[DebrisStage] = []

        for deb in self.debris:
            # 2-body gravitational acceleration
            a_grav = body.gravitational_acceleration_at(deb.position)
            deb.velocity = deb.velocity + a_grav * dt_val
            deb.position = deb.position + deb.velocity * dt_val
            deb.attitude += deb.angular_velocity * dt_val

            # Remove upon surface impact
            if body.altitude(deb.position) > 0.0:
                active_list.append(deb)

        self.debris = active_list

    def render_debris(self, camera: Camera) -> None:
        """Render all active separated stages tumbling away."""
        for deb in self.debris:
            u_hat = Vector2D(math.cos(deb.attitude), math.sin(deb.attitude))
            n_hat = Vector2D(-math.sin(deb.attitude), math.cos(deb.attitude))

            for p in deb.stage.parts:
                d_y = p.position_offset
                if isinstance(p, FuelTank):
                    self._render_fuel_tank(
                        camera=camera,
                        craft_pos=deb.position,
                        u_hat=u_hat,
                        n_hat=n_hat,
                        d_y=d_y,
                        x_lat=0.0,
                        tank=p,
                    )
                elif isinstance(p, Engine):
                    self._render_engine(
                        camera=camera,
                        craft_pos=deb.position,
                        u_hat=u_hat,
                        n_hat=n_hat,
                        d_y=d_y,
                        x_lat=0.0,
                        engine=p,
                        p_amb=0.0,
                        render_plume=False,
                    )

    # =========================================================================
    # 4. Flight Vector Indicators (Prograde, Retrograde, Wind, Heading)
    # =========================================================================

    def render_flight_vectors(
        self,
        camera: Camera,
        vehicle: FlightVehicle,
    ) -> None:
        """Draw HUD flight direction markers around vehicle CoM."""
        cx, cy = camera.world_to_screen(vehicle.position)
        sw = self.surface.get_width()
        sh = self.surface.get_height()

        # Only draw if craft is inside screen bounds
        if not (-50 <= cx <= sw + 50 and -50 <= cy <= sh + 50):
            return

        r_hud = 42.0

        # 1. Heading vector (white tick)
        u_screen_ang = camera.world_vector_to_screen_angle(vehicle.heading_vector)
        hx = cx + math.cos(u_screen_ang) * r_hud
        hy = cy + math.sin(u_screen_ang) * r_hud
        pygame.draw.line(self.surface, (255, 255, 255), (cx, cy), (hx, hy), width=2)
        pygame.draw.circle(self.surface, (255, 255, 255), (int(hx), int(hy)), 3)

        # 2. Prograde & Retrograde vectors (if speed > 2 m/s)
        spd = vehicle.speed
        if spd > 2.0:
            v_dir = vehicle.velocity.normalized()
            v_screen_ang = camera.world_vector_to_screen_angle(v_dir)

            # Prograde (green circle with ticks)
            px = cx + math.cos(v_screen_ang) * r_hud
            py = cy + math.sin(v_screen_ang) * r_hud
            pygame.draw.circle(self.surface, ACCENT_GREEN, (int(px), int(py)), 5, width=1)
            pygame.draw.circle(self.surface, ACCENT_GREEN, (int(px), int(py)), 1)

            # Retrograde (red circle with X)
            rx = cx - math.cos(v_screen_ang) * r_hud
            ry = cy - math.sin(v_screen_ang) * r_hud
            pygame.draw.circle(self.surface, ACCENT_RED, (int(rx), int(ry)), 5, width=1)
            pygame.draw.line(self.surface, ACCENT_RED, (rx - 3, ry - 3), (rx + 3, ry + 3), width=1)
            pygame.draw.line(self.surface, ACCENT_RED, (rx - 3, ry + 3), (rx + 3, ry - 3), width=1)

    # =========================================================================
    # 5. Real-Time Keplerian Orbit Conic Trajectory Rendering
    # =========================================================================

    def render_orbit_trajectory(
        self,
        camera: Camera,
        vehicle: FlightVehicle,
        body: CelestialBody,
    ) -> None:
        """Analytically compute and draw predicted Keplerian orbital trajectory conic."""
        try:
            orbit = calculate_orbit(vehicle.position, vehicle.velocity, body, warn=False)
        except Exception:
            return

        e = orbit.eccentricity
        a = orbit.semi_major_axis
        h = orbit.specific_angular_momentum
        r_planet = float(body.radius)

        # Perifocal unit vectors: p_hat points to periapsis, q_hat perpendicular in plane
        if e > 1e-6:
            p_hat = orbit.eccentricity_vector.normalized()
        else:
            p_hat = Vector2D(1.0, 0.0)

        # Prograde / Retrograde orbital direction
        q_hat = (
            Vector2D(-p_hat.y, p_hat.x)
            if h >= 0.0
            else Vector2D(p_hat.y, -p_hat.x)
        )

        body_pos = body.position
        screen_points: list[tuple[float, float]] = []

        # =====================================================================
        # Case A: Closed Elliptic Orbit (e < 1.0)
        # =====================================================================
        if e < 1.0 and a > 0.0:
            num_samples = 140
            angles_e = np.linspace(0.0, 2.0 * math.pi, num_samples, endpoint=True)
            b = a * math.sqrt(max(0.0, 1.0 - e * e))

            for E_val in angles_e:
                # Perifocal coordinates
                x_p = a * (math.cos(E_val) - e)
                y_p = b * math.sin(E_val)
                r_world = body_pos + p_hat * x_p + q_hat * y_p
                screen_points.append(camera.world_to_screen(r_world))

            # Draw trajectory loop
            # Suborbital (periapsis < surface): amber/orange; Orbital: cyan
            traj_color = ACCENT_CYAN if orbit.h_p >= 0.0 else ACCENT_ORANGE
            if len(screen_points) >= 3:
                pygame.draw.lines(self.surface, traj_color, True, screen_points, width=2)

        # =====================================================================
        # Case B: Open Hyperbolic Trajectory (e > 1.0)
        # =====================================================================
        elif e > 1.0:
            nu_inf = math.acos(-1.0 / e)
            nu_max = min(nu_inf * 0.95, math.radians(160.0))
            num_samples = 100
            angles_nu = np.linspace(-nu_max, nu_max, num_samples)

            for nu in angles_nu:
                denom = 1.0 + e * math.cos(nu)
                if denom > 1e-5:
                    r_val = abs(a * (1.0 - e * e)) / denom
                    r_world = body_pos + p_hat * (r_val * math.cos(nu)) + q_hat * (r_val * math.sin(nu))
                    screen_points.append(camera.world_to_screen(r_world))

            if len(screen_points) >= 2:
                pygame.draw.lines(self.surface, ACCENT_YELLOW, False, screen_points, width=2)

        # =====================================================================
        # Node Markers: Periapsis (Pe) & Apoapsis (Ap)
        # =====================================================================
        # 1. Periapsis
        pe_world = body_pos + p_hat * orbit.periapsis_radius
        pe_x, pe_y = camera.world_to_screen(pe_world)
        pe_color = ACCENT_CYAN if orbit.h_p >= 0.0 else ACCENT_RED

        pygame.draw.circle(self.surface, pe_color, (int(pe_x), int(pe_y)), 5)
        pygame.draw.circle(self.surface, (255, 255, 255), (int(pe_x), int(pe_y)), 2)
        pe_lbl = f"Pe: {format_altitude(orbit.h_p)}"
        draw_text(self.surface, pe_lbl, pe_x + 8, pe_y - 8, size=11, color=pe_color, bold=True)

        # 2. Apoapsis (only for elliptic orbits)
        if e < 1.0 and not math.isinf(orbit.apoapsis_radius):
            ap_world = body_pos - p_hat * orbit.apoapsis_radius
            ap_x, ap_y = camera.world_to_screen(ap_world)
            pygame.draw.circle(self.surface, ACCENT_CYAN, (int(ap_x), int(ap_y)), 5)
            pygame.draw.circle(self.surface, (255, 255, 255), (int(ap_x), int(ap_y)), 2)
            ap_lbl = f"Ap: {format_altitude(orbit.h_a)}"
            draw_text(self.surface, ap_lbl, ap_x + 8, ap_y - 8, size=11, color=ACCENT_CYAN, bold=True)

        # 3. Spacecraft Node on Map
        craft_sx, craft_sy = camera.world_to_screen(vehicle.position)
        pygame.draw.circle(self.surface, ACCENT_GREEN, (int(craft_sx), int(craft_sy)), 6)
        pygame.draw.circle(self.surface, (255, 255, 255), (int(craft_sx), int(craft_sy)), 3)


__all__ = [
    "DebrisStage",
    "Renderer",
]
