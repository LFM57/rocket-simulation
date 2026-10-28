"""Flight View, Orbital Map View, and Post-Flight Telemetry Modal.

Integrates real-time 3-DOF flight simulation, dual-mode physics/on-rails time warp,
procedural vector rendering, HUD controls (throttle slider, SAS selector, staging),
digital telemetry gauges, and post-flight mission performance graphs.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Optional

import pygame

from core.constants import g0
from core.vector2d import Vector2D
from models.presets import create_falcon_9
from models.rocket import Rocket
from physics.celestial import CelestialBody, get_earth
from simulation.autopilot import SASMode
from simulation.flight_dynamics import FlightVehicle, TelemetrySnapshot, VehicleState
from simulation.simulation import Simulation, WarpMode
from simulation.telemetry import TelemetryLogger
from ui.camera import Camera, CameraMode
from ui.renderer import Renderer
from ui.widgets import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_RED,
    ACCENT_YELLOW,
    BUTTON_ACTIVE,
    BUTTON_NORMAL,
    PANEL_BG,
    PANEL_BORDER,
    PANEL_BORDER_LIGHT,
    Button,
    Panel,
    PlotCanvas,
    ProgressBar,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_WHITE,
    ToastManager,
    VerticalSlider,
    draw_text,
    format_altitude,
    format_mass,
    format_speed,
    format_thrust,
    format_time,
    get_font,
)


class TelemetryModal:
    """Post-flight performance summary modal with 4 compact time-series plots."""

    def __init__(
        self,
        surface: pygame.Surface,
        logger: TelemetryLogger,
        final_state: VehicleState,
        on_revert: Callable[[], None],
        on_vab: Callable[[], None],
        on_resume: Optional[Callable[[], None]] = None,
    ) -> None:
        self.surface: pygame.Surface = surface
        self.logger: TelemetryLogger = logger
        self.final_state: VehicleState = final_state
        self.on_revert: Callable[[], None] = on_revert
        self.on_vab: Callable[[], None] = on_vab
        self.on_resume: Optional[Callable[[], None]] = on_resume

        # 4 Telemetry plots
        self.plot_alt: PlotCanvas = PlotCanvas(pygame.Rect(0, 0, 10, 10), "Altitude", unit="m", line_color=ACCENT_CYAN)
        self.plot_spd: PlotCanvas = PlotCanvas(pygame.Rect(0, 0, 10, 10), "Speed", unit="m/s", line_color=ACCENT_GREEN)
        self.plot_q: PlotCanvas = PlotCanvas(pygame.Rect(0, 0, 10, 10), "Dynamic Pressure (q)", unit="Pa", line_color=ACCENT_ORANGE)
        self.plot_g: PlotCanvas = PlotCanvas(pygame.Rect(0, 0, 10, 10), "G-Force", unit="g0", line_color=ACCENT_RED)

        self.buttons: list[Button] = []
        self.modal_rect: pygame.Rect = pygame.Rect(0, 0, 10, 10)
        self.toast_msg: str = ""
        self.toast_timer: float = 0.0

        self.update_layout(surface.get_width(), surface.get_height())

    def update_layout(self, width: int, height: int) -> None:
        """Update modal and plot dimensions on window resize."""
        mw = min(1000, width - 60)
        mh = min(620, height - 60)
        mx = (width - mw) // 2
        my = (height - mh) // 2
        self.modal_rect = pygame.Rect(mx, my, mw, mh)

        # Plot grid layout (2 rows x 2 columns)
        grid_x = mx + 20
        grid_y = my + 130
        col_w = (mw - 50) // 2
        row_h = (mh - 220) // 2

        self.plot_alt.rect = pygame.Rect(grid_x, grid_y, col_w, row_h)
        self.plot_spd.rect = pygame.Rect(grid_x + col_w + 10, grid_y, col_w, row_h)
        self.plot_q.rect = pygame.Rect(grid_x, grid_y + row_h + 10, col_w, row_h)
        self.plot_g.rect = pygame.Rect(grid_x + col_w + 10, grid_y + row_h + 10, col_w, row_h)

        # Action buttons along bottom
        self.buttons.clear()
        by = self.modal_rect.bottom - 46
        bx = self.modal_rect.left + 20

        b_csv = Button(
            rect=pygame.Rect(bx, by, 160, 32),
            text="Export Telemetry CSV",
            on_click=self._export_csv,
            font_size=12,
        )
        bx += 170

        b_revert = Button(
            rect=pygame.Rect(bx, by, 140, 32),
            text="Revert to Launch",
            shortcut="[R]",
            on_click=self.on_revert,
            font_size=12,
        )
        bx += 150

        b_vab = Button(
            rect=pygame.Rect(bx, by, 140, 32),
            text="Return to VAB",
            shortcut="[ESC]",
            on_click=self.on_vab,
            font_size=12,
        )
        bx += 150

        self.buttons = [b_csv, b_revert, b_vab]
        if self.on_resume is not None and self.final_state not in (VehicleState.LANDED, VehicleState.DESTROYED):
            b_resume = Button(
                rect=pygame.Rect(self.modal_rect.right - 140, by, 120, 32),
                text="Resume Flight",
                bg_color=ACCENT_GREEN,
                text_color=(10, 20, 15),
                on_click=self.on_resume,
                font_size=12,
            )
            self.buttons.append(b_resume)

    def _export_csv(self) -> None:
        """Export telemetry buffer to local CSV."""
        try:
            p = Path("telemetry.csv")
            self.logger.to_csv(p)
            self.toast_msg = "Saved telemetry to ./telemetry.csv"
            self.toast_timer = 3.0
        except Exception as e:
            self.toast_msg = f"Export failed: {e}"
            self.toast_timer = 3.0

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle clicks inside modal."""
        for btn in self.buttons:
            if btn.handle_event(event):
                return True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.on_revert()
                return True
            elif event.key == pygame.K_ESCAPE:
                if self.on_resume is not None and self.final_state not in (VehicleState.LANDED, VehicleState.DESTROYED):
                    self.on_resume()
                else:
                    self.on_vab()
                return True

        return True  # Consume all events while modal is open

    def update(self, dt: float) -> None:
        """Update toast timer."""
        if self.toast_timer > 0.0:
            self.toast_timer = max(0.0, self.toast_timer - dt)
            if self.toast_timer == 0.0:
                self.toast_msg = ""

    def render(self) -> None:
        """Render darkened backdrop, summary grid, and 4 time-series plots."""
        sw = self.surface.get_width()
        sh = self.surface.get_height()

        # Dim background
        dim_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
        dim_surf.fill((10, 12, 18, 200))
        self.surface.blit(dim_surf, (0, 0))

        # Modal window container
        pygame.draw.rect(self.surface, PANEL_BG, self.modal_rect, border_radius=10)
        pygame.draw.rect(self.surface, PANEL_BORDER_LIGHT, self.modal_rect, width=1, border_radius=10)

        # Header Title & State Badge
        draw_text(self.surface, "MISSION TELEMETRY SUMMARY", self.modal_rect.left + 20, self.modal_rect.top + 22, size=18, color=TEXT_WHITE, bold=True)

        state_str = str(self.final_state)
        state_c = ACCENT_GREEN if self.final_state == VehicleState.LANDED else (ACCENT_RED if self.final_state == VehicleState.DESTROYED else ACCENT_CYAN)
        draw_text(self.surface, f"STATUS: {state_str}", self.modal_rect.right - 20, self.modal_rect.top + 22, size=14, color=state_c, bold=True, align="right")

        pygame.draw.line(self.surface, PANEL_BORDER, (self.modal_rect.left + 20, self.modal_rect.top + 46), (self.modal_rect.right - 20, self.modal_rect.top + 46), width=1)

        # Summary KPIs
        summary = self.logger.get_summary()
        sy = self.modal_rect.top + 60
        sx = self.modal_rect.left + 20

        # Row 1
        draw_text(self.surface, f"Peak Altitude: {format_altitude(summary['max_altitude'])}", sx, sy, size=12, color=TEXT_WHITE, bold=True)
        draw_text(self.surface, f"Max Speed: {format_speed(summary['max_velocity'])}", sx + 240, sy, size=12, color=TEXT_WHITE, bold=True)
        draw_text(self.surface, f"Max G-Force: {summary['max_g_force']:.2f} g0", sx + 480, sy, size=12, color=TEXT_WHITE, bold=True)
        draw_text(self.surface, f"Duration: {format_time(summary['total_mission_elapsed_time'])}", sx + 700, sy, size=12, color=TEXT_WHITE, bold=True)

        # Row 2
        sy += 24
        draw_text(self.surface, f"Max-Q: {summary['max_dynamic_pressure']/1000.0:.2f} kPa (at T+{summary['max_q_time']:.1f}s)", sx, sy, size=11, color=TEXT_MUTED)
        draw_text(self.surface, f"Delta-V Expended: {summary['delta_v_expended']:.1f} m/s", sx + 240, sy, size=11, color=TEXT_MUTED)
        draw_text(self.surface, f"Apoapsis: {format_altitude(summary['apoapsis'])}", sx + 480, sy, size=11, color=TEXT_MUTED)
        draw_text(self.surface, f"Periapsis: {format_altitude(summary['periapsis'])}", sx + 700, sy, size=11, color=TEXT_MUTED)

        # Render 4 Time-Series Plots
        t_dict = self.logger.to_dict()
        times = t_dict.get("time", [])
        if times:
            self.plot_alt.draw(self.surface, times, t_dict.get("altitude", []))
            self.plot_spd.draw(self.surface, times, t_dict.get("speed", []))
            self.plot_q.draw(self.surface, times, t_dict.get("dynamic_pressure", []))
            self.plot_g.draw(self.surface, times, t_dict.get("g_force", []))

        # Action Buttons
        for btn in self.buttons:
            btn.draw(self.surface)

        # Toast confirmation
        if self.toast_msg and self.toast_timer > 0.0:
            draw_text(self.surface, self.toast_msg, self.modal_rect.centerx, self.modal_rect.bottom - 20, size=12, color=ACCENT_GREEN, bold=True, align="center")


class FlightView:
    """Integrated Close-Up Flight View, Orbital Map View, and Flight HUD."""

    def __init__(
        self,
        surface: pygame.Surface,
        rocket: Optional[Rocket] = None,
        body: Optional[CelestialBody] = None,
        on_return_vab: Optional[Callable[[], None]] = None,
    ) -> None:
        self.surface: pygame.Surface = surface
        self.on_return_vab: Optional[Callable[[], None]] = on_return_vab

        self.body: CelestialBody = body if body is not None else get_earth()
        self.rocket: Rocket = rocket if rocket is not None else create_falcon_9()

        # Initialize FlightVehicle standing vertical on the planetary surface launchpad
        pad_pos = Vector2D(0.0, self.body.radius)
        self.vehicle: FlightVehicle = FlightVehicle(
            rocket=self.rocket,
            body=self.body,
            position=pad_pos,
            velocity=self.body.surface_velocity_at(pad_pos),
            attitude=math.pi * 0.5,  # Vertical
            state=VehicleState.ON_PAD,
        )

        self.logger: TelemetryLogger = TelemetryLogger(sample_dt=0.05)
        self.sim: Simulation = Simulation(
            vehicle=self.vehicle,
            body=self.body,
            logger=self.logger,
            time_warp=1.0,
        )

        # High-precision 64-bit camera
        self.camera: Camera = Camera(
            viewport_width=surface.get_width(),
            viewport_height=surface.get_height(),
            position=self.vehicle.position,
            scale=12.0,
            mode=CameraMode.HORIZON_UP,
        )

        self.renderer: Renderer = Renderer(surface)
        self.toast: ToastManager = ToastManager()

        # View state
        self.is_map_view: bool = False
        self.modal: Optional[TelemetryModal] = None
        self.is_paused: bool = False

        # Input command states
        self.throttle_cmd: float = 0.0
        self.steer_rate: float = 0.0  # rad/s

        # UI Layout
        self.throttle_slider: VerticalSlider = VerticalSlider(pygame.Rect(20, 200, 50, 260), value=0.0)
        self.fuel_gauge: ProgressBar = ProgressBar(pygame.Rect(20, 480, 180, 24), label="Propellant")

        self.hud_buttons: list[Button] = []
        self.sas_buttons: dict[SASMode, Button] = {}
        self.warp_buttons: list[Button] = []

        self.update_layout(surface.get_width(), surface.get_height())

    def update_layout(self, width: int, height: int) -> None:
        """Update HUD layout geometries on window resize."""
        self.camera.set_viewport(width, height)
        self.renderer.set_surface(self.surface)

        # Left HUD: Slider & Fuel Gauge
        slider_y = height // 2 - 140
        self.throttle_slider.rect = pygame.Rect(24, slider_y, 54, 260)
        self.fuel_gauge.rect = pygame.Rect(24, slider_y + 276, 176, 26)

        # SAS Selector Buttons
        sas_x = 24
        sas_y = slider_y - 84
        sas_w = 42
        sas_h = 28

        self.sas_buttons.clear()
        modes = [
            (SASMode.OFF, "OFF"),
            (SASMode.HOLD_ATTITUDE, "HOLD"),
            (SASMode.PROGRADE, "PRO"),
            (SASMode.RETROGRADE, "RET"),
        ]
        for i, (m, lbl) in enumerate(modes):
            b = Button(
                rect=pygame.Rect(sas_x + i * 45, sas_y, sas_w, sas_h),
                text=lbl,
                is_toggle=False,
                is_active=(self.vehicle.autopilot.mode == m),
                on_click=lambda mode=m: self._set_sas_mode(mode),
                font_size=11,
            )
            self.sas_buttons[m] = b

        # Stage Button
        b_stage = Button(
            rect=pygame.Rect(24, slider_y + 312, 176, 42),
            text="STAGE",
            shortcut="[SPACE]",
            bg_color=ACCENT_ORANGE,
            text_color=(15, 20, 25),
            on_click=self.trigger_stage,
            font_size=16,
        )

        # Top Bar: Time Warp Buttons
        self.warp_buttons.clear()
        warps = [1.0, 2.0, 4.0, 10.0, 50.0, 100.0, 1000.0]
        wx = width // 2 - 240
        wy = 12
        for w_val in warps:
            w_text = f"{int(w_val)}x" if w_val >= 1.0 else f"{w_val}x"
            btn = Button(
                rect=pygame.Rect(wx, wy, 44, 28),
                text=w_text,
                is_active=math.isclose(self.sim.time_warp, w_val),
                on_click=lambda w=w_val: self._request_time_warp(w),
                font_size=11,
            )
            self.warp_buttons.append(btn)
            wx += 48

        # Camera & Map View Toggles
        b_cam = Button(
            rect=pygame.Rect(width - 240, 12, 110, 28),
            text="Camera [C]",
            on_click=self._toggle_camera_mode,
            font_size=11,
        )
        b_map = Button(
            rect=pygame.Rect(width - 120, 12, 100, 28),
            text="Map [M]",
            on_click=self.toggle_map_view,
            font_size=11,
        )

        # End Mission Button
        b_end = Button(
            rect=pygame.Rect(24, 12, 100, 28),
            text="Menu [V]",
            on_click=self.open_modal,
            font_size=11,
        )

        self.hud_buttons = [b_stage, b_cam, b_map, b_end]

        if self.modal is not None:
            self.modal.update_layout(width, height)

    def _set_sas_mode(self, mode: SASMode) -> None:
        """Switch SAS flight guidance mode."""
        self.vehicle.autopilot.set_mode(mode)
        if mode == SASMode.HOLD_ATTITUDE:
            self.vehicle.autopilot.target_attitude = self.vehicle.attitude

    def _toggle_camera_mode(self) -> None:
        """Switch camera between Horizon-Up and Inertial."""
        new_mode = self.camera.toggle_mode()
        self.toast.show(f"Camera: {new_mode.value}", duration=1.5, color=ACCENT_CYAN)

    def toggle_map_view(self) -> None:
        """Switch cleanly between close-up Flight View and orbital Map View."""
        self.is_map_view = not self.is_map_view
        if self.is_map_view:
            self.camera.set_mode(CameraMode.INERTIAL)
            self.camera.snap_scale(4e-5)  # View Earth scale
            self.toast.show("Map View [Inertial]", duration=1.5, color=ACCENT_CYAN)
        else:
            self.camera.set_mode(CameraMode.HORIZON_UP)
            self.camera.snap_scale(10.0)  # Close up on craft
            self.toast.show("Flight View [Horizon Up]", duration=1.5, color=ACCENT_CYAN)

    def _request_time_warp(self, warp_factor: float) -> None:
        """Request a time warp factor, showing toast if clamped."""
        old_warp = self.sim.time_warp
        self.sim.set_time_warp(warp_factor)

        if warp_factor > 4.0 and self.sim.time_warp <= 4.0:
            if self.vehicle.altitude < self.sim.cutoff_altitude:
                self.toast.show("Cannot warp >4x below 100 km (Atmosphere)", duration=2.5, color=ACCENT_ORANGE)
            elif self.sim.is_thrusting():
                self.toast.show("Cannot warp >4x while engines firing", duration=2.5, color=ACCENT_ORANGE)
            else:
                self.toast.show("Time warp clamped to 4x (Safety)", duration=2.0, color=ACCENT_ORANGE)
        else:
            mode_badge = "ON-RAILS" if self.sim.warp_mode == WarpMode.ON_RAILS else "PHYSICS"
            self.toast.show(f"Warp: {self.sim.time_warp:.0f}x [{mode_badge}]", duration=1.5, color=ACCENT_CYAN)

    def trigger_stage(self) -> None:
        """Perform stage separation."""
        cur_pos = self.vehicle.position
        cur_vel = self.vehicle.velocity
        cur_att = self.vehicle.attitude
        cur_om = self.vehicle.angular_velocity

        decoupled = self.vehicle.stage(ignite=True)
        if decoupled is not None:
            # Add to visual coasting debris
            self.renderer.add_debris(
                stage=decoupled,
                position=cur_pos,
                velocity=cur_vel - self.vehicle.heading_vector * 3.0,
                attitude=cur_att,
                omega=cur_om + (random.random() - 0.5) * 0.1,
            )
            self.toast.show(f"Stage Separated: {decoupled.name}", duration=2.0, color=ACCENT_GREEN)

    def revert_to_launch(self) -> None:
        """Reset flight vehicle back to pad."""
        self.modal = None
        self.is_paused = False
        # Recreate fresh rocket preset
        self.rocket = create_falcon_9()
        pad_pos = Vector2D(0.0, self.body.radius)
        self.vehicle = FlightVehicle(
            rocket=self.rocket,
            body=self.body,
            position=pad_pos,
            velocity=self.body.surface_velocity_at(pad_pos),
            attitude=math.pi * 0.5,
            state=VehicleState.ON_PAD,
        )
        self.logger = TelemetryLogger(sample_dt=0.05)
        self.sim = Simulation(vehicle=self.vehicle, body=self.body, logger=self.logger)
        self.throttle_slider.set_value(0.0)
        self.throttle_cmd = 0.0
        self.camera.snap_to(self.vehicle.position)
        self.camera.snap_scale(12.0)
        self.camera.set_mode(CameraMode.HORIZON_UP)
        self.is_map_view = False
        self.toast.show("Reverted to Launchpad", duration=2.0, color=ACCENT_CYAN)

    def open_modal(self) -> None:
        """Open mission summary modal."""
        self.modal = TelemetryModal(
            surface=self.surface,
            logger=self.logger,
            final_state=self.vehicle.state,
            on_revert=self.revert_to_launch,
            on_vab=self.return_to_vab,
            on_resume=self.resume_flight,
        )

    def resume_flight(self) -> None:
        """Resume active flight from modal."""
        self.modal = None

    def return_to_vab(self) -> None:
        """Exit back to Blueprint Editor."""
        self.modal = None
        if self.on_return_vab is not None:
            self.on_return_vab()

    # =========================================================================
    # Event Handling
    # =========================================================================

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle keyboard flight controls and GUI interaction."""
        # If modal is open, delegate events directly
        if self.modal is not None:
            return self.modal.handle_event(event)

        # HUD Buttons
        for btn in self.hud_buttons:
            if btn.handle_event(event):
                return True

        for btn in self.sas_buttons.values():
            if btn.handle_event(event):
                return True

        for btn in self.warp_buttons:
            if btn.handle_event(event):
                return True

        # Throttle Slider
        if self.throttle_slider.handle_event(event):
            self.throttle_cmd = self.throttle_slider.value
            return True

        # Keyboard Controls
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.trigger_stage()
                return True
            elif event.key == pygame.K_m:
                self.toggle_map_view()
                return True
            elif event.key == pygame.K_c:
                self._toggle_camera_mode()
                return True
            elif event.key == pygame.K_r:
                self.revert_to_launch()
                return True
            elif event.key in (pygame.K_v, pygame.K_ESCAPE):
                self.open_modal()
                return True
            elif event.key == pygame.K_z:  # Max throttle
                self.throttle_slider.set_value(1.0)
                self.throttle_cmd = 1.0
                return True
            elif event.key == pygame.K_x:  # Cut throttle
                self.throttle_slider.set_value(0.0)
                self.throttle_cmd = 0.0
                return True
            elif event.key in (pygame.K_UP, pygame.K_LSHIFT):
                self.throttle_cmd = min(1.0, self.throttle_cmd + 0.05)
                self.throttle_slider.set_value(self.throttle_cmd)
                return True
            elif event.key in (pygame.K_DOWN, pygame.K_LCTRL):
                self.throttle_cmd = max(0.0, self.throttle_cmd - 0.05)
                self.throttle_slider.set_value(self.throttle_cmd)
                return True
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self.steer_rate = 0.4  # Steer CCW
                return True
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self.steer_rate = -0.4  # Steer CW
                return True
            elif event.key == pygame.K_PERIOD:  # Increase warp
                self._step_warp(1)
                return True
            elif event.key == pygame.K_COMMA:  # Decrease warp
                self._step_warp(-1)
                return True

        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d):
                self.steer_rate = 0.0
                return True

        # Mouse Zoom
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Wheel up
                self.camera.zoom_in(1.25)
                return True
            elif event.button == 5:  # Wheel down
                self.camera.zoom_out(1.25)
                return True

        return False

    def _step_warp(self, direction: int) -> None:
        """Step time warp up or down in discrete levels."""
        warps = [1.0, 2.0, 4.0, 10.0, 50.0, 100.0, 1000.0]
        cur = self.sim.time_warp
        closest_idx = min(range(len(warps)), key=lambda i: abs(warps[i] - cur))
        target_idx = max(0, min(len(warps) - 1, closest_idx + direction))
        self._request_time_warp(warps[target_idx])

    # =========================================================================
    # Simulation Step & Rendering
    # =========================================================================

    def update(self, dt: float) -> None:
        """Advance physical simulation and camera tracking."""
        dt_val = min(0.1, float(dt))

        # Advance modal toast if open
        if self.modal is not None:
            self.modal.update(dt_val)
            return

        # Advance steering control
        if abs(self.steer_rate) > 1e-4:
            if self.vehicle.autopilot.mode == SASMode.HOLD_ATTITUDE:
                self.vehicle.autopilot.target_attitude += self.steer_rate * dt_val
            elif self.vehicle.autopilot.mode == SASMode.OFF:
                # Direct manual gimbal steering
                if self.vehicle.rocket.active_stage is not None:
                    for eng in self.vehicle.rocket.active_stage.engines:
                        eng.set_gimbal(eng.current_gimbal + self.steer_rate * dt_val * 0.2)

        # Advance physical simulation
        target_pitch = (
            self.vehicle.autopilot.target_attitude
            if self.vehicle.autopilot.mode == SASMode.HOLD_ATTITUDE
            else None
        )
        snap = self.sim.step(
            wall_dt=dt_val,
            target_pitch=target_pitch,
            throttle=self.throttle_cmd,
        )

        # Update stage debris
        self.renderer.update_debris(dt_val, self.body)

        # Trigger modal on landing or destruction
        if self.vehicle.state in (VehicleState.LANDED, VehicleState.DESTROYED) and self.modal is None:
            self.open_modal()

        # Camera tracking
        self.camera.set_reference_position(self.vehicle.position)
        self.camera.set_target(self.vehicle.position)
        self.camera.update(dt_val)

        # Update toast manager
        self.toast.update(dt_val)

    def render(self) -> None:
        """Render Flight View, Map View, HUD, and Telemetry Modal."""
        # 1. Atmospheric sky & starfield
        self.renderer.render_sky_and_stars(self.camera, self.vehicle, self.body)

        # 2. Curved Planetary Terrain
        self.renderer.render_terrain(self.camera, self.body)

        # 3. Keplerian Trajectory in Map View (or high zoom)
        if self.is_map_view or self.camera.scale < 0.1:
            self.renderer.render_orbit_trajectory(self.camera, self.vehicle, self.body)

        # 4. Separated Coasting Debris
        self.renderer.render_debris(self.camera)

        # 5. Rocket & Multi-Layer Plumes
        self.renderer.render_rocket(
            camera=self.camera,
            vehicle=self.vehicle,
            render_plumes=True,
        )

        # 6. Flight Vector Glyphs (Prograde, Retrograde, Heading)
        if not self.is_map_view:
            self.renderer.render_flight_vectors(self.camera, self.vehicle)

        # 7. Render Flight HUD Overlays
        self._render_hud()

        # 8. Render Toast Banners
        self.toast.draw(self.surface)

        # 9. Render Post-Flight Modal if active
        if self.modal is not None:
            self.modal.render()

    def _render_hud(self) -> None:
        """Draw interactive HUD panels, buttons, throttle lever, and readouts."""
        sw = self.surface.get_width()
        sh = self.surface.get_height()

        # 1. Left HUD: Throttle slider, Fuel gauge, SAS buttons, Stage button
        self.throttle_slider.draw(self.surface)

        # Active stage propellant fraction
        active_st = self.vehicle.rocket.active_stage
        if active_st is not None:
            prop_cur = active_st.propellant_mass()
            prop_cap = sum(getattr(t, "propellant_capacity", prop_cur) for t in active_st.fuel_tanks)
            self.fuel_gauge.set_value(prop_cur, max_val=max(1.0, prop_cap))
            self.fuel_gauge.label = f"Propellant ({format_mass(prop_cur)})"
        else:
            self.fuel_gauge.set_value(0.0, max_val=1.0)
            self.fuel_gauge.label = "No Active Stage"
        self.fuel_gauge.draw(self.surface)

        # SAS Buttons
        for mode, btn in self.sas_buttons.items():
            btn.is_active = (self.vehicle.autopilot.mode == mode)
            btn.draw(self.surface)

        # Top Bar Buttons
        for btn in self.hud_buttons:
            btn.draw(self.surface)

        # Time Warp Buttons
        for btn in self.warp_buttons:
            w_val = float(btn.text.replace("x", ""))
            btn.is_active = math.isclose(self.sim.time_warp, w_val)
            btn.draw(self.surface)

        # Top Center: Warp Mode Badge & Mission Clock
        cx = sw // 2
        mode_badge = "ON-RAILS" if self.sim.warp_mode == WarpMode.ON_RAILS else "PHYSICS"
        badge_c = ACCENT_CYAN if self.sim.warp_mode == WarpMode.ON_RAILS else ACCENT_GREEN
        draw_text(self.surface, f"{self.sim.time_warp:.0f}x [{mode_badge}]", cx, 48, size=12, color=badge_c, bold=True, align="center")
        draw_text(self.surface, format_time(self.sim.elapsed_time), cx, 66, size=14, color=TEXT_WHITE, bold=True, mono=True, align="center")

        # 2. Right HUD: Digital Telemetry Panel
        self._render_telemetry_panel(sw - 230, 48, 216)

    def _render_telemetry_panel(self, x: int, y: int, width: int) -> None:
        """Render digital flight telemetry readouts."""
        snap = self.vehicle.get_telemetry_snapshot()

        panel_h = 360
        panel_rect = pygame.Rect(x, y, width, panel_h)
        pygame.draw.rect(self.surface, PANEL_BG, panel_rect, border_radius=8)
        pygame.draw.rect(self.surface, PANEL_BORDER, panel_rect, width=1, border_radius=8)

        draw_text(self.surface, "TELEMETRY", x + 12, y + 14, size=12, color=TEXT_WHITE, bold=True)
        # Vehicle state badge
        st_color = ACCENT_GREEN if snap.state in (VehicleState.ON_PAD, VehicleState.IN_FLIGHT) else ACCENT_RED
        draw_text(self.surface, str(snap.state or ""), x + width - 12, y + 14, size=11, color=st_color, bold=True, align="right")

        pygame.draw.line(self.surface, PANEL_BORDER, (x + 10, y + 28), (x + width - 10, y + 28), width=1)

        ty = y + 42
        dy = 22
        rx = x + width - 14

        # Readouts
        readouts: list[tuple[str, str, tuple[int, int, int]]] = [
            ("Altitude", format_altitude(snap.altitude), TEXT_WHITE),
            ("Apoapsis (Ap)", format_altitude(snap.apoapsis), ACCENT_CYAN),
            ("Periapsis (Pe)", format_altitude(snap.periapsis), ACCENT_CYAN if snap.periapsis >= 0 else ACCENT_RED),
            ("Speed", format_speed(snap.speed), TEXT_WHITE),
            ("Mach", f"{snap.mach:.2f}", TEXT_MUTED),
            ("Dynamic Press (q)", f"{snap.dynamic_pressure/1000.0:.2f} kPa", ACCENT_ORANGE if snap.dynamic_pressure > 30000 else TEXT_MUTED),
            ("G-Force", f"{snap.g_force:.2f} g0", ACCENT_RED if snap.g_force > 5.0 else TEXT_WHITE),
            ("Local Pitch", f"{math.degrees(snap.local_pitch):+.1f}°", ACCENT_YELLOW),
            ("Angle of Attack", f"{snap.aoa_deg:.2f}°", TEXT_MUTED),
            ("Total Mass", format_mass(snap.mass), TEXT_WHITE),
            ("Thrust", format_thrust(snap.thrust), ACCENT_ORANGE if snap.thrust > 0 else TEXT_MUTED),
            ("Delta-V Expended", f"{snap.delta_v_expended:.1f} m/s", ACCENT_CYAN),
        ]

        for lbl, val, val_c in readouts:
            draw_text(self.surface, lbl, x + 12, ty, size=11, color=TEXT_MUTED)
            draw_text(self.surface, val, rx, ty, size=11, color=val_c, bold=True, align="right")
            ty += dy


__all__ = [
    "FlightView",
    "TelemetryModal",
]
