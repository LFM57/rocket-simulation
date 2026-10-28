"""Vehicle Assembly Building (VAB) / Spaceflight Simulator style Blueprint Editor.

Provides an interactive blueprint workspace with:
    - Left part catalog sidebar (Engines, Tanks, Decouplers, Payloads).
    - Center blueprint canvas with meter scale grid, vertical stack auto-snapping,
      parallel engine clustering, part property inspector, and live CoM / CoP markers.
    - Right staging & engineering panel with mass rollups, sea-level & vacuum TWR,
      analytical Tsiolkovsky stage Delta-V calculations, and Falcon 9 preset loading.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pygame

from core.constants import g0
from models.parts import Decoupler, Engine, FuelTank, Part, Payload
from models.presets import create_falcon_9
from models.rocket import Rocket, Stage
from ui.widgets import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_RED,
    ACCENT_YELLOW,
    BG_BLUEPRINT,
    BG_DARK,
    BUTTON_ACTIVE,
    BUTTON_NORMAL,
    GRID_LINE,
    GRID_MAJOR,
    PANEL_BG,
    PANEL_BORDER,
    PANEL_BORDER_LIGHT,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_WHITE,
    Button,
    Panel,
    draw_text,
    format_mass,
    format_thrust,
    get_font,
)


@dataclass
class StageEngineeringStats:
    """Analytical engineering metrics for a single stage within the rocket stack."""

    stage_index: int
    stage_name: str
    total_mass: float
    dry_mass: float
    propellant_mass: float
    thrust_sl: float
    thrust_vac: float
    twr_sl: float
    twr_vac: float
    isp_vac: float
    delta_v: float


@dataclass
class RocketEngineeringSummary:
    """Complete analytical engineering summary for a rocket vehicle."""

    total_mass: float
    dry_mass: float
    propellant_mass: float
    total_delta_v: float
    stage0_twr_sl: float
    is_launchable: bool
    stages: list[StageEngineeringStats]


def calculate_rocket_engineering_stats(rocket: Rocket) -> RocketEngineeringSummary:
    """Compute analytical mass rollup, TWR, and Tsiolkovsky Delta-V for each stage.

    Stage Delta-V uses Tsiolkovsky's Rocket Equation:
        Delta-v_i = g0 * Isp_vac,i * ln(m_initial / m_final)
    where:
        m_initial is the total mass of the vehicle when stage i ignites
        (including stage i and all subsequent payload/upper stages).
        m_final = m_initial - m_propellant(stage i).

    Average Isp_vac is thrust-weighted:
        Isp_vac = sum(F_vac) / sum(F_vac / Isp)
    """
    if not rocket.stages:
        return RocketEngineeringSummary(
            total_mass=0.0,
            dry_mass=0.0,
            propellant_mass=0.0,
            total_delta_v=0.0,
            stage0_twr_sl=0.0,
            is_launchable=False,
            stages=[],
        )

    # Sort non-separated stages by staging order (0 is first/liftoff stage)
    active_stages = sorted(
        [s for s in rocket.stages if not s.is_separated],
        key=lambda s: s.staging_order,
    )

    total_mass = float(sum(s.total_mass() for s in active_stages))
    total_dry = float(sum(s.dry_mass() for s in active_stages))
    total_prop = float(sum(s.propellant_mass() for s in active_stages))

    # Calculate cumulative stack mass remaining at ignition of each stage
    # Stage 0 has full stack mass; Stage 1 has full stack minus Stage 0, etc.
    stage_stats: list[StageEngineeringStats] = []
    current_stack_mass = total_mass
    total_dv = 0.0

    for idx, st in enumerate(active_stages):
        st_mass = st.total_mass()
        st_dry = st.dry_mass()
        st_prop = st.propellant_mass()

        # Engine thrust aggregations
        f_sl = sum(e.thrust_sl for e in st.engines)
        f_vac = sum(e.thrust_vac for e in st.engines)

        # Average vacuum Isp
        flow_sum = sum(e.thrust_vac / max(1.0, e.isp_vac) for e in st.engines)
        avg_isp_vac = (f_vac / flow_sum) if flow_sum > 0.0 else 0.0

        # TWR based on stack mass at ignition
        weight = current_stack_mass * g0
        twr_sl = (f_sl / weight) if weight > 0.0 else 0.0
        twr_vac = (f_vac / weight) if weight > 0.0 else 0.0

        # Tsiolkovsky Delta-V
        m_init = current_stack_mass
        m_burnout = max(1e-6, m_init - st_prop)
        if avg_isp_vac > 0.0 and m_init > m_burnout:
            st_dv = g0 * avg_isp_vac * math.log(m_init / m_burnout)
        else:
            st_dv = 0.0

        total_dv += st_dv

        stage_stats.append(
            StageEngineeringStats(
                stage_index=st.staging_order,
                stage_name=st.name,
                total_mass=st_mass,
                dry_mass=st_dry,
                propellant_mass=st_prop,
                thrust_sl=f_sl,
                thrust_vac=f_vac,
                twr_sl=twr_sl,
                twr_vac=twr_vac,
                isp_vac=avg_isp_vac,
                delta_v=st_dv,
            )
        )

        # Subtract full stage mass for subsequent upper stage calculations
        current_stack_mass = max(1.0, current_stack_mass - st_mass)

    stage0_twr_sl = stage_stats[0].twr_sl if stage_stats else 0.0
    is_launchable = stage0_twr_sl > 1.0 and any(s.thrust_sl > 0.0 for s in stage_stats)

    return RocketEngineeringSummary(
        total_mass=total_mass,
        dry_mass=total_dry,
        propellant_mass=total_prop,
        total_delta_v=total_dv,
        stage0_twr_sl=stage0_twr_sl,
        is_launchable=is_launchable,
        stages=stage_stats,
    )


# Catalog Part Templates for Spawning
CATALOG_TEMPLATES: list[dict[str, Any]] = [
    # Engines
    {
        "category": "Engines",
        "name": "Merlin 1D",
        "factory": lambda: Engine(
            id=f"merlin_1d_{random.randint(100, 999)}",
            name="Merlin 1D",
            dry_mass=470.0,
            length=2.7,
            radius=0.6,
            position_offset=1.35,
            thrust_vac=914000.0,
            thrust_sl=845000.0,
            isp_vac=311.0,
            isp_sl=282.0,
        ),
    },
    {
        "category": "Engines",
        "name": "Merlin 1D Vacuum",
        "factory": lambda: Engine(
            id=f"merlin_vac_{random.randint(100, 999)}",
            name="Merlin 1D Vacuum",
            dry_mass=900.0,
            length=3.0,
            radius=1.2,
            position_offset=1.5,
            thrust_vac=981000.0,
            thrust_sl=400000.0,
            isp_vac=348.0,
            isp_sl=200.0,
        ),
    },
    {
        "category": "Engines",
        "name": "Booster Thruster",
        "factory": lambda: Engine(
            id=f"booster_thruster_{random.randint(100, 999)}",
            name="Booster Thruster",
            dry_mass=250.0,
            length=1.8,
            radius=0.45,
            position_offset=0.9,
            thrust_vac=350000.0,
            thrust_sl=300000.0,
            isp_vac=290.0,
            isp_sl=260.0,
        ),
    },
    # Fuel Tanks
    {
        "category": "Fuel Tanks",
        "name": "Large Stage 1 Tank",
        "factory": lambda: FuelTank(
            id=f"tank_s1_{random.randint(100, 999)}",
            name="Large Stage 1 Tank",
            dry_mass=20000.0,
            length=40.0,
            radius=1.85,
            position_offset=20.0,
            propellant_mass=410000.0,
            propellant_capacity=410000.0,
        ),
    },
    {
        "category": "Fuel Tanks",
        "name": "Upper Stage Tank",
        "factory": lambda: FuelTank(
            id=f"tank_s2_{random.randint(100, 999)}",
            name="Upper Stage Tank",
            dry_mass=4000.0,
            length=10.0,
            radius=1.85,
            position_offset=5.0,
            propellant_mass=107500.0,
            propellant_capacity=107500.0,
        ),
    },
    {
        "category": "Fuel Tanks",
        "name": "Medium Propellant Tank",
        "factory": lambda: FuelTank(
            id=f"tank_med_{random.randint(100, 999)}",
            name="Medium Propellant Tank",
            dry_mass=2500.0,
            length=8.0,
            radius=1.85,
            position_offset=4.0,
            propellant_mass=45000.0,
            propellant_capacity=45000.0,
        ),
    },
    {
        "category": "Fuel Tanks",
        "name": "Small Propellant Tank",
        "factory": lambda: FuelTank(
            id=f"tank_small_{random.randint(100, 999)}",
            name="Small Propellant Tank",
            dry_mass=1000.0,
            length=4.0,
            radius=1.2,
            position_offset=2.0,
            propellant_mass=15000.0,
            propellant_capacity=15000.0,
        ),
    },
    # Decouplers
    {
        "category": "Decouplers",
        "name": "Interstage Decoupler",
        "factory": lambda: Decoupler(
            id=f"decoupler_{random.randint(100, 999)}",
            name="Interstage Decoupler",
            dry_mass=1000.0,
            length=3.0,
            radius=1.85,
            position_offset=1.5,
            separation_impulse=5000.0,
        ),
    },
    # Payloads
    {
        "category": "Payloads",
        "name": "Dragon Capsule",
        "factory": lambda: Payload(
            id=f"capsule_{random.randint(100, 999)}",
            name="Dragon Payload Capsule",
            dry_mass=5000.0,
            length=8.0,
            radius=1.85,
            position_offset=4.0,
        ),
    },
    {
        "category": "Payloads",
        "name": "Satellite Fairing",
        "factory": lambda: Payload(
            id=f"fairing_{random.randint(100, 999)}",
            name="Satellite Fairing",
            dry_mass=1800.0,
            length=13.0,
            radius=2.6,
            position_offset=6.5,
        ),
    },
]


class VABView:
    """Interactive Vehicle Assembly Building screen."""

    def __init__(
        self,
        surface: pygame.Surface,
        on_launch: Optional[Callable[[Rocket], None]] = None,
    ) -> None:
        self.surface: pygame.Surface = surface
        self.on_launch: Optional[Callable[[Rocket], None]] = on_launch

        # Active rocket under construction
        self.rocket: Rocket = create_falcon_9()
        self.selected_part: Optional[Part] = None
        self.selected_stage_idx: int = 0

        # Canvas pan & zoom
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.scale: float = 8.0  # px/meter
        self.is_panning: bool = False
        self.pan_start: tuple[int, int] = (0, 0)

        # UI Layout Rectangles (computed in update_layout)
        self.sidebar_left_rect: pygame.Rect = pygame.Rect(0, 0, 240, 720)
        self.sidebar_right_rect: pygame.Rect = pygame.Rect(1000, 0, 280, 720)
        self.top_bar_rect: pygame.Rect = pygame.Rect(240, 0, 760, 48)
        self.canvas_rect: pygame.Rect = pygame.Rect(240, 48, 760, 672)

        # Catalog buttons
        self.catalog_buttons: list[Button] = []
        self.top_buttons: list[Button] = []
        self.action_buttons: list[Button] = []

        self._init_ui()
        self.recenter_canvas()

    def update_layout(self, width: int, height: int) -> None:
        """Update layout geometries when window is resized."""
        left_w = 230
        right_w = 290
        top_h = 46

        self.sidebar_left_rect = pygame.Rect(0, 0, left_w, height)
        self.sidebar_right_rect = pygame.Rect(width - right_w, 0, right_w, height)
        self.top_bar_rect = pygame.Rect(left_w, 0, width - left_w - right_w, top_h)
        self.canvas_rect = pygame.Rect(left_w, top_h, width - left_w - right_w, height - top_h)

        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI widgets and buttons."""
        self.catalog_buttons.clear()
        self.top_buttons.clear()
        self.action_buttons.clear()

        # 1. Left Catalog Buttons
        by = 50
        for item in CATALOG_TEMPLATES:
            btn_rect = pygame.Rect(12, by, self.sidebar_left_rect.width - 24, 34)
            # Capture factory in closure
            factory_fn = item["factory"]
            btn = Button(
                rect=btn_rect,
                text=item["name"],
                on_click=lambda fn=factory_fn: self._spawn_part(fn()),
                font_size=12,
            )
            self.catalog_buttons.append(btn)
            by += 40

        # 2. Top Bar Buttons
        tx = self.top_bar_rect.left + 12
        ty = 8

        b_preset = Button(
            rect=pygame.Rect(tx, ty, 150, 30),
            text="Falcon 9 Preset",
            on_click=self._load_falcon9,
            font_size=12,
        )
        tx += 160

        b_clear = Button(
            rect=pygame.Rect(tx, ty, 70, 30),
            text="Clear",
            on_click=self._clear_rocket,
            font_size=12,
        )
        tx += 80

        b_save = Button(
            rect=pygame.Rect(tx, ty, 80, 30),
            text="Save JSON",
            on_click=self._save_json,
            font_size=12,
        )
        tx += 90

        b_load = Button(
            rect=pygame.Rect(tx, ty, 80, 30),
            text="Load JSON",
            on_click=self._load_json,
            font_size=12,
        )
        tx += 100

        self.top_buttons = [b_preset, b_clear, b_save, b_load]

        # 3. Right Panel Action Button (LAUNCH)
        rx = self.sidebar_right_rect.left + 16
        ry = self.sidebar_right_rect.bottom - 54
        rw = self.sidebar_right_rect.width - 32

        b_launch = Button(
            rect=pygame.Rect(rx, ry, rw, 40),
            text="LAUNCH",
            shortcut="[SPACE / ENTER]",
            on_click=self._trigger_launch,
            bg_color=ACCENT_GREEN,
            active_color=(35, 180, 85),
            text_color=(10, 20, 15),
            font_size=16,
        )
        self.action_buttons = [b_launch]

    def recenter_canvas(self) -> None:
        """Center the blueprint canvas on the rocket."""
        total_len = 0.0
        for s in self.rocket.stages:
            for p in s.parts:
                total_len = max(total_len, s.position_offset + p.position_offset + p.length * 0.5)

        self.scale = 10.0
        self.pan_x = self.canvas_rect.centerx
        self.pan_y = self.canvas_rect.centery + (total_len * 0.5 * self.scale)

    def _spawn_part(self, new_part: Part) -> None:
        """Spawn a part from the catalog and stack it onto the current stage."""
        if not self.rocket.stages:
            st0 = Stage(stage_id="stage_0", name="Stage 0 Booster", staging_order=0)
            self.rocket.stages.append(st0)

        # Place into active stage
        target_stage = self.rocket.stages[0]
        for s in self.rocket.stages:
            if s.staging_order == self.selected_stage_idx:
                target_stage = s
                break

        # Calculate snap position at the top of the stage
        if target_stage.parts:
            top_y = max(p.position_offset + p.length * 0.5 for p in target_stage.parts)
            new_part.position_offset = top_y + new_part.length * 0.5
        else:
            new_part.position_offset = new_part.length * 0.5

        target_stage.parts.append(new_part)
        self.selected_part = new_part

    def _load_falcon9(self) -> None:
        """Load Falcon 9 reference preset."""
        self.rocket = create_falcon_9()
        self.selected_part = None
        self.recenter_canvas()

    def _clear_rocket(self) -> None:
        """Clear all stages and parts."""
        self.rocket = Rocket(name="Custom Rocket", stages=[Stage(stage_id="stage_0", name="Stage 0 Booster")])
        self.selected_part = None

    def _save_json(self) -> None:
        """Save rocket to local rocket.json."""
        try:
            with open("rocket.json", "w", encoding="utf-8") as f:
                f.write(self.rocket.to_json(indent=2))
        except Exception as e:
            print(f"Error saving rocket JSON: {e}")

    def _load_json(self) -> None:
        """Load rocket from local rocket.json."""
        path = Path("rocket.json")
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.rocket = Rocket.from_json(f.read())
                    self.selected_part = None
                    self.recenter_canvas()
            except Exception as e:
                print(f"Error loading rocket JSON: {e}")

    def _trigger_launch(self) -> None:
        """Launch vehicle into active Flight View."""
        if self.on_launch is not None:
            self.on_launch(self.rocket)

    # =========================================================================
    # Event Handling
    # =========================================================================

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse clicks, part selection, dragging, and keyboard shortcuts."""
        # Top bar buttons
        for btn in self.top_buttons:
            if btn.handle_event(event):
                return True

        # Action buttons
        for btn in self.action_buttons:
            if btn.handle_event(event):
                return True

        # Catalog buttons
        for btn in self.catalog_buttons:
            if btn.handle_event(event):
                return True

        # Keyboard shortcuts
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                self._trigger_launch()
                return True
            elif event.key == pygame.K_DELETE and self.selected_part is not None:
                self._delete_selected_part()
                return True
            elif event.key == pygame.K_c:
                self.recenter_canvas()
                return True

        # Canvas pan & zoom
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.canvas_rect.collidepoint(event.pos):
                if event.button == 1:
                    # Check part selection
                    part = self._get_part_at_mouse(event.pos)
                    if part is not None:
                        self.selected_part = part
                    else:
                        self.is_panning = True
                        self.pan_start = event.pos
                    return True
                elif event.button == 3:  # Right-click pan
                    self.is_panning = True
                    self.pan_start = event.pos
                    return True
                elif event.button == 4:  # Wheel up: zoom in
                    self._zoom(1.15, event.pos)
                    return True
                elif event.button == 5:  # Wheel down: zoom out
                    self._zoom(1.0 / 1.15, event.pos)
                    return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button in (1, 3) and self.is_panning:
                self.is_panning = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            if self.is_panning:
                dx = event.pos[0] - self.pan_start[0]
                dy = event.pos[1] - self.pan_start[1]
                self.pan_x += dx
                self.pan_y += dy
                self.pan_start = event.pos
                return True

        return False

    def _delete_selected_part(self) -> None:
        """Remove currently selected part from its stage."""
        if self.selected_part is None:
            return
        for s in self.rocket.stages:
            if self.selected_part in s.parts:
                s.parts.remove(self.selected_part)
                break
        self.selected_part = None

    def _zoom(self, factor: float, mouse_pos: tuple[int, int]) -> None:
        """Zoom blueprint canvas centered on mouse position."""
        mx, my = mouse_pos
        old_scale = self.scale
        new_scale = max(2.0, min(80.0, old_scale * factor))
        self.scale = new_scale

        # Shift pan so mouse point remains stationary
        self.pan_x = mx - (mx - self.pan_x) * (new_scale / old_scale)
        self.pan_y = my - (my - self.pan_y) * (new_scale / old_scale)

    def _get_part_at_mouse(self, mouse_pos: tuple[int, int]) -> Optional[Part]:
        """Hit test part under mouse cursor."""
        mx, my = mouse_pos
        for s in self.rocket.stages:
            stage_base = s.position_offset
            for p in s.parts:
                y_center = stage_base + p.position_offset
                l_half = p.length * 0.5
                r = p.radius

                # Screen bounds
                sx1 = self.pan_x - r * self.scale
                sx2 = self.pan_x + r * self.scale
                sy1 = self.pan_y - (y_center + l_half) * self.scale
                sy2 = self.pan_y - (y_center - l_half) * self.scale

                if sx1 <= mx <= sx2 and sy1 <= my <= sy2:
                    return p
        return None

    # =========================================================================
    # Rendering
    # =========================================================================

    def render(self) -> None:
        """Render complete VAB screen: catalog, canvas, stack, CoM/CoP, and readout."""
        # 1. Blueprint Grid Canvas
        self._render_blueprint_canvas()

        # 2. Left Catalog Panel
        self._render_catalog_panel()

        # 3. Top Toolbar
        self._render_top_toolbar()

        # 4. Right Staging & Engineering Readout
        self._render_engineering_panel()

    def _render_blueprint_canvas(self) -> None:
        """Draw blueprint dark blue background, grid, and stacked rocket parts."""
        # Clip to canvas rect
        self.surface.set_clip(self.canvas_rect)
        pygame.draw.rect(self.surface, BG_BLUEPRINT, self.canvas_rect)

        # Draw grid lines
        grid_m = 5.0 if self.scale < 10.0 else (1.0 if self.scale < 30.0 else 0.5)
        step_px = grid_m * self.scale

        # Vertical grid lines
        start_x = self.pan_x % step_px
        x = self.canvas_rect.left + ((start_x - self.canvas_rect.left) % step_px)
        while x < self.canvas_rect.right:
            is_centerline = abs(x - self.pan_x) < 1.0
            color = (55, 75, 110) if is_centerline else GRID_LINE
            width = 2 if is_centerline else 1
            pygame.draw.line(self.surface, color, (x, self.canvas_rect.top), (x, self.canvas_rect.bottom), width=width)
            x += step_px

        # Horizontal grid lines
        start_y = self.pan_y % step_px
        y = self.canvas_rect.top + ((start_y - self.canvas_rect.top) % step_px)
        while y < self.canvas_rect.bottom:
            is_baseline = abs(y - self.pan_y) < 1.0
            color = (55, 75, 110) if is_baseline else GRID_LINE
            width = 2 if is_baseline else 1
            pygame.draw.line(self.surface, color, (self.canvas_rect.left, y), (self.canvas_rect.right, y), width=width)
            y += step_px

        # Centerline text marker
        draw_text(self.surface, "Centerline (X=0)", self.pan_x + 6, self.canvas_rect.top + 10, size=11, color=(80, 110, 150))
        draw_text(self.surface, "Pad Base (Y=0)", self.canvas_rect.left + 10, self.pan_y - 12, size=11, color=(80, 110, 150))

        # Render Rocket Stack
        self._render_rocket_stack()

        # Render Live Center of Mass & Center of Pressure markers
        self._render_blueprint_markers()

        # Reset clip
        self.surface.set_clip(None)

    def _render_rocket_stack(self) -> None:
        """Render all stages and parts on the blueprint canvas."""
        for s in self.rocket.stages:
            stage_base = s.position_offset
            for p in s.parts:
                y_center = stage_base + p.position_offset
                l_half = p.length * 0.5
                r = p.radius

                # Screen bounding rect
                px = self.pan_x - r * self.scale
                py = self.pan_y - (y_center + l_half) * self.scale
                pw = 2.0 * r * self.scale
                ph = p.length * self.scale

                part_rect = pygame.Rect(int(px), int(py), int(pw), int(ph))
                is_selected = (p is self.selected_part)

                # Part visual style
                if isinstance(p, Engine):
                    # Bell trapezoid
                    throat_w = pw * 0.35
                    pts = [
                        (self.pan_x - throat_w * 0.5, py),
                        (px, py + ph),
                        (px + pw, py + ph),
                        (self.pan_x + throat_w * 0.5, py),
                    ]
                    color = (70, 75, 90) if not is_selected else (90, 120, 160)
                    pygame.draw.polygon(self.surface, color, pts)
                    pygame.draw.polygon(self.surface, ACCENT_CYAN if is_selected else (120, 130, 150), pts, width=2 if is_selected else 1)

                elif isinstance(p, Decoupler):
                    color = (48, 54, 66) if not is_selected else (70, 95, 130)
                    pygame.draw.rect(self.surface, color, part_rect)
                    pygame.draw.rect(self.surface, ACCENT_CYAN if is_selected else ACCENT_YELLOW, part_rect, width=2 if is_selected else 1)

                elif isinstance(p, Payload):
                    # Conical ogive
                    pts = [
                        (px, py + ph),
                        (px, py + ph * 0.3),
                        (self.pan_x, py),
                        (px + pw, py + ph * 0.3),
                        (px + pw, py + ph),
                    ]
                    color = (220, 228, 240) if not is_selected else (180, 215, 255)
                    pygame.draw.polygon(self.surface, color, pts)
                    pygame.draw.polygon(self.surface, ACCENT_CYAN if is_selected else (140, 155, 175), pts, width=2 if is_selected else 1)

                else:  # Fuel Tank
                    color = (205, 215, 228) if not is_selected else (170, 210, 250)
                    pygame.draw.rect(self.surface, color, part_rect, border_radius=4)
                    pygame.draw.rect(self.surface, ACCENT_CYAN if is_selected else (120, 135, 160), part_rect, width=2 if is_selected else 1, border_radius=4)

                # Label on part
                if ph > 18:
                    lbl_color = (20, 25, 35) if isinstance(p, (FuelTank, Payload)) else (230, 235, 245)
                    draw_text(self.surface, p.name, self.pan_x, py + ph * 0.5, size=11, color=lbl_color, bold=True, align="center")

    def _render_blueprint_markers(self) -> None:
        """Render CoM (Center of Mass) and CoP (Center of Pressure) markers."""
        if not self.rocket.stages or all(not s.parts for s in self.rocket.stages):
            return

        y_com = self.rocket.center_of_mass()
        y_cop = self.rocket.center_of_pressure()

        # 1. CoM (yellow & black divided circle)
        com_y = self.pan_y - y_com * self.scale
        com_x = self.pan_x + 55  # Offset to right of rocket centerline
        r = 9

        pygame.draw.circle(self.surface, (20, 20, 20), (int(com_x), int(com_y)), r)
        pygame.draw.circle(self.surface, ACCENT_YELLOW, (int(com_x), int(com_y)), r, width=2)
        pygame.draw.line(self.surface, ACCENT_YELLOW, (com_x - r, com_y), (com_x + r, com_y), width=2)
        pygame.draw.line(self.surface, ACCENT_YELLOW, (com_x, com_y - r), (com_x, com_y + r), width=2)
        draw_text(self.surface, f"CoM: {y_com:.2f}m", com_x + 14, com_y, size=11, color=ACCENT_YELLOW, bold=True)

        # 2. CoP (cyan & white bullseye circle)
        cop_y = self.pan_y - y_cop * self.scale
        cop_x = self.pan_x - 55  # Offset to left of rocket centerline
        pygame.draw.circle(self.surface, ACCENT_CYAN, (int(cop_x), int(cop_y)), r, width=2)
        pygame.draw.circle(self.surface, (255, 255, 255), (int(cop_x), int(cop_y)), 3)
        draw_text(self.surface, f"CoP: {y_cop:.2f}m", cop_x - 14, cop_y, size=11, color=ACCENT_CYAN, bold=True, align="right")

    def _render_catalog_panel(self) -> None:
        """Render left sidebar catalog."""
        pygame.draw.rect(self.surface, PANEL_BG, self.sidebar_left_rect)
        pygame.draw.rect(self.surface, PANEL_BORDER, self.sidebar_left_rect, width=1)

        draw_text(self.surface, "PART CATALOG", 16, 20, size=14, color=TEXT_WHITE, bold=True)
        draw_text(self.surface, "Click to spawn & stack", 16, 36, size=11, color=TEXT_MUTED)

        for btn in self.catalog_buttons:
            btn.draw(self.surface)

    def _render_top_toolbar(self) -> None:
        """Render top toolbar buttons."""
        pygame.draw.rect(self.surface, PANEL_BG, self.top_bar_rect)
        pygame.draw.line(self.surface, PANEL_BORDER, (self.top_bar_rect.left, self.top_bar_rect.bottom), (self.top_bar_rect.right, self.top_bar_rect.bottom), width=1)

        draw_text(self.surface, f"Rocket: {self.rocket.name}", self.top_bar_rect.left + 16, self.top_bar_rect.centery, size=13, color=TEXT_WHITE, bold=True)

        for btn in self.top_buttons:
            btn.draw(self.surface)

    def _render_engineering_panel(self) -> None:
        """Render right staging stack breakdown and engineering analysis metrics."""
        pygame.draw.rect(self.surface, PANEL_BG, self.sidebar_right_rect)
        pygame.draw.rect(self.surface, PANEL_BORDER, self.sidebar_right_rect, width=1)

        px = self.sidebar_right_rect.left + 14
        py = 18

        draw_text(self.surface, "ENGINEERING READOUT", px, py, size=14, color=TEXT_WHITE, bold=True)
        py += 26

        # Compute live stats
        stats = calculate_rocket_engineering_stats(self.rocket)

        # Vehicle Summary Box
        draw_text(self.surface, f"Total Mass: {format_mass(stats.total_mass)}", px, py, size=12, color=TEXT_WHITE, bold=True)
        py += 18
        draw_text(self.surface, f"Dry Mass:   {format_mass(stats.dry_mass)}", px, py, size=11, color=TEXT_MUTED)
        py += 16
        draw_text(self.surface, f"Propellant: {format_mass(stats.propellant_mass)}", px, py, size=11, color=TEXT_MUTED)
        py += 20

        # Liftoff TWR
        twr_color = ACCENT_GREEN if stats.stage0_twr_sl > 1.05 else (ACCENT_ORANGE if stats.stage0_twr_sl >= 1.0 else ACCENT_RED)
        draw_text(self.surface, f"Stage 0 SL TWR:  {stats.stage0_twr_sl:.2f}", px, py, size=13, color=twr_color, bold=True)
        py += 18
        if stats.stage0_twr_sl <= 1.0:
            draw_text(self.surface, "WARNING: TWR <= 1.0 (No Liftoff)", px, py, size=10, color=ACCENT_RED, bold=True)
            py += 16

        # Total Delta-V
        draw_text(self.surface, f"Total Delta-V:  {stats.total_delta_v:.0f} m/s", px, py, size=13, color=ACCENT_CYAN, bold=True)
        py += 26

        pygame.draw.line(self.surface, PANEL_BORDER, (px, py), (self.sidebar_right_rect.right - 14, py), width=1)
        py += 14

        draw_text(self.surface, "STAGING STACK", px, py, size=13, color=TEXT_WHITE, bold=True)
        py += 20

        for st_stat in stats.stages:
            st_box = pygame.Rect(px, py, self.sidebar_right_rect.width - 28, 64)
            pygame.draw.rect(self.surface, (28, 34, 48), st_box, border_radius=6)
            pygame.draw.rect(self.surface, PANEL_BORDER_LIGHT, st_box, width=1, border_radius=6)

            draw_text(self.surface, f"{st_stat.stage_name}", px + 8, py + 12, size=12, color=TEXT_WHITE, bold=True)
            draw_text(self.surface, f"Mass: {format_mass(st_stat.total_mass)}", px + 8, py + 30, size=11, color=TEXT_MUTED)
            draw_text(self.surface, f"TWR: {st_stat.twr_sl:.2f} (SL) / {st_stat.twr_vac:.2f} (Vac)", px + 8, py + 46, size=11, color=TEXT_MUTED)
            draw_text(self.surface, f"Dv: {st_stat.delta_v:.0f} m/s", st_box.right - 8, py + 12, size=12, color=ACCENT_CYAN, bold=True, align="right")
            py += 72

        # Draw Selected Part Inspector if a part is selected
        if self.selected_part is not None:
            py = max(py, 420)
            pygame.draw.line(self.surface, PANEL_BORDER, (px, py), (self.sidebar_right_rect.right - 14, py), width=1)
            py += 10
            draw_text(self.surface, "PART INSPECTOR", px, py, size=12, color=ACCENT_YELLOW, bold=True)
            py += 16
            draw_text(self.surface, f"Name: {self.selected_part.name}", px, py, size=11, color=TEXT_WHITE)
            py += 14
            draw_text(self.surface, f"Dry Mass: {self.selected_part.dry_mass:.0f} kg", px, py, size=11, color=TEXT_MUTED)
            py += 14
            draw_text(self.surface, f"Length: {self.selected_part.length:.1f} m, Rad: {self.selected_part.radius:.2f} m", px, py, size=11, color=TEXT_MUTED)
            py += 14
            if isinstance(self.selected_part, FuelTank):
                draw_text(self.surface, f"Fuel: {self.selected_part.propellant_mass:.0f} / {self.selected_part.propellant_capacity:.0f} kg", px, py, size=11, color=ACCENT_CYAN)
                py += 14
            elif isinstance(self.selected_part, Engine):
                draw_text(self.surface, f"Thrust SL: {format_thrust(self.selected_part.thrust_sl)}", px, py, size=11, color=ACCENT_ORANGE)
                py += 14
                draw_text(self.surface, f"Isp Vac: {self.selected_part.isp_vac:.0f} s", px, py, size=11, color=TEXT_MUTED)
                py += 14
            draw_text(self.surface, "[Del] Delete Part", px, py + 4, size=11, color=ACCENT_RED)

        # Action Buttons (LAUNCH)
        for btn in self.action_buttons:
            btn.draw(self.surface)


__all__ = [
    "CATALOG_TEMPLATES",
    "RocketEngineeringSummary",
    "StageEngineeringStats",
    "VABView",
    "calculate_rocket_engineering_stats",
]
