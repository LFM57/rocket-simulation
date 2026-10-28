"""Reusable SFS-style UI widgets, panels, buttons, sliders, and graph canvases.

Built using procedural Pygame primitives with a modern aerospace dark theme.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import pygame

# ============================================================================
# Color Palette & Theme Constants
# ============================================================================

BG_DARK: tuple[int, int, int] = (12, 14, 20)
BG_BLUEPRINT: tuple[int, int, int] = (16, 22, 34)

PANEL_BG: tuple[int, int, int] = (22, 27, 38)
PANEL_BG_ALPHA: tuple[int, int, int, int] = (22, 27, 38, 220)
PANEL_BORDER: tuple[int, int, int] = (45, 56, 78)
PANEL_BORDER_LIGHT: tuple[int, int, int] = (70, 85, 120)

GRID_LINE: tuple[int, int, int] = (26, 36, 52)
GRID_MAJOR: tuple[int, int, int] = (40, 54, 78)

TEXT_WHITE: tuple[int, int, int] = (245, 248, 255)
TEXT_MUTED: tuple[int, int, int] = (145, 160, 185)
TEXT_DIM: tuple[int, int, int] = (95, 110, 130)

ACCENT_CYAN: tuple[int, int, int] = (0, 200, 255)
ACCENT_GREEN: tuple[int, int, int] = (45, 215, 110)
ACCENT_ORANGE: tuple[int, int, int] = (255, 155, 45)
ACCENT_RED: tuple[int, int, int] = (255, 75, 75)
ACCENT_YELLOW: tuple[int, int, int] = (255, 220, 50)

BUTTON_NORMAL: tuple[int, int, int] = (34, 42, 60)
BUTTON_HOVER: tuple[int, int, int] = (50, 64, 90)
BUTTON_ACTIVE: tuple[int, int, int] = (0, 140, 210)
BUTTON_DISABLED: tuple[int, int, int] = (24, 28, 38)

# ============================================================================
# Font Cache
# ============================================================================

_FONT_CACHE: dict[tuple[int, bool, bool], pygame.font.Font] = {}


def clear_font_cache() -> None:
    """Clear cached font objects."""
    _FONT_CACHE.clear()


def get_font(size: int = 14, bold: bool = False, mono: bool = False) -> pygame.font.Font:
    """Retrieve cached pygame.font.Font instance."""
    if not pygame.font.get_init():
        pygame.font.init()
        _FONT_CACHE.clear()

    key = (size, bold, mono)
    if key not in _FONT_CACHE:
        try:
            if mono:
                font_name = pygame.font.match_font("monospace", bold=bold)
            else:
                font_name = pygame.font.match_font("dejavusans,segoeui,arial,helvetica", bold=bold)

            if font_name:
                _FONT_CACHE[key] = pygame.font.Font(font_name, size)
            else:
                _FONT_CACHE[key] = pygame.font.Font(None, size)
        except Exception:
            _FONT_CACHE[key] = pygame.font.Font(None, size)

    return _FONT_CACHE[key]


# ============================================================================
# Text Formatting Utilities
# ============================================================================


def format_altitude(alt_m: float) -> str:
    """Format altitude with adaptive units (m vs km)."""
    if math.isinf(alt_m):
        return "Escaping (inf)"
    if alt_m < 0.0:
        return "0 m"
    if alt_m < 1000.0:
        return f"{alt_m:.1f} m"
    if alt_m < 1_000_000.0:
        return f"{alt_m / 1000.0:.2f} km"
    return f"{alt_m / 1_000_000.0:.3f} Mm"


def format_speed(speed_mps: float) -> str:
    """Format speed with adaptive units (m/s vs km/s)."""
    if speed_mps < 1000.0:
        return f"{speed_mps:.1f} m/s"
    return f"{speed_mps / 1000.0:.3f} km/s"


def format_mass(mass_kg: float) -> str:
    """Format mass with adaptive units (kg vs t)."""
    if mass_kg < 1000.0:
        return f"{mass_kg:.1f} kg"
    return f"{mass_kg / 1000.0:.2f} t"


def format_thrust(thrust_n: float) -> str:
    """Format thrust in kN or MN."""
    if thrust_n < 1_000.0:
        return f"{thrust_n:.0f} N"
    if thrust_n < 1_000_000.0:
        return f"{thrust_n / 1000.0:.1f} kN"
    return f"{thrust_n / 1_000_000.0:.2f} MN"


def format_time(seconds: float) -> str:
    """Format elapsed mission time as T+ HH:MM:SS.s."""
    sec = max(0.0, float(seconds))
    hrs = int(sec // 3600)
    mins = int((sec % 3600) // 60)
    s = sec % 60
    if hrs > 0:
        return f"T+ {hrs:02d}:{mins:02d}:{s:04.1f}"
    return f"T+ {mins:02d}:{s:04.1f}"


def draw_text(
    surface: pygame.Surface,
    text: str,
    x: float,
    y: float,
    size: int = 14,
    color: tuple[int, int, int] = TEXT_WHITE,
    bold: bool = False,
    mono: bool = False,
    align: str = "left",
) -> pygame.Rect:
    """Draw text on surface with alignment ('left', 'center', 'right')."""
    font = get_font(size=size, bold=bold, mono=mono)
    text_surf = font.render(str(text), True, color)
    w, h = text_surf.get_size()

    rx = x
    if align == "center":
        rx = x - w * 0.5
    elif align == "right":
        rx = x - w

    ry = y - h * 0.5
    rect = pygame.Rect(int(rx), int(ry), w, h)
    surface.blit(text_surf, rect)
    return rect


# ============================================================================
# Interactive UI Widgets
# ============================================================================


class Button:
    """SFS-style sleek push or toggle button."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        shortcut: str = "",
        on_click: Optional[Callable[[], None]] = None,
        is_toggle: bool = False,
        is_active: bool = False,
        bg_color: tuple[int, int, int] = BUTTON_NORMAL,
        active_color: tuple[int, int, int] = BUTTON_ACTIVE,
        text_color: tuple[int, int, int] = TEXT_WHITE,
        border_radius: int = 6,
        font_size: int = 14,
        tooltip: str = "",
    ) -> None:
        self.rect: pygame.Rect = pygame.Rect(rect)
        self.text: str = str(text)
        self.shortcut: str = str(shortcut)
        self.on_click: Optional[Callable[[], None]] = on_click
        self.is_toggle: bool = is_toggle
        self.is_active: bool = is_active
        self.is_enabled: bool = True
        self.is_hovered: bool = False
        self.is_pressed: bool = False

        self.bg_color: tuple[int, int, int] = bg_color
        self.active_color: tuple[int, int, int] = active_color
        self.text_color: tuple[int, int, int] = text_color
        self.border_radius: int = border_radius
        self.font_size: int = font_size
        self.tooltip: str = tooltip

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse hover and click events. Returns True if clicked."""
        if not self.is_enabled:
            self.is_hovered = False
            self.is_pressed = False
            return False

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_pressed and self.rect.collidepoint(event.pos):
                self.is_pressed = False
                if self.is_toggle:
                    self.is_active = not self.is_active
                if self.on_click is not None:
                    self.on_click()
                return True
            self.is_pressed = False

        return False

    def draw(self, surface: pygame.Surface) -> None:
        """Render button to target surface."""
        if not self.is_enabled:
            color = BUTTON_DISABLED
            border_c = PANEL_BORDER
            txt_c = TEXT_DIM
        elif self.is_active:
            color = self.active_color
            border_c = ACCENT_CYAN
            txt_c = (255, 255, 255)
        elif self.is_pressed:
            color = (
                max(0, self.bg_color[0] - 15),
                max(0, self.bg_color[1] - 15),
                max(0, self.bg_color[2] - 15),
            )
            border_c = ACCENT_CYAN
            txt_c = self.text_color
        elif self.is_hovered:
            color = BUTTON_HOVER
            border_c = PANEL_BORDER_LIGHT
            txt_c = (255, 255, 255)
        else:
            color = self.bg_color
            border_c = PANEL_BORDER
            txt_c = self.text_color

        pygame.draw.rect(surface, color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, border_c, self.rect, width=1, border_radius=self.border_radius)

        # Draw text & optional shortcut hint
        cx = self.rect.centerx
        cy = self.rect.centery
        if self.shortcut:
            draw_text(surface, self.text, cx, cy - 6, size=self.font_size, color=txt_c, bold=True, align="center")
            draw_text(surface, self.shortcut, cx, cy + 8, size=10, color=TEXT_MUTED, align="center")
        else:
            draw_text(surface, self.text, cx, cy, size=self.font_size, color=txt_c, bold=True, align="center")


class VerticalSlider:
    r"""Interactive vertical throttle slider lever ($0\% \to 100\%$)."""

    def __init__(
        self,
        rect: pygame.Rect,
        value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        label: str = "THROTTLE",
        on_change: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.rect: pygame.Rect = pygame.Rect(rect)
        self.value: float = max(min_value, min(max_value, float(value)))
        self.min_value: float = float(min_value)
        self.max_value: float = float(max_value)
        self.label: str = label
        self.on_change: Optional[Callable[[float], None]] = on_change
        self.is_dragging: bool = False
        self.is_hovered: bool = False

    def set_value(self, val: float, trigger_callback: bool = True) -> None:
        """Set slider value programmatically."""
        clamped = max(self.min_value, min(self.max_value, float(val)))
        if clamped != self.value:
            self.value = clamped
            if trigger_callback and self.on_change is not None:
                self.on_change(self.value)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse drag events."""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            if self.is_dragging:
                self._update_from_mouse_y(event.pos[1])
                return True

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_dragging = True
                self._update_from_mouse_y(event.pos[1])
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_dragging:
                self.is_dragging = False
                return True

        return False

    def _update_from_mouse_y(self, mouse_y: int) -> None:
        """Calculate fractional value from mouse Y position."""
        track_top = self.rect.top + 20
        track_bottom = self.rect.bottom - 20
        track_h = max(1, track_bottom - track_top)

        # Clamped fractional progress from bottom to top (bottom = 0, top = 1)
        progress = max(0.0, min(1.0, (track_bottom - mouse_y) / track_h))
        new_val = self.min_value + progress * (self.max_value - self.min_value)
        self.set_value(new_val)

    def draw(self, surface: pygame.Surface) -> None:
        """Render vertical throttle track, fill bar, grip handle, and readout."""
        # Panel background
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=8)

        # Header label
        draw_text(
            surface,
            self.label,
            self.rect.centerx,
            self.rect.top + 10,
            size=11,
            color=TEXT_MUTED,
            bold=True,
            align="center",
        )

        # Track bounds
        track_top = self.rect.top + 26
        track_bottom = self.rect.bottom - 26
        track_h = track_bottom - track_top
        track_w = 12
        track_x = self.rect.centerx - track_w // 2

        track_rect = pygame.Rect(track_x, track_top, track_w, track_h)
        pygame.draw.rect(surface, (18, 22, 32), track_rect, border_radius=4)
        pygame.draw.rect(surface, PANEL_BORDER, track_rect, width=1, border_radius=4)

        # Fill bar
        frac = (self.value - self.min_value) / max(1e-6, self.max_value - self.min_value)
        fill_h = int(track_h * frac)
        if fill_h > 0:
            fill_rect = pygame.Rect(track_x + 1, track_bottom - fill_h, track_w - 2, fill_h)
            # Interpolate fill color from cyan to bright orange
            fill_color = ACCENT_CYAN if frac < 0.8 else ACCENT_ORANGE
            pygame.draw.rect(surface, fill_color, fill_rect, border_radius=3)

        # Handle knob
        handle_y = track_bottom - int(track_h * frac)
        handle_w = self.rect.width - 16
        handle_h = 16
        handle_rect = pygame.Rect(
            self.rect.centerx - handle_w // 2,
            handle_y - handle_h // 2,
            handle_w,
            handle_h,
        )

        h_color = BUTTON_HOVER if (self.is_dragging or self.is_hovered) else BUTTON_NORMAL
        border_c = ACCENT_CYAN if (self.is_dragging or self.is_hovered) else PANEL_BORDER_LIGHT
        pygame.draw.rect(surface, h_color, handle_rect, border_radius=4)
        pygame.draw.rect(surface, border_c, handle_rect, width=1, border_radius=4)

        # Grip tick lines on handle
        pygame.draw.line(
            surface,
            TEXT_MUTED,
            (handle_rect.centerx - 8, handle_rect.centery),
            (handle_rect.centerx + 8, handle_rect.centery),
            width=2,
        )

        # Digital readout at bottom
        pct_text = f"{int(frac * 100)}%"
        draw_text(
            surface,
            pct_text,
            self.rect.centerx,
            self.rect.bottom - 12,
            size=13,
            color=TEXT_WHITE,
            bold=True,
            mono=True,
            align="center",
        )


class ProgressBar:
    """Horizontal status or propellant level bar with color gradient."""

    def __init__(
        self,
        rect: pygame.Rect,
        value: float = 1.0,
        max_value: float = 1.0,
        label: str = "",
        show_percentage: bool = True,
        fill_color: tuple[int, int, int] = ACCENT_GREEN,
    ) -> None:
        self.rect: pygame.Rect = pygame.Rect(rect)
        self.value: float = float(value)
        self.max_value: float = max(1e-6, float(max_value))
        self.label: str = label
        self.show_percentage: bool = show_percentage
        self.fill_color: tuple[int, int, int] = fill_color

    def set_value(self, val: float, max_val: Optional[float] = None) -> None:
        """Update current value and optional max capacity."""
        if max_val is not None:
            self.max_value = max(1e-6, float(max_val))
        self.value = max(0.0, min(self.max_value, float(val)))

    def draw(self, surface: pygame.Surface) -> None:
        """Render gauge bar with percentage."""
        pygame.draw.rect(surface, (18, 22, 32), self.rect, border_radius=4)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=4)

        frac = max(0.0, min(1.0, self.value / self.max_value))
        fill_w = int((self.rect.width - 2) * frac)
        if fill_w > 0:
            fill_rect = pygame.Rect(self.rect.left + 1, self.rect.top + 1, fill_w, self.rect.height - 2)
            # Adaptive color: green (>40%), yellow (15-40%), red (<15%)
            if frac > 0.4:
                c = ACCENT_GREEN
            elif frac > 0.15:
                c = ACCENT_YELLOW
            else:
                c = ACCENT_RED
            pygame.draw.rect(surface, c, fill_rect, border_radius=3)

        # Label & Text overlay
        txt = self.label
        if self.show_percentage:
            pct_str = f"{frac * 100.0:.1f}%"
            txt = f"{self.label}: {pct_str}" if self.label else pct_str

        if txt:
            f_size = 11
            font = get_font(size=f_size, bold=True)
            while font.size(txt)[0] > self.rect.width - 8 and f_size > 8:
                f_size -= 1
                font = get_font(size=f_size, bold=True)
            draw_text(
                surface,
                txt,
                self.rect.centerx,
                self.rect.centery,
                size=f_size,
                color=TEXT_WHITE,
                bold=True,
                align="center",
            )


class Panel:
    """Semi-transparent container HUD panel."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "",
        bg_alpha: int = 220,
    ) -> None:
        self.rect: pygame.Rect = pygame.Rect(rect)
        self.title: str = title
        self.bg_alpha: int = bg_alpha

    def draw(self, surface: pygame.Surface) -> None:
        """Render dark rounded HUD panel."""
        panel_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        panel_surf.fill((PANEL_BG[0], PANEL_BG[1], PANEL_BG[2], self.bg_alpha))
        surface.blit(panel_surf, self.rect.topleft)

        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=8)

        if self.title:
            # Title header separator
            header_h = 24
            header_rect = pygame.Rect(self.rect.left, self.rect.top, self.rect.width, header_h)
            pygame.draw.line(
                surface,
                PANEL_BORDER,
                (self.rect.left, self.rect.top + header_h),
                (self.rect.right, self.rect.top + header_h),
                width=1,
            )
            draw_text(
                surface,
                self.title,
                self.rect.left + 10,
                self.rect.top + header_h // 2,
                size=12,
                color=TEXT_WHITE,
                bold=True,
                align="left",
            )


class PlotCanvas:
    """Compact high-performance time-series plot canvas for telemetry analysis."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str,
        unit: str = "",
        line_color: tuple[int, int, int] = ACCENT_CYAN,
    ) -> None:
        self.rect: pygame.Rect = pygame.Rect(rect)
        self.title: str = title
        self.unit: str = unit
        self.line_color: tuple[int, int, int] = line_color

    def draw(
        self,
        surface: pygame.Surface,
        x_values: Sequence[float],
        y_values: Sequence[float],
    ) -> None:
        """Render axes, labels, and curve from series data."""
        # Background box
        pygame.draw.rect(surface, (16, 20, 28), self.rect, border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=6)

        # Plot margins
        pad_l = 48
        pad_r = 16
        pad_t = 28
        pad_b = 24

        plot_w = max(10, self.rect.width - pad_l - pad_r)
        plot_h = max(10, self.rect.height - pad_t - pad_b)
        origin_x = self.rect.left + pad_l
        origin_y = self.rect.top + pad_t

        # Title
        unit_str = f" ({self.unit})" if self.unit else ""
        draw_text(
            surface,
            f"{self.title}{unit_str}",
            self.rect.left + 8,
            self.rect.top + 14,
            size=12,
            color=TEXT_WHITE,
            bold=True,
            align="left",
        )

        if not x_values or not y_values or len(x_values) != len(y_values) or len(x_values) < 2:
            draw_text(
                surface,
                "No telemetry data",
                origin_x + plot_w // 2,
                origin_y + plot_h // 2,
                size=12,
                color=TEXT_MUTED,
                align="center",
            )
            return

        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)

        if math.isclose(max_x, min_x):
            max_x = min_x + 1.0
        if math.isclose(max_y, min_y):
            max_y = min_y + 1.0

        # Draw grid lines (3 horizontal)
        for i in range(3):
            gy = origin_y + int(plot_h * (i / 2.0))
            pygame.draw.line(surface, GRID_LINE, (origin_x, gy), (origin_x + plot_w, gy), width=1)
            val_y = max_y - i * (max_y - min_y) * 0.5
            val_str = f"{val_y:.1f}" if abs(val_y) < 1000.0 else f"{val_y/1000.0:.1f}k"
            draw_text(surface, val_str, origin_x - 6, gy, size=10, color=TEXT_MUTED, mono=True, align="right")

        # Draw time X ticks (start, end)
        draw_text(surface, f"{min_x:.0f}s", origin_x, origin_y + plot_h + 10, size=10, color=TEXT_MUTED, align="left")
        draw_text(surface, f"{max_x:.0f}s", origin_x + plot_w, origin_y + plot_h + 10, size=10, color=TEXT_MUTED, align="right")

        # Subsample line points to keep rendering fast
        step = max(1, len(x_values) // plot_w)
        indices = list(range(0, len(x_values), step))
        if indices[-1] != len(x_values) - 1:
            indices.append(len(x_values) - 1)

        points: list[tuple[int, int]] = []
        for idx in indices:
            px = origin_x + int(((x_values[idx] - min_x) / (max_x - min_x)) * plot_w)
            py = origin_y + plot_h - int(((y_values[idx] - min_y) / (max_y - min_y)) * plot_h)
            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(surface, self.line_color, False, points, width=2)

        # Peak indicator in top right
        peak_str = f"Peak: {max_y:.1f} {self.unit}" if abs(max_y) < 1000.0 else f"Peak: {max_y/1000.0:.2f} k{self.unit}"
        draw_text(
            surface,
            peak_str,
            self.rect.right - 10,
            self.rect.top + 14,
            size=11,
            color=self.line_color,
            bold=True,
            align="right",
        )


class ToastManager:
    """Manages temporary floating toast notification banners."""

    def __init__(self) -> None:
        self.message: str = ""
        self.timer: float = 0.0
        self.duration: float = 3.0
        self.color: tuple[int, int, int] = ACCENT_ORANGE

    def show(self, message: str, duration: float = 3.0, color: tuple[int, int, int] = ACCENT_ORANGE) -> None:
        """Trigger a toast message."""
        self.message = str(message)
        self.duration = float(duration)
        self.timer = self.duration
        self.color = color

    def update(self, dt: float) -> None:
        """Advance toast timer."""
        if self.timer > 0.0:
            self.timer = max(0.0, self.timer - dt)
            if self.timer == 0.0:
                self.message = ""

    def draw(self, surface: pygame.Surface) -> None:
        """Render toast banner at top center of viewport if active."""
        if not self.message or self.timer <= 0.0:
            return

        font = get_font(size=14, bold=True)
        txt_surf = font.render(self.message, True, TEXT_WHITE)
        tw, th = txt_surf.get_size()

        bw = tw + 32
        bh = th + 16
        sw = surface.get_width()
        bx = (sw - bw) // 2
        by = 50

        # Alpha fade out during the last 0.5 seconds
        alpha = 240
        if self.timer < 0.5:
            alpha = int(240 * (self.timer / 0.5))

        banner_surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
        banner_surf.fill((22, 28, 40, alpha))
        surface.blit(banner_surf, (bx, by))

        rect = pygame.Rect(bx, by, bw, bh)
        pygame.draw.rect(surface, self.color, rect, width=1, border_radius=6)

        surface.blit(txt_surf, (bx + 16, by + 8))


__all__ = [
    "ACCENT_CYAN",
    "ACCENT_GREEN",
    "ACCENT_ORANGE",
    "ACCENT_RED",
    "ACCENT_YELLOW",
    "BG_BLUEPRINT",
    "BG_DARK",
    "BUTTON_ACTIVE",
    "BUTTON_DISABLED",
    "BUTTON_HOVER",
    "BUTTON_NORMAL",
    "Button",
    "GRID_LINE",
    "GRID_MAJOR",
    "PANEL_BG",
    "PANEL_BG_ALPHA",
    "PANEL_BORDER",
    "PANEL_BORDER_LIGHT",
    "Panel",
    "PlotCanvas",
    "ProgressBar",
    "TEXT_DIM",
    "TEXT_MUTED",
    "TEXT_WHITE",
    "ToastManager",
    "VerticalSlider",
    "draw_text",
    "format_altitude",
    "format_mass",
    "format_speed",
    "format_thrust",
    "format_time",
    "get_font",
]
