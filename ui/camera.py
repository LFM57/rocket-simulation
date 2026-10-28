"""64-bit high-precision camera system with anti-jitter transforms.

This module provides coordinate transformations between 64-bit planetary/orbital
world space (Earth radius R ~ 6.371e6 m) and 2D Pygame screen space.
To prevent 32-bit floating-point and integer jitter, all camera translations
and rotations are computed in 64-bit precision BEFORE scaling to screen pixels:
    Delta_r = r_world - r_cam
    r_frame = rotate(Delta_r, phi)
    x_screen = W/2 + r_frame.x * scale
    y_screen = H/2 - r_frame.y * scale
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Sequence, Union

import numpy as np

from core.vector2d import Vector2D
from physics.celestial import CelestialBody


class CameraMode(str, Enum):
    """Operational reference frame mode for camera orientation."""

    HORIZON_UP = "HORIZON_UP"
    INERTIAL = "INERTIAL"


class Camera:
    """High-precision 64-bit 2D camera supporting smooth zoom, pan, and tracking.

    Parameters
    ----------
    viewport_width : int, default=1280
        Width of the rendering viewport in pixels.
    viewport_height : int, default=720
        Height of the rendering viewport in pixels.
    position : Union[Vector2D, Sequence[float]], default=(0.0, 0.0)
        Initial camera world position in meters [m].
    scale : float, default=10.0
        Initial zoom scale in pixels per meter [px/m].
    min_scale : float, default=1e-8
        Minimum allowable zoom scale (deep space / solar system view).
    max_scale : float, default=500.0
        Maximum allowable zoom scale (extreme close-up on parts).
    mode : CameraMode, default=CameraMode.HORIZON_UP
        Active camera orientation mode.
    """

    def __init__(
        self,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        position: Union[Vector2D, Sequence[float]] = (0.0, 0.0),
        scale: float = 10.0,
        min_scale: float = 1e-8,
        max_scale: float = 500.0,
        mode: CameraMode = CameraMode.HORIZON_UP,
    ) -> None:
        self.viewport_width: int = max(1, int(viewport_width))
        self.viewport_height: int = max(1, int(viewport_height))

        self.position: Vector2D = (
            position if isinstance(position, Vector2D) else Vector2D.from_tuple(position)
        )
        self.target_position: Vector2D = self.position

        self.scale: float = float(np.float64(scale))
        self.target_scale: float = self.scale
        self.min_scale: float = float(np.float64(min_scale))
        self.max_scale: float = float(np.float64(max_scale))

        self.mode: CameraMode = mode if isinstance(mode, CameraMode) else CameraMode(mode)
        self.rotation_angle: float = 0.0  # Camera frame rotation angle phi [rad]
        self.target_rotation_angle: float = 0.0

        # Reference position for Horizon-Up mode (usually the craft's position)
        self.reference_position: Optional[Vector2D] = None

        # Tracking smoothing parameters
        self.position_smoothing: float = 12.0  # Spring-damper rate for position lerp
        self.zoom_smoothing: float = 15.0      # Exponential zoom lerp rate
        self.rotation_smoothing: float = 10.0  # Angle lerp rate

        # Initialize rotation immediately
        self._update_rotation(immediate=True)

    def set_viewport(self, width: int, height: int) -> None:
        """Update viewport pixel dimensions."""
        self.viewport_width = max(1, int(width))
        self.viewport_height = max(1, int(height))

    def set_mode(self, mode: Union[CameraMode, str]) -> None:
        """Set camera reference frame mode."""
        self.mode = mode if isinstance(mode, CameraMode) else CameraMode(str(mode).upper())
        self._update_rotation(immediate=True)

    def toggle_mode(self) -> CameraMode:
        """Toggle between HORIZON_UP and INERTIAL camera modes."""
        new_mode = (
            CameraMode.INERTIAL
            if self.mode == CameraMode.HORIZON_UP
            else CameraMode.HORIZON_UP
        )
        self.set_mode(new_mode)
        return self.mode

    def set_reference_position(
        self,
        pos: Optional[Union[Vector2D, Sequence[float]]],
        immediate: bool = False,
    ) -> None:
        """Set reference position (craft CoM) used for Horizon-Up alignment."""
        if pos is None:
            self.reference_position = None
        elif isinstance(pos, Vector2D):
            self.reference_position = pos
        else:
            self.reference_position = Vector2D.from_tuple(pos)
        self._update_rotation(immediate=immediate)

    def _update_rotation(self, immediate: bool = False) -> None:
        """Compute camera rotation angle based on current mode and reference position."""
        if self.mode == CameraMode.HORIZON_UP:
            # Use reference position, or fallback to camera position
            ref = self.reference_position if self.reference_position is not None else self.position
            ref_mag = ref.magnitude()
            if ref_mag > 1e-4:
                # Local outward radial direction theta_0 = atan2(y, x)
                # We want radial vector (local Up) to map to camera +Y (screen up, -Y_screen)
                # theta + phi = pi/2  =>  phi = pi/2 - theta_0
                theta_0 = math.atan2(ref.y, ref.x)
                self.target_rotation_angle = float(np.float64(0.5 * math.pi - theta_0))
            else:
                self.target_rotation_angle = 0.0
        else:
            # INERTIAL: World +X right, World +Y screen up (phi = 0)
            self.target_rotation_angle = 0.0

        if immediate:
            self.rotation_angle = self.target_rotation_angle

    def update(self, dt: float) -> None:
        """Advance smooth position, rotation, and zoom interpolations over time step dt."""
        dt_val = float(np.float64(max(0.0, dt)))
        if dt_val <= 0.0:
            return

        # 1. Smooth position tracking
        p_factor = 1.0 - math.exp(-self.position_smoothing * dt_val)
        delta_p = self.target_position - self.position
        self.position = self.position + delta_p * p_factor

        # 2. Smooth zoom interpolation (exponential in log-space)
        z_factor = 1.0 - math.exp(-self.zoom_smoothing * dt_val)
        ln_scale = math.log(max(self.min_scale, self.scale))
        ln_target = math.log(max(self.min_scale, self.target_scale))
        ln_new = ln_scale + (ln_target - ln_scale) * z_factor
        self.scale = float(np.float64(math.exp(ln_new)))

        # 3. Smooth angle rotation interpolation (wrapping around 2*pi)
        self._update_rotation(immediate=False)
        r_factor = 1.0 - math.exp(-self.rotation_smoothing * dt_val)
        d_ang = math.atan2(
            math.sin(self.target_rotation_angle - self.rotation_angle),
            math.cos(self.target_rotation_angle - self.rotation_angle),
        )
        self.rotation_angle = float(np.float64(self.rotation_angle + d_ang * r_factor))

    def snap_to(self, position: Union[Vector2D, Sequence[float]]) -> None:
        """Snap camera position immediately without smoothing."""
        pos = position if isinstance(position, Vector2D) else Vector2D.from_tuple(position)
        self.position = pos
        self.target_position = pos
        self._update_rotation(immediate=True)

    def snap_scale(self, scale: float) -> None:
        """Snap camera scale immediately without smoothing."""
        clamped = max(self.min_scale, min(self.max_scale, float(scale)))
        self.scale = clamped
        self.target_scale = clamped

    def set_target(self, position: Union[Vector2D, Sequence[float]]) -> None:
        """Set desired target position for smooth camera tracking."""
        pos = position if isinstance(position, Vector2D) else Vector2D.from_tuple(position)
        self.target_position = pos

    def zoom_in(self, factor: float = 1.3) -> None:
        """Zoom in smoothly by a factor."""
        self.target_scale = min(self.max_scale, self.target_scale * float(factor))

    def zoom_out(self, factor: float = 1.3) -> None:
        """Zoom out smoothly by a factor."""
        self.target_scale = max(self.min_scale, self.target_scale / float(factor))

    def zoom_at_screen_point(
        self,
        factor: float,
        screen_pos: tuple[float, float],
    ) -> None:
        """Zoom toward/away from a specific screen coordinate (e.g. mouse cursor)."""
        world_before = self.screen_to_world(screen_pos)
        old_target_scale = self.target_scale
        new_target_scale = max(
            self.min_scale, min(self.max_scale, old_target_scale * float(factor))
        )
        self.target_scale = new_target_scale
        # Immediately adjust target position so world_before remains under the cursor
        self.scale = new_target_scale
        world_after = self.screen_to_world(screen_pos)
        self.target_position = self.target_position + (world_before - world_after)
        self.position = self.target_position

    # --- Core Coordinate Transformations ---

    def world_to_screen(
        self,
        world_pos: Union[Vector2D, Sequence[float]],
    ) -> tuple[float, float]:
        """Transform 64-bit world coordinates to 2D Pygame screen pixel coordinates.

        Anti-Jitter Pipeline:
            1. Delta_r = world_pos - camera.position (computed in float64)
            2. r_cam = rotate(Delta_r, phi)
            3. x_screen = W/2 + r_cam.x * scale
            4. y_screen = H/2 - r_cam.y * scale

        Parameters
        ----------
        world_pos : Vector2D or Sequence[float]
            Inertial 64-bit coordinates [m].

        Returns
        -------
        tuple[float, float]
            Pixel coordinates (x_screen, y_screen).
        """
        pos = world_pos if isinstance(world_pos, Vector2D) else Vector2D.from_tuple(world_pos)

        # 1. 64-bit camera-relative offset
        delta_r = pos - self.position

        # 2. Rotate into camera frame
        r_cam = delta_r.rotate(self.rotation_angle) if abs(self.rotation_angle) > 1e-12 else delta_r

        # 3. Project to screen viewport
        half_w = 0.5 * self.viewport_width
        half_h = 0.5 * self.viewport_height
        x_screen = half_w + r_cam.x * self.scale
        y_screen = half_h - r_cam.y * self.scale

        return float(x_screen), float(y_screen)

    def screen_to_world(
        self,
        screen_pos: tuple[float, float],
    ) -> Vector2D:
        """Transform 2D Pygame screen pixel coordinates back to 64-bit world coordinates.

        Inverse Pipeline:
            1. r_cam.x = (x_screen - W/2) / scale
            2. r_cam.y = - (y_screen - H/2) / scale
            3. Delta_r = rotate(r_cam, -phi)
            4. world_pos = camera.position + Delta_r

        Parameters
        ----------
        screen_pos : tuple[float, float]
            Pixel coordinates (x_screen, y_screen).

        Returns
        -------
        Vector2D
            Inertial 64-bit coordinates [m].
        """
        half_w = 0.5 * self.viewport_width
        half_h = 0.5 * self.viewport_height

        x_cam = (float(screen_pos[0]) - half_w) / self.scale
        y_cam = -(float(screen_pos[1]) - half_h) / self.scale
        r_cam = Vector2D(x_cam, y_cam)

        delta_r = (
            r_cam.rotate(-self.rotation_angle)
            if abs(self.rotation_angle) > 1e-12
            else r_cam
        )
        return self.position + delta_r

    def world_to_screen_rel(
        self,
        rel_vec: Union[Vector2D, Sequence[float]],
    ) -> tuple[float, float]:
        """Rotate and scale a relative displacement vector (e.g. part offset) into screen pixels."""
        vec = rel_vec if isinstance(rel_vec, Vector2D) else Vector2D.from_tuple(rel_vec)
        v_cam = vec.rotate(self.rotation_angle) if abs(self.rotation_angle) > 1e-12 else vec
        return float(v_cam.x * self.scale), float(-v_cam.y * self.scale)

    def world_vector_to_screen_angle(self, vector: Vector2D) -> float:
        """Return the angle [rad] of a world vector in screen coordinates (0 = screen right)."""
        v_cam = vector.rotate(self.rotation_angle) if abs(self.rotation_angle) > 1e-12 else vector
        # Screen Y points down, so angle on screen is atan2(-y_cam, x_cam)
        return math.atan2(-v_cam.y, v_cam.x)

    def is_world_point_visible(
        self,
        world_pos: Vector2D,
        margin_px: float = 100.0,
    ) -> bool:
        """Check whether a given world coordinate is within the visible screen viewport."""
        sx, sy = self.world_to_screen(world_pos)
        return (
            -margin_px <= sx <= self.viewport_width + margin_px
            and -margin_px <= sy <= self.viewport_height + margin_px
        )

    # --- Planetary Horizon Arc Generation (Anti-Overflow) ---

    def get_visible_horizon_points(
        self,
        body: CelestialBody,
        num_points: int = 80,
    ) -> list[Vector2D]:
        """Compute world coordinates along the visible planetary surface arc.

        At high zoom levels (e.g. scale = 20 px/m), Earth's radius (6.371e6 m)
        corresponds to 1.27e8 pixels. Drawing a full Pygame circle of that radius
        causes integer overflow and clipping artifacts.

        This method analytically samples angular coordinates delta_theta around
        the craft / camera ground track within the visible camera viewport,
        ensuring smooth, accurate, and memory-safe planetary horizon rendering.

        Parameters
        ----------
        body : CelestialBody
            Primary celestial body (e.g. Earth).
        num_points : int, default=80
            Number of vertices to sample along the visible arc.

        Returns
        -------
        list[Vector2D]
            Ordered list of points on the planetary surface in world space.
        """
        r_planet = float(body.radius)
        ref = self.reference_position if self.reference_position is not None else self.position
        ref_mag = ref.magnitude()
        if ref_mag < 1e-4:
            ref = Vector2D(0.0, r_planet)

        theta_center = math.atan2(ref.y, ref.x)

        # Estimate half-width of the visible screen in meters
        half_screen_m = (0.5 * max(self.viewport_width, self.viewport_height)) / max(
            1e-8, self.scale
        )
        # Add safety margin to cover full viewport width
        span_m = half_screen_m * 2.5
        d_theta = min(math.pi * 0.95, max(1e-4, span_m / max(1.0, r_planet)))

        angles = np.linspace(theta_center - d_theta, theta_center + d_theta, num_points)
        body_pos = body.position

        points: list[Vector2D] = []
        for ang in angles:
            p = body_pos + Vector2D(r_planet * math.cos(ang), r_planet * math.sin(ang))
            points.append(p)

        return points


__all__ = [
    "Camera",
    "CameraMode",
]
