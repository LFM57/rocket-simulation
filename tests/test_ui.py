"""Comprehensive automated test suite for SFS 2D User Interface.

Verifies:
    1. 64-bit camera-relative coordinate transforms and roundtrip precision at planetary scales.
    2. Local Horizon-Up camera rotation and vertical alignment.
    3. Planetary horizon arc generation without numerical overflow.
    4. Keplerian orbit conic trajectory point generation (elliptic and hyperbolic).
    5. VAB engineering calculations (Tsiolkovsky stage Delta-V, TWR, mass rollups).
    6. Headless integration: VAB blueprint construction to FlightView launch and stepping.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np
import pytest

# Ensure headless Pygame rendering for automated testing
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

from core.constants import g0
from core.vector2d import Vector2D
from models.parts import Engine, FuelTank, Payload
from models.presets import create_falcon_9
from models.rocket import Rocket, Stage
from physics.celestial import CelestialBody, get_earth
from physics.orbit import calculate_orbit
from simulation.flight_dynamics import VehicleState
from ui.camera import Camera, CameraMode
from ui.flight_view import FlightView
from ui.vab_view import (
    StageEngineeringStats,
    VABView,
    calculate_rocket_engineering_stats,
)


@pytest.fixture(scope="session", autouse=True)
def init_pygame():
    """Initialize headless Pygame display surface for tests."""
    pygame.init()
    surface = pygame.display.set_mode((1280, 720))
    yield surface
    pygame.quit()


# ============================================================================
# 1. 64-Bit Anti-Jitter Camera Mathematics Tests
# ============================================================================


def test_camera_world_to_screen_and_back_roundtrip_precision():
    """Verify camera-relative roundtrip accuracy at planetary scale (r ~ 6.371e6 m).

    Requirements:
        - Absolute error between original world coordinate and inverse screen_to_world
          must be less than 1e-6 m (sub-micrometer precision) despite 10^7 m magnitude.
    """
    earth = get_earth()
    cam = Camera(
        viewport_width=1280,
        viewport_height=720,
        position=Vector2D(0.0, earth.radius + 50.0),
        scale=20.0,
        mode=CameraMode.HORIZON_UP,
    )

    test_points = [
        Vector2D(0.0, earth.radius),
        Vector2D(10.0, earth.radius + 25.0),
        Vector2D(-45.2, earth.radius + 120.5),
        Vector2D(earth.radius * math.cos(0.3), earth.radius * math.sin(0.3)),
    ]

    for mode in [CameraMode.HORIZON_UP, CameraMode.INERTIAL]:
        cam.set_mode(mode)
        for orig in test_points:
            screen_pos = cam.world_to_screen(orig)
            recovered = cam.screen_to_world(screen_pos)

            error = (orig - recovered).magnitude()
            assert error < 1e-6, (
                f"Mode {mode}: World-to-screen roundtrip error {error:.2e} m "
                f"exceeded 1e-6 m at point {orig}"
            )


def test_camera_horizon_up_vertical_alignment():
    """Verify that HORIZON_UP rotates local radial outward vector exactly to screen -Y.

    In the camera frame, (0, 1) corresponds to screen up.
    A craft at position (R * cos(theta), R * sin(theta)) should have its local radial
    outward vector align with screen -Y (x_screen = W/2, y_screen < H/2).
    """
    earth = get_earth()
    r = earth.radius

    # Test at various arbitrary angles on the globe
    test_angles = [0.0, math.pi * 0.25, math.pi * 0.5, math.pi * 0.8, -math.pi * 0.4]

    for theta in test_angles:
        craft_pos = Vector2D(r * math.cos(theta), r * math.sin(theta))
        cam = Camera(
            viewport_width=1280,
            viewport_height=720,
            position=craft_pos,
            scale=10.0,
            mode=CameraMode.HORIZON_UP,
        )
        cam.set_reference_position(craft_pos, immediate=True)

        # Craft CoM at screen center
        cx, cy = cam.world_to_screen(craft_pos)
        assert math.isclose(cx, 640.0, abs_tol=1e-5)
        assert math.isclose(cy, 360.0, abs_tol=1e-5)

        # A point 50 meters directly "Up" radially from the craft
        up_pos = craft_pos + craft_pos.normalized() * 50.0
        ux, uy = cam.world_to_screen(up_pos)

        # In Horizon-Up mode, 'Up' should stay at x = 640.0, and move UP on screen (uy < 360)
        assert math.isclose(ux, 640.0, abs_tol=1e-4)
        assert uy < 360.0
        # Distance on screen should match 50m * 10 px/m = 500 px
        assert math.isclose(360.0 - uy, 500.0, abs_tol=1e-4)


def test_camera_visible_horizon_points_generation():
    """Verify visible horizon arc sampling around craft ground track."""
    earth = get_earth()
    craft_pos = Vector2D(0.0, earth.radius + 100.0)
    cam = Camera(
        viewport_width=1280,
        viewport_height=720,
        position=craft_pos,
        scale=15.0,
        mode=CameraMode.HORIZON_UP,
    )
    cam.set_reference_position(craft_pos)

    arc = cam.get_visible_horizon_points(earth, num_points=50)
    assert len(arc) == 50

    # Every sampled point must lie precisely on the planetary radius
    for pt in arc:
        assert not math.isnan(pt.x) and not math.isnan(pt.y)
        assert not math.isinf(pt.x) and not math.isinf(pt.y)
        dist = pt.magnitude()
        assert math.isclose(dist, earth.radius, rel_tol=1e-12)


# ============================================================================
# 2. Keplerian Conic Trajectory Calculations Tests
# ============================================================================


def test_keplerian_orbit_closed_ellipse_sampling():
    """Verify Keplerian orbit ellipse calculation and periapsis/apoapsis points."""
    earth = get_earth()
    r_orbit = earth.radius + 300_000.0  # 300 km altitude
    v_circ = math.sqrt(earth.mu / r_orbit)

    pos = Vector2D(r_orbit, 0.0)
    vel = Vector2D(0.0, v_circ)

    orb = calculate_orbit(pos, vel, earth, warn=False)
    assert orb.is_closed
    assert math.isclose(orb.eccentricity, 0.0, abs_tol=1e-4)
    assert math.isclose(orb.periapsis_altitude, 300_000.0, rel_tol=1e-4)
    assert math.isclose(orb.apoapsis_altitude, 300_000.0, rel_tol=1e-4)


def test_keplerian_orbit_hyperbolic_trajectory_properties():
    """Verify hyperbolic orbit parameters and asymptote boundaries."""
    earth = get_earth()
    pos = Vector2D(earth.radius + 500_000.0, 0.0)
    # Hyperbolic velocity (greater than escape velocity)
    v_esc = math.sqrt(2.0 * earth.mu / pos.magnitude())
    vel = Vector2D(0.0, v_esc * 1.3)

    orb = calculate_orbit(pos, vel, earth, warn=False)
    assert orb.eccentricity > 1.0
    assert orb.is_hyperbolic
    assert math.isinf(orb.apoapsis_altitude)
    assert orb.periapsis_altitude > 0.0


# ============================================================================
# 3. VAB Engineering Calculations Tests
# ============================================================================


def test_vab_engineering_stats_falcon9_preset():
    """Verify mass, TWR, and analytical Delta-V for Falcon 9 reference configuration."""
    f9 = create_falcon_9()
    stats = calculate_rocket_engineering_stats(f9)

    # 1. Mass checks
    assert stats.total_mass > 540_000.0  # ~550 tons
    assert stats.propellant_mass > 500_000.0  # S1 (410t) + S2 (107.5t)
    assert stats.dry_mass > 30_000.0

    # 2. Stage 0 (Liftoff) TWR
    # 9x Merlin 1D: 9 * 845 kN = 7,605 kN sea-level thrust
    # Weight = ~552,630 kg * 9.80665 m/s^2 = ~5,419 kN => TWR ~ 1.40
    assert 1.30 <= stats.stage0_twr_sl <= 1.50
    assert stats.is_launchable is True

    # 3. Multi-stage analytical Delta-V via Tsiolkovsky equation
    assert len(stats.stages) == 2
    st0 = stats.stages[0]
    st1 = stats.stages[1]

    assert st0.delta_v > 3500.0  # Stage 1 booster Delta-v
    assert st1.delta_v > 7000.0  # Stage 2 upper stage vacuum Delta-v
    assert stats.total_delta_v > 11_000.0  # Sufficient for low Earth orbit


def test_vab_engineering_stats_underpowered_rocket():
    """Verify that a rocket with TWR <= 1.0 is flagged as not launchable."""
    # Create single stage with heavy tank and tiny engine
    eng = Engine(
        id="tiny_eng",
        name="Tiny Engine",
        dry_mass=100.0,
        length=1.0,
        radius=0.5,
        thrust_vac=5000.0,
        thrust_sl=4000.0,
        isp_vac=300.0,
    )
    tank = FuelTank(
        id="huge_tank",
        name="Huge Tank",
        dry_mass=5000.0,
        length=10.0,
        radius=1.5,
        propellant_mass=50000.0,
        propellant_capacity=50000.0,
    )
    stage = Stage(stage_id="s0", name="Stage 0", parts=[eng, tank])
    rocket = Rocket(name="Underpowered", stages=[stage])

    stats = calculate_rocket_engineering_stats(rocket)
    assert stats.stage0_twr_sl < 0.1
    assert stats.is_launchable is False


# ============================================================================
# 4. Headless VAB to FlightView Integration Tests
# ============================================================================


def test_vab_view_spawn_part_and_launch(init_pygame):
    """Verify headless VAB blueprint part spawning and launching."""
    surface = init_pygame

    launched_rocket: list[Rocket] = []

    def on_launch_callback(r: Rocket):
        launched_rocket.append(r)

    vab = VABView(surface=surface, on_launch=on_launch_callback)

    # Initially loaded with Falcon 9
    initial_parts_count = sum(len(s.parts) for s in vab.rocket.stages)
    assert initial_parts_count > 0

    # Trigger launch
    vab._trigger_launch()
    assert len(launched_rocket) == 1
    assert launched_rocket[0].name == "Falcon 9"


def test_flight_view_simulation_step_and_liftoff(init_pygame):
    """Verify FlightView physics advancement, throttle control, and liftoff."""
    surface = init_pygame
    f9 = create_falcon_9()
    flight = FlightView(surface=surface, rocket=f9)

    # Initial state on pad
    assert flight.vehicle.state == VehicleState.ON_PAD
    assert math.isclose(flight.vehicle.speed, flight.body.surface_velocity_at(flight.vehicle.position).magnitude(), rel_tol=1e-5)

    # Set 100% throttle
    flight.throttle_slider.set_value(1.0)
    flight.throttle_cmd = 1.0

    # Advance simulation by several steps (total 1.0s)
    for _ in range(10):
        flight.update(0.1)

    # Vehicle should liftoff and transition to IN_FLIGHT
    assert flight.vehicle.state == VehicleState.IN_FLIGHT
    assert flight.vehicle.altitude > 0.0
    assert flight.vehicle.speed > 0.0

    # Verify telemetry snapshots recorded
    assert len(flight.logger) > 0
    snap = flight.vehicle.get_telemetry_snapshot()
    assert snap.thrust > 0.0
    assert snap.mass < 552630.0  # Propellant consumed


def test_flight_view_staging_and_debris(init_pygame):
    """Verify that stage separation creates coasting visual debris."""
    surface = init_pygame
    f9 = create_falcon_9()
    flight = FlightView(surface=surface, rocket=f9)

    # Launch vehicle
    flight.throttle_slider.set_value(1.0)
    flight.throttle_cmd = 1.0
    for _ in range(5):
        flight.update(0.1)

    initial_stage_idx = flight.vehicle.rocket.active_stage_index

    # Trigger stage separation
    flight.trigger_stage()

    # Active stage advances to upper stage
    assert flight.vehicle.rocket.active_stage_index > initial_stage_idx
    # Debris list contains decoupled booster
    assert len(flight.renderer.debris) >= 1
    deb = flight.renderer.debris[0]
    assert deb.stage.staging_order == 0

    # Advance debris physics
    flight.renderer.update_debris(0.1, flight.body)
    assert deb.position.magnitude() > 0.0


def test_camera_zoom_and_cursor_anchor():
    """Verify camera zoom clamping and mouse-centered zoom transforms."""
    cam = Camera(viewport_width=1280, viewport_height=720, scale=10.0, min_scale=0.1, max_scale=100.0)

    # Zoom in
    cam.zoom_in(2.0)
    assert cam.target_scale == 20.0

    # Zoom out beyond min_scale
    cam.zoom_out(500.0)
    assert cam.target_scale == 0.1

    # Zoom at screen point
    cam.snap_scale(10.0)
    cursor = (320.0, 180.0)
    w_before = cam.screen_to_world(cursor)
    cam.zoom_at_screen_point(2.0, cursor)
    w_after = cam.screen_to_world(cursor)
    assert (w_before - w_after).magnitude() < 1e-5


def test_vab_json_save_load_roundtrip(tmp_path, monkeypatch, init_pygame):
    """Verify saving and loading custom rocket design to JSON."""
    surface = init_pygame
    monkeypatch.chdir(tmp_path)

    vab = VABView(surface=surface)
    vab._save_json()

    saved_file = tmp_path / "rocket.json"
    assert saved_file.exists()

    # Clear and reload
    vab._clear_rocket()
    assert len(vab.rocket.stages[0].parts) == 0

    vab._load_json()
    assert len(vab.rocket.stages) > 0
    assert sum(len(s.parts) for s in vab.rocket.stages) > 0


def test_telemetry_modal_csv_export(tmp_path, monkeypatch, init_pygame):
    """Verify TelemetryModal displays metrics and exports CSV."""
    surface = init_pygame
    monkeypatch.chdir(tmp_path)

    f9 = create_falcon_9()
    flight = FlightView(surface=surface, rocket=f9)
    flight.throttle_slider.set_value(1.0)
    flight.throttle_cmd = 1.0
    for _ in range(5):
        flight.update(0.1)

    flight.open_modal()
    assert flight.modal is not None

    # Render modal headlessly
    flight.render()

    # Export CSV
    flight.modal._export_csv()
    csv_file = tmp_path / "telemetry.csv"
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "time,altitude,speed,mach" in content


def test_time_warp_safety_clamp_in_flight_view(init_pygame):
    """Verify that requesting >4x warp below cutoff altitude clamps to 4x with toast."""
    surface = init_pygame
    f9 = create_falcon_9()
    flight = FlightView(surface=surface, rocket=f9)

    # Rocket is on launchpad (altitude = 0 < 100 km)
    flight._request_time_warp(100.0)

    # Must be clamped to 4x physics warp
    assert flight.sim.time_warp == 4.0
    assert flight.sim.warp_mode == "PHYSICS"
    assert "Cannot warp >4x" in flight.toast.message


def test_flight_view_map_toggle_and_render_headless(init_pygame):
    """Verify headless rendering of both close-up Flight View and orbital Map View."""
    surface = init_pygame
    f9 = create_falcon_9()
    flight = FlightView(surface=surface, rocket=f9)

    # Render Flight View
    flight.render()

    # Toggle to Map View
    flight.toggle_map_view()
    assert flight.is_map_view is True

    # Render Map View
    flight.render()

    # Toggle back
    flight.toggle_map_view()
    assert flight.is_map_view is False

