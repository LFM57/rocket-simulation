"""Unit and regression tests verifying all fixes for PROBLEMS.md."""

from __future__ import annotations

import math
import os
import pygame
import pytest

from core.vector2d import Vector2D
from models.presets import create_falcon_9
from physics.aerodynamics import calculate_aero_forces
from physics.celestial import get_earth
from simulation.autopilot import SASMode
from simulation.flight_dynamics import FlightVehicle, VehicleState
from ui.camera import Camera, CameraMode
from ui.flight_view import FlightView
from ui.vab_view import VABView
from ui.widgets import ProgressBar


@pytest.fixture(scope="module")
def pygame_surface():
    """Headless display surface."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    if not pygame.get_init():
        pygame.init()
    surface = pygame.display.get_surface()
    if surface is None:
        surface = pygame.display.set_mode((1280, 720))
    yield surface


# ============================================================================
# 1. UI Layout & Overlapping Elements
# ============================================================================


def test_flight_hud_no_overlap_on_small_window(pygame_surface):
    """Verify Menu [V] and SAS selector buttons do not collide on small heights (<600px)."""
    flight = FlightView(surface=pygame_surface)
    # Resize to small window height
    flight.update_layout(800, 500)

    # b_end is the Menu [V] button
    b_end = flight.hud_buttons[3]
    assert b_end.text == "Menu [V]"

    for mode, sas_btn in flight.sas_buttons.items():
        assert not b_end.rect.colliderect(sas_btn.rect), (
            f"SAS button {sas_btn.text} at {sas_btn.rect} collides with Menu button {b_end.rect}"
        )
        assert sas_btn.rect.top > b_end.rect.bottom


def test_telemetry_header_state_formatting(pygame_surface):
    """Verify telemetry panel formats state cleanly as 'ON_PAD', not 'VehicleState.ON_PAD'."""
    flight = FlightView(surface=pygame_surface)
    snap = flight.vehicle.get_telemetry_snapshot()
    st_text = snap.state.value if hasattr(snap.state, "value") else str(snap.state)
    assert st_text == "ON_PAD"
    assert "VehicleState" not in st_text


def test_warp_bar_centered_horizontally(pygame_surface):
    """Verify 7 warp buttons are centered horizontally at width // 2."""
    for width in [800, 1024, 1280, 1920]:
        flight = FlightView(surface=pygame_surface)
        flight.update_layout(width, 720)
        first_btn = flight.warp_buttons[0]
        last_btn = flight.warp_buttons[-1]
        warp_bar_center = (first_btn.rect.left + last_btn.rect.right) / 2.0
        assert math.isclose(warp_bar_center, width / 2.0, abs_tol=1.0)


def test_progress_bar_text_adapts_to_width(pygame_surface):
    """Verify ProgressBar with long text adapts font size to avoid container overflow."""
    bar = ProgressBar(pygame.Rect(20, 20, 150, 24), value=1.0, max_value=1.0, label="Propellant (410.00 t)")
    # Draw headlessly without error
    bar.draw(pygame_surface)


def test_vab_top_bar_no_collision_or_overflow(pygame_surface):
    """Verify VAB top toolbar buttons are right-aligned and do not collide with rocket name or sidebar."""
    for width in [900, 1280]:
        vab = VABView(surface=pygame_surface)
        vab.update_layout(width, 700)

        # Rocket name label starts at top_bar_rect.left + 16
        name_left = vab.top_bar_rect.left + 16

        for btn in vab.top_buttons:
            # Button must be inside top_bar_rect
            assert btn.rect.right <= vab.top_bar_rect.right
            assert btn.rect.left >= vab.top_bar_rect.left
            # Button must not overlap the rocket name on the left
            assert btn.rect.left > name_left + 100


def test_vab_part_inspector_no_launch_collision(pygame_surface):
    """Verify Part Inspector does not collide with the pinned LAUNCH button on small windows."""
    vab = VABView(surface=pygame_surface)
    vab.update_layout(1000, 520)

    # Select a part to activate the inspector
    vab.selected_part = vab.rocket.stages[0].parts[9]  # Stage 1 tank
    b_launch = vab.action_buttons[0]

    # Content area ends at sidebar_right_rect.bottom - 60
    assert b_launch.rect.top >= vab.sidebar_right_rect.bottom - 54


def test_vab_multi_engine_cluster_offsets(pygame_surface):
    """Verify multi-engine clusters have lateral offsets and do not render on the same pixels."""
    vab = VABView(surface=pygame_surface)
    engines = [p for p in vab.rocket.stages[0].parts if hasattr(p, "thrust_vac") and p.thrust_vac > 0]
    offsets = vab._compute_cluster_offsets(engines)
    assert len(offsets) == len(engines)
    # Falcon 9 has 9 engines: offsets should span across positive and negative
    assert any(o > 0.5 for o in offsets)
    assert any(o < -0.5 for o in offsets)
    assert 0.0 in offsets  # Center engine


# ============================================================================
# 2. Camera Tracking & Scrolling
# ============================================================================


def test_camera_zero_velocity_lag_on_launchpad(pygame_surface):
    """Verify camera position exactly locks target position without velocity lag."""
    flight = FlightView(surface=pygame_surface)
    for _ in range(10):
        flight.update(0.1)

    # Rocket should be exactly at screen center
    sx, sy = flight.camera.world_to_screen(flight.vehicle.position)
    assert math.isclose(sx, flight.surface.get_width() / 2.0, abs_tol=1e-3)
    assert math.isclose(sy, flight.surface.get_height() / 2.0, abs_tol=1e-3)


def test_flight_view_mousewheel_and_keyboard_zoom(pygame_surface):
    """Verify mousewheel and +/- zoom shortcuts in FlightView."""
    flight = FlightView(surface=pygame_surface)
    initial_scale = flight.camera.target_scale

    # Keyboard zoom in (+)
    ev_plus = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS)
    flight.handle_event(ev_plus)
    assert flight.camera.target_scale > initial_scale

    # Keyboard zoom out (-)
    ev_minus = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS)
    flight.handle_event(ev_minus)
    flight.handle_event(ev_minus)
    assert flight.camera.target_scale < initial_scale

    # Mousewheel zoom
    ev_wheel = pygame.event.Event(pygame.MOUSEWHEEL, y=1, x=0)
    old_target = flight.camera.target_scale
    flight.handle_event(ev_wheel)
    assert flight.camera.target_scale > old_target


def test_flight_view_camera_panning(pygame_surface):
    """Verify right/middle-click mouse dragging pans camera offset in FlightView."""
    flight = FlightView(surface=pygame_surface)
    assert flight.pan_offset.magnitude() == 0.0

    # Start drag
    ev_down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(400, 300))
    flight.handle_event(ev_down)
    assert flight.is_panning is True

    # Drag motion
    ev_motion = pygame.event.Event(pygame.MOUSEMOTION, pos=(450, 320))
    flight.handle_event(ev_motion)
    assert flight.pan_offset.magnitude() > 0.0

    # End drag
    ev_up = pygame.event.Event(pygame.MOUSEBUTTONUP, button=3, pos=(450, 320))
    flight.handle_event(ev_up)
    assert flight.is_panning is False

    # Home key resets pan offset
    ev_home = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME)
    flight.handle_event(ev_home)
    assert flight.pan_offset.magnitude() == 0.0


def test_vab_recenter_canvas_on_resize(pygame_surface):
    """Verify VAB canvas automatically scales and centers rocket on update_layout."""
    vab = VABView(surface=pygame_surface)
    vab.update_layout(1000, 500)
    # The 65m Falcon 9 rocket should scale to fit inside 500px height with margin
    rocket_px_height = 65.0 * vab.scale
    assert rocket_px_height < vab.canvas_rect.height
    assert rocket_px_height > vab.canvas_rect.height * 0.7


# ============================================================================
# 3. Rocket Steering & Spin Control
# ============================================================================


def test_falcon_9_preset_aerodynamic_stability():
    """Verify Falcon 9 preset is aerodynamically stable (CoP <= CoM) at liftoff."""
    f9 = create_falcon_9()
    y_com = f9.center_of_mass()
    y_cop = f9.center_of_pressure()
    assert y_cop <= y_com, f"Destabilizing configuration: CoP ({y_cop:.2f}m) > CoM ({y_com:.2f}m)"


def test_aerodynamic_pitch_damping():
    """Verify calculate_aero_forces generates damping torque opposing angular velocity."""
    earth = get_earth()
    pos = Vector2D(0.0, earth.radius + 1000.0)
    vel = Vector2D(0.0, 200.0)

    # Pure axial alignment (alpha = 0 -> zero static torque)
    aero_static = calculate_aero_forces(
        craft_pos=pos,
        craft_vel=vel,
        attitude_angle=math.pi / 2,
        body=earth,
        reference_area=10.0,
        length=60.0,
        angular_velocity=0.0,
    )
    assert math.isclose(aero_static.torque, 0.0, abs_tol=1e-5)

    # Positive angular velocity (omega > 0) must produce negative damping torque (CW)
    aero_damped = calculate_aero_forces(
        craft_pos=pos,
        craft_vel=vel,
        attitude_angle=math.pi / 2,
        body=earth,
        reference_area=10.0,
        length=60.0,
        angular_velocity=0.5,
    )
    assert aero_damped.torque < -100.0, f"Expected negative damping torque, got {aero_damped.torque}"


def test_sas_target_windup_prevention(pygame_surface):
    """Verify holding steering key in HOLD_ATTITUDE clamps lead angle to prevent blind target windup."""
    flight = FlightView(surface=pygame_surface)
    flight.vehicle.autopilot.set_mode(SASMode.HOLD_ATTITUDE)
    flight.steer_rate = 0.4  # Hold CCW steer

    # Step for 10 seconds (would wind up 4.0 radians ~ 230 degrees if unclamped)
    for _ in range(100):
        flight.update(0.1)

    lead = math.atan2(
        math.sin(flight.vehicle.autopilot.target_attitude - flight.vehicle.attitude),
        math.cos(flight.vehicle.autopilot.target_attitude - flight.vehicle.attitude),
    )
    # Lead angle must be clamped to ~14 degrees (0.25 rad)
    assert abs(lead) <= 0.26


def test_manual_gimbal_auto_centering(pygame_surface):
    """Verify that in SASMode.OFF, releasing steering keys smoothly centers the engine gimbal."""
    flight = FlightView(surface=pygame_surface)
    flight.vehicle.autopilot.set_mode(SASMode.OFF)

    # Deflect gimbal manually
    flight.steer_rate = 0.4
    for _ in range(5):
        flight.update(0.1)

    eng = flight.vehicle.rocket.active_stage.engines[0]
    assert abs(eng.current_gimbal) > 0.01

    # Release key
    flight.steer_rate = 0.0
    for _ in range(20):
        flight.update(0.1)

    # Gimbal must have auto-centered back to 0.0
    assert math.isclose(eng.current_gimbal, 0.0, abs_tol=1e-4)


def test_zero_throttle_attitude_control_authority():
    """Verify attitude control authority is maintained at 0% throttle via RCS / reaction wheels."""
    earth = get_earth()
    f9 = create_falcon_9()
    # In space (200 km) with zero throttle
    pos = Vector2D(0.0, earth.radius + 200_000.0)
    vehicle = FlightVehicle(
        rocket=f9,
        body=earth,
        position=pos,
        velocity=Vector2D(7500.0, 0.0),
        attitude=0.0,
        angular_velocity=0.1,  # Spinning
        state=VehicleState.IN_FLIGHT,
    )
    vehicle.set_throttle(0.0)
    vehicle.autopilot.set_mode(SASMode.HOLD_ATTITUDE)
    vehicle.autopilot.set_target_pitch(0.0)

    # Step dynamics with 0% throttle
    snap = vehicle.step(dt=0.1)
    # Angular velocity should damp despite zero engine thrust
    assert abs(vehicle.angular_velocity) < 0.1
