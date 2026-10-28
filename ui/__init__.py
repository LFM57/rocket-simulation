"""Spaceflight Simulator (SFS) style 2D graphical user interface.

Modules:
    - camera: 64-bit anti-jitter camera math, horizon-up and inertial frames.
    - widgets: Procedural UI buttons, throttle slider, fuel gauges, panels, plots.
    - renderer: Procedural rocket parts, dynamic plumes, sky, terrain, and orbits.
    - vab_view: Blueprint editor and vehicle assembly building.
    - flight_view: Real-time flight simulation, map view, and telemetry summary.
    - app: Main application state machine.
"""

from ui.app import App, AppScreen
from ui.camera import Camera, CameraMode
from ui.flight_view import FlightView, TelemetryModal
from ui.renderer import DebrisStage, Renderer
from ui.vab_view import VABView, calculate_rocket_engineering_stats
from ui.widgets import Button, Panel, PlotCanvas, ProgressBar, ToastManager, VerticalSlider

__all__ = [
    "App",
    "AppScreen",
    "Button",
    "Camera",
    "CameraMode",
    "DebrisStage",
    "FlightView",
    "Panel",
    "PlotCanvas",
    "ProgressBar",
    "Renderer",
    "TelemetryModal",
    "ToastManager",
    "VABView",
    "VerticalSlider",
    "calculate_rocket_engineering_stats",
]
