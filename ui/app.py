"""Application state machine and window manager for Spaceflight Simulator (SFS).

Coordinates transitions between the Vehicle Assembly Building (VAB / Blueprint Editor)
and the active Flight View / Orbital Map View.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Optional

import pygame

from models.presets import create_falcon_9
from models.rocket import Rocket
from physics.celestial import CelestialBody, get_earth
from ui.flight_view import FlightView
from ui.vab_view import VABView


class AppScreen(str, Enum):
    """Active user interface screen state."""

    VAB = "VAB"
    FLIGHT = "FLIGHT"


class App:
    """Main application state machine coordinating views, event loops, and rendering.

    Parameters
    ----------
    width : int, default=1280
        Initial window width in pixels.
    height : int, default=720
        Initial window height in pixels.
    fullscreen : bool, default=False
        Whether to launch in fullscreen display mode.
    initial_screen : AppScreen, default=AppScreen.VAB
        Initial screen state to open.
    rocket : Optional[Rocket], default=None
        Initial rocket vehicle. If None, Falcon 9 preset is used.
    body : Optional[CelestialBody], default=None
        Primary celestial body. If None, Earth is used.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fullscreen: bool = False,
        initial_screen: AppScreen = AppScreen.VAB,
        rocket: Optional[Rocket] = None,
        body: Optional[CelestialBody] = None,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Spaceflight Simulator (SFS) - 2D Aerospace Physics")

        self.width: int = max(640, int(width))
        self.height: int = max(480, int(height))
        self.fullscreen: bool = fullscreen

        flags = pygame.RESIZABLE
        if self.fullscreen:
            flags |= pygame.FULLSCREEN

        self.surface: pygame.Surface = pygame.display.set_mode((self.width, self.height), flags)
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.is_running: bool = False

        self.body: CelestialBody = body if body is not None else get_earth()
        self.rocket: Rocket = rocket if rocket is not None else create_falcon_9()

        # Screens
        self.vab_view: VABView = VABView(
            surface=self.surface,
            on_launch=self.launch_rocket,
        )
        self.flight_view: Optional[FlightView] = None

        self.active_screen: AppScreen = (
            initial_screen if isinstance(initial_screen, AppScreen) else AppScreen(initial_screen)
        )

        if self.active_screen == AppScreen.FLIGHT:
            self.launch_rocket(self.rocket)

    def launch_rocket(self, rocket: Rocket) -> None:
        """Transition from VAB Blueprint Editor into active Flight View."""
        self.rocket = rocket
        self.flight_view = FlightView(
            surface=self.surface,
            rocket=self.rocket,
            body=self.body,
            on_return_vab=self.return_to_vab,
        )
        self.active_screen = AppScreen.FLIGHT

    def return_to_vab(self) -> None:
        """Transition from Flight View back to VAB Blueprint Editor."""
        self.flight_view = None
        self.active_screen = AppScreen.VAB
        self.vab_view.update_layout(self.width, self.height)

    def run(self) -> None:
        """Execute the primary 60 FPS application game loop."""
        self.is_running = True

        while self.is_running:
            dt = self.clock.tick(60) / 1000.0  # seconds

            # 1. Event Polling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                    break

                elif event.type == pygame.VIDEORESIZE:
                    self.width = max(640, event.w)
                    self.height = max(480, event.h)
                    self.surface = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self.vab_view.surface = self.surface
                    self.vab_view.update_layout(self.width, self.height)
                    if self.flight_view is not None:
                        self.flight_view.surface = self.surface
                        self.flight_view.update_layout(self.width, self.height)

                else:
                    # Delegate event to active screen
                    if self.active_screen == AppScreen.VAB:
                        self.vab_view.handle_event(event)
                    elif self.active_screen == AppScreen.FLIGHT and self.flight_view is not None:
                        self.flight_view.handle_event(event)

            # 2. Physics & Logic Update
            if self.active_screen == AppScreen.FLIGHT and self.flight_view is not None:
                self.flight_view.update(dt)

            # 3. View Rendering
            if self.active_screen == AppScreen.VAB:
                self.vab_view.render()
            elif self.active_screen == AppScreen.FLIGHT and self.flight_view is not None:
                self.flight_view.render()

            pygame.display.flip()

        pygame.quit()


__all__ = [
    "App",
    "AppScreen",
]
