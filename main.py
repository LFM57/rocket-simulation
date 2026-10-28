#!/usr/bin/env python3
"""Spaceflight Simulator (SFS) - Interactive 2D Aerospace Simulation GUI.

Usage:
    python main.py                  # Open VAB Blueprint Editor (Default)
    python main.py --launch         # Launch directly into Flight View on Launchpad
    python main.py --fullscreen     # Launch in fullscreen mode
    python main.py --width 1600 --height 900
"""

from __future__ import annotations

import argparse
import sys

from models.presets import create_falcon_9
from ui.app import App, AppScreen


def parse_args() -> argparse.Namespace:
    """Parse command line configuration flags."""
    parser = argparse.ArgumentParser(
        description="Spaceflight Simulator (SFS) 2D Graphical User Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Initial window display width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Initial window display height in pixels",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Launch the application in fullscreen mode",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Skip VAB editor and launch directly into Flight View on the launchpad",
    )
    return parser.parse_args()


def main() -> None:
    """Application entrypoint."""
    args = parse_args()

    initial_screen = AppScreen.FLIGHT if args.launch else AppScreen.VAB
    rocket = create_falcon_9()

    app = App(
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        initial_screen=initial_screen,
        rocket=rocket,
    )
    app.run()


if __name__ == "__main__":
    main()
