"""Small helpers shared by the LOS and NLOS sensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..entities import Robot


def unit(angle: float) -> np.ndarray:
    """Return the unit direction at an angle."""
    return np.array([np.cos(angle), np.sin(angle)])


def position(robot: Robot) -> np.ndarray:
    """Return the robot's (x, y) position."""
    return np.array([robot.x, robot.y])


def fan_angles(center: float, width: float, count: int) -> np.ndarray:
    """Evenly spaced angles across `width` centered on `center`.

    Use an ODD count so one ray points exactly along `center`.
    """
    if count == 1:
        return np.array([center])
    return center + np.linspace(-width / 2, width / 2, count)
