"""Kinematic robot dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entities import Robot

DEFAULT_MAX_SPEED = 5.0


@dataclass(frozen=True)
class Action:
    """Desired world-frame velocity in meters per second."""

    vx: float = 0.0
    vy: float = 0.0


class Dynamics:
    """Applies desired velocity with a maximum-speed clamp."""

    def __init__(self, max_speed: float = DEFAULT_MAX_SPEED) -> None:
        if max_speed <= 0.0:
            raise ValueError("Maximum speed must be positive.")
        self.max_speed = max_speed

    def step(self, robot: Robot, action: Action, dt: float) -> None:
        """Set the robot velocity and integrate its position for one step."""
        if dt < 0.0:
            raise ValueError("Time step cannot be negative.")
        velocity = np.array([action.vx, action.vy], dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed:
            velocity *= self.max_speed / speed
        robot.velocity = velocity
        x, y, theta = robot.pose
        robot.set_pose(
            x + velocity[0] * dt,
            y + velocity[1] * dt,
            theta,
        )
