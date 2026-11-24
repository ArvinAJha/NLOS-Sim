"""Kinematic robot dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import math

from .entities import Robot

DEFAULT_MAX_SPEED = 5.0
DEFAULT_MAX_ANGULAR_SPEED = 2  # m/s. Below this the direction is noise, so keep heading.


@dataclass(frozen=True)
class Action:
    """Desired world-frame velocity in meters per second."""

    vx: float = 0.0
    vy: float = 0.0

@dataclass(frozen=True)
class UniAction:
    v: float = 0
    w: float = 0


class Dynamics:
    """Applies desired velocity with a maximum-speed clamp."""

    def __init__(self, max_speed: float = DEFAULT_MAX_SPEED, max_angular_speed: float = DEFAULT_MAX_ANGULAR_SPEED) -> None:
        if max_speed <= 0.0 or max_angular_speed <= 0.0:
            raise ValueError("Max linear and angular speeds must be positive.")
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed

    # TODO: input: new action (v, w). Applies action, clips to speed limits, and moves robot
    def step(self, robot: Robot, action: UniAction, dt: float) -> None:
        """Set the robot velocity and integrate its position for one step."""
        
        if dt < 0.0:
            raise ValueError("Time step cannot be negative.")

        v = max(0.0, min(self.max_speed, action.v))     # action's linear velocity clamped [-max, max]
        w = max(-self.max_angular_speed, min(self.max_angular_speed, action.w))     # action's angular velocity clamped [-max, max]

        x, y, theta = robot.pose

        mid_theta = theta + (0.5*w*dt)  

        new_x = x + v*dt*math.cos(mid_theta)
        new_y = y + v*dt*math.sin(mid_theta)
        new_theta = theta + w*dt        # this value is not wrapped to [-pi, pi]. Only when drawn. 

        robot.set_pose(new_x, new_y, new_theta)

    # TODO: new helper function move robot that takes velocity and w and sets pose to ?? (forward in the direction of the theta by the value of v)

    @staticmethod
    def convert_si_to_uni(vx, vy, theta, lookahead=0.8) -> UniAction:
        """
        Converting single integrator into unicycle dynamics with near-identity diffeomorphism. 
        Done by defining a look-ahead point, inverse Jacobian matrix, and vx, vy, theta.
        [v w]' = J^-1 * [vx vy]' 
        """

        cos_part = math.cos(theta)
        sin_part = math.sin(theta)

        invLook = 1.0 / lookahead

        v = cos_part*vx + sin_part*vy
        w = (-sin_part*vx + cos_part*vy) * invLook

        return UniAction(
            v=v,
            w=w
        )

    @staticmethod
    def barrier_certificate(unsafe_action: UniAction) -> UniAction:
        return UniAction()
