"""A minimal scan-driven controller for animation examples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from abc import ABC
from dataclasses import dataclass
from typing import Any

from ..dynamics import Action
from ..entities import Robot, Wall
from ..geometry.types import RayHit, Vector

MOCK_SPEED = 2.4
STOP_DISTANCE = 0.6

class BaseController(ABC):
    """Parent class for all controllers"""

    def get_action(
        self, 
        robot: Robot, 
        readings: Sequence[tuple[Vector, RayHit | None]],
    ) -> Action:
        raise NotImplementedError("Controllers must implement get_action method.")

class GradientController(BaseController):
    # Adjust these constants to tune the behavior
    REPULSIVE_GAIN = 10.0
    ROBOT_WEIGHT = 1.0
    WALL_WEIGHT = 0.3
    MIN_DISTANCE = 0.01 # Prevent division by zero
    MAX_SPEED = 1.0

    def get_action(
        self, robot: Robot, 
        readings: Sequence[tuple[Vector, RayHit | None]]
    ) -> Action:
        
        # Filter out rays that didn't hit anything
        ray_hits = list(filter(None, (item[1] for item in readings)))
        
        if not ray_hits:
            # No obstacles detected, return zero action or a default forward movement
            return Action(
                vx=self.MAX_SPEED/10 * math.cos(robot.theta),
                vy=self.MAX_SPEED/10 * math.sin(robot.theta)
            )

        total_vx = 0.0
        total_vy = 0.0

        for hit in ray_hits:
            # 1. Determine the weight based on the entity type
            weight = self.ROBOT_WEIGHT
            if hit.entity and isinstance(hit.entity, Wall):
                weight = self.WALL_WEIGHT
            
            # 2. Calculate the magnitude of the repulsion
            distance = max(hit.distance, self.MIN_DISTANCE) - robot.radius
            magnitude = (self.REPULSIVE_GAIN * weight) / (distance ** 2) # distance^2
            
            # 3. Apply the magnitude along the surface normal
            total_vx += hit.normal[0] * magnitude
            total_vy += hit.normal[1] * magnitude


        speed = math.hypot(total_vx, total_vy)
        if speed > self.MAX_SPEED:
            total_vx = (total_vx / speed) * self.MAX_SPEED
            total_vy = (total_vy / speed) * self.MAX_SPEED

        return Action(
            vx=total_vx, 
            vy=total_vy
        )

class MockController(BaseController):
    """Move forward unless the center ray reports a nearby obstacle."""

    def get_action(
        self,
        robot: Robot,
        readings: Sequence[tuple[Vector, RayHit | None]],
    ) -> Action:
        if not readings:
            return Action()

        center_hit = readings[len(readings) // 2][1]
        if center_hit is not None and center_hit.distance <= STOP_DISTANCE:
            return Action()

        return Action(
            vx=MOCK_SPEED * math.cos(robot.theta),
            vy=MOCK_SPEED * math.sin(robot.theta),
        )