"""Repel from nearby LOS hits using their surface normals."""

from __future__ import annotations

import math

from ..dynamics import Action
from ..entities import Robot, Wall
from ..observation import Observation
from .parent_controller import Controller


class GradientController(Controller):
    """Use inverse-square repulsion inside a configurable surface buffer."""

    REPULSIVE_GAIN = 10.0
    ROBOT_WEIGHT = 1.0
    WALL_WEIGHT = 0.3
    MIN_DISTANCE = 0.01
    BUFFER_DISTANCE = 0.6
    MAX_SPEED = 1.0

    def get_action(self, robot: Robot, observation: Observation) -> Action:
        total_vx = 0.0
        total_vy = 0.0
        active_hits = 0

        for _, hit in observation.los:
            if hit is None:
                continue

            clearance = hit.distance - robot.radius
            if clearance >= self.BUFFER_DISTANCE:
                continue

            weight = self.WALL_WEIGHT if isinstance(hit.entity, Wall) else self.ROBOT_WEIGHT
            distance = max(clearance, self.MIN_DISTANCE)
            magnitude = self.REPULSIVE_GAIN * weight / (distance**2)
            total_vx += float(hit.normal[0]) * magnitude
            total_vy += float(hit.normal[1]) * magnitude
            active_hits += 1

        if active_hits == 0:
            forward_speed = self.MAX_SPEED / 10.0
            return Action(
                vx=forward_speed * math.cos(robot.theta),
                vy=forward_speed * math.sin(robot.theta),
            )

        speed = math.hypot(total_vx, total_vy)
        if speed > self.MAX_SPEED:
            total_vx *= self.MAX_SPEED / speed
            total_vy *= self.MAX_SPEED / speed
        return Action(total_vx, total_vy)