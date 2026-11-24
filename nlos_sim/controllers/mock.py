"""A small LOS-only baseline controller."""

from __future__ import annotations

import math

from ..dynamics import Action
from ..entities import Robot
from ..observation import Observation
from .parent_controller import Controller

MOCK_SPEED = 2.4
STOP_DISTANCE = 0.6


class MockController(Controller):
    """Move forward unless the center LOS ray reports a nearby obstacle."""

    def get_action(self, robot: Robot, observation: Observation) -> Action:
        readings = observation.los
        if not readings:
            return Action()

        center_hit = readings[len(readings) // 2][1]
        if center_hit is not None and center_hit.distance <= STOP_DISTANCE:
            return Action()

        return Action(
            vx=MOCK_SPEED * math.cos(robot.theta),
            vy=MOCK_SPEED * math.sin(robot.theta),
        )