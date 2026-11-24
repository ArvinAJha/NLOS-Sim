"""Line-of-sight range sensor."""

import numpy as np

from ..entities import Robot
from ..geometry.types import RayHit
from ..world import World
from .rays_utils import fan_angles, position, unit

LOS_FOV = np.radians(45)
LOS_RAY_COUNT = 15  # odd, so there is a true center ray
LOS_RANGE = 6.0


def cast_los_scan(
    world: World, robot: Robot
) -> list[tuple[np.ndarray, RayHit | None]]:
    """One (direction, hit) pair per ray. hit is None when nothing is hit."""
    origin = position(robot)
    readings = []
    for angle in fan_angles(robot.theta, LOS_FOV, LOS_RAY_COUNT):
        direction = unit(angle)
        hit = world.raycast(origin, direction, LOS_RANGE, ignore=robot)
        readings.append((direction, hit))
    return readings
