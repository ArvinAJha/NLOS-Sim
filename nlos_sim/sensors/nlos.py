"""Optional two-bounce (NLOS) range sensor."""

import numpy as np

from ..entities import Robot
from ..observation import NlosSample, NlosScan
from ..world import World
from .rays_utils import fan_angles, position, unit

PRIMARY_FOV = np.radians(8)
PRIMARY_RAY_COUNT = 5
SECONDARY_FAN = np.radians(180)
SECONDARY_RAY_COUNT = 15
NLOS_MAX_PATH = 8.0  # meters: primary + secondary distance combined
BIN_WIDTH = 0.5
BIN_EDGES = np.arange(0, NLOS_MAX_PATH + BIN_WIDTH, BIN_WIDTH)


def cast_nlos_scan(world: World, robot: Robot) -> NlosScan:
    """Narrow fan -> relay points -> a half-circle fan from each relay point."""

    origin = position(robot)
    samples = []
    primary_readings = []
    for angle in fan_angles(robot.theta, PRIMARY_FOV, PRIMARY_RAY_COUNT):
        direction = unit(angle)
        relay = world.raycast(origin, direction, NLOS_MAX_PATH, ignore=robot)
        primary_readings.append((direction, relay))

        if relay is not None:
            samples += secondary_fan(world, robot, relay)
    return NlosScan(samples, make_histogram(samples), primary_readings)


def secondary_fan(world: World, robot: Robot, relay) -> list[NlosSample]:
    """Scatter a half-circle from the relay along its surface normal.

    The relay entity is ignored so a ray cannot re-hit the surface it starts
    on. This works because all shapes are convex. Misses are dropped; each ray
    keeps only its nearest hit.
    """
    samples = []
    normal_angle = np.arctan2(relay.normal[1], relay.normal[0])
    budget = NLOS_MAX_PATH - relay.distance
    for angle in fan_angles(normal_angle, SECONDARY_FAN, SECONDARY_RAY_COUNT):
        hit = world.raycast(
            relay.point, unit(angle), budget, ignore=[robot, relay.entity]
        )
        if hit is not None:
            samples.append(
                NlosSample(
                    relay.point,
                    hit.point,
                    relay.distance + hit.distance,
                )
            )
    return samples


def make_histogram(samples: list[NlosSample]) -> np.ndarray:
    """Fixed bins, so histograms from different frames compare directly."""
    lengths = [sample.path_length for sample in samples]
    counts, _ = np.histogram(lengths, bins=BIN_EDGES)  # empty list -> zeros
    return counts
    # EXTENSION POINT: weight each sample by cos(angle) and 1/path_length**2
