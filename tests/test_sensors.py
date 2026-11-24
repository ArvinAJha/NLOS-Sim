"""Headless tests for the robot flags and the LOS/NLOS sensors."""

import numpy as np
import pytest

from nlos_sim.entities import Robot, Wall
from nlos_sim.geometry.shapes import Box
from nlos_sim.observation import NlosSample
from nlos_sim.sensors.los import LOS_RAY_COUNT, cast_los_scan
from nlos_sim.sensors.nlos import BIN_EDGES, cast_nlos_scan, make_histogram
from nlos_sim.world import World


def wall(x, y, width, height):
    return Wall(Box(width, height), np.array((x, y, 0.0)))


def make_world(walls, robot_pose):
    world = World()
    for item in walls:
        world.add(item)
    world.add(Robot(0.3, robot_pose))
    return world


def test_robot_nlos_defaults_off_and_los_is_gone():
    robot = Robot(0.3, (0.0, 0.0, 0.0))

    assert robot.nlos is False
    with pytest.raises(TypeError):
        Robot(0.3, (0.0, 0.0, 0.0), los=True)


def test_los_scan_is_odd_and_center_ray_follows_heading():
    world = make_world([wall(3.25, 0.0, 0.5, 1.0)], (0.0, 0.0, 0.0))
    robot = world.robots[0]

    scan = cast_los_scan(world, robot)

    assert len(scan) == LOS_RAY_COUNT
    assert LOS_RAY_COUNT % 2 == 1
    direction, hit = scan[len(scan) // 2]
    np.testing.assert_allclose(direction, [1.0, 0.0])
    assert hit is not None
    assert hit.entity is world.walls[0]


def test_los_scan_never_hits_its_own_robot():
    world = make_world([wall(3.25, 0.0, 0.5, 1.0)], (0.0, 0.0, 0.0))

    for _, hit in cast_los_scan(world, world.robots[0]):
        assert hit is None or hit.entity is not world.robots[0]


def test_nlos_path_length_straight_up():
    # Robot at origin facing +x. Relay wall: left face at x=3.
    # Hidden wall: bottom face at y=2. Center ray hits (3, 0), then (3, 2).
    world = make_world(
        walls=[wall(3.25, 0.0, 0.5, 1.0), wall(3.0, 2.25, 4.0, 0.5)],
        robot_pose=(0.0, 0.0, 0.0),
    )
    scan = cast_nlos_scan(world, world.robots[0])

    match = [
        sample
        for sample in scan.samples
        if np.allclose(sample.relay_point, [3.0, 0.0])
        and np.allclose(sample.hit_point, [3.0, 2.0])
    ]
    assert len(match) == 1
    assert match[0].path_length == pytest.approx(3.0 + 2.0)


def test_empty_nlos_scan_has_all_zero_histogram():
    world = make_world(walls=[], robot_pose=(0.0, 0.0, 0.0))

    scan = cast_nlos_scan(world, world.robots[0])

    assert scan.samples == []
    assert len(scan.counts) == len(BIN_EDGES) - 1
    assert scan.counts.sum() == 0


def test_known_path_length_lands_in_expected_bin():
    sample = NlosSample(np.zeros(2), np.zeros(2), 2.75)

    counts = make_histogram([sample])

    assert counts[5] == 1
    assert counts.sum() == 1
