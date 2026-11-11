"""Headless tests for entities, environments, and the world."""

import numpy as np
import pytest

from nlos_sim.dynamics import Action, Dynamics
from nlos_sim.entities import Robot, Wall
from nlos_sim.environments import JUNCTION_CENTER, make_t_corridor
from nlos_sim.geometry.shapes import Box
from nlos_sim.world import World


def test_t_corridor_has_one_robot_and_eight_walls() -> None:
    world = make_t_corridor()

    assert len(world.robots) == 1
    assert len(world.walls) == 8
    assert world.robots[0].theta == pytest.approx(np.pi / 2.0)


def test_t_corridor_is_sealed_from_the_junction() -> None:
    world = make_t_corridor()
    origin = np.array(JUNCTION_CENTER)

    for angle in np.linspace(0.0, 2.0 * np.pi, 73)[:-1]:
        direction = np.array([np.cos(angle), np.sin(angle)])
        hit = world.raycast(origin, direction, 100.0, ignore=world.robots)
        assert hit is not None
        assert isinstance(hit.entity, Wall)


def test_ignore_prevents_robot_self_hit() -> None:
    world = make_t_corridor()
    robot = world.robots[0]
    origin = robot.pose[:2].copy()
    direction = np.array([1.0, 0.0])

    self_hit = world.raycast(origin, direction, 10.0)
    wall_hit = world.raycast(origin, direction, 10.0, ignore=robot)

    assert self_hit is not None
    assert self_hit.entity is robot
    assert wall_hit is not None
    assert isinstance(wall_hit.entity, Wall)


def test_robot_can_be_placed_at_arbitrary_poses() -> None:
    world = make_t_corridor()
    robot = world.robots[0]

    robot.set_pose(4.0, 1.0, 0.0)
    assert robot.pose == pytest.approx([4.0, 1.0, 0.0])
    assert world.overlaps(robot) == []

    robot.pose = (0.0, 1.0, np.pi)
    hit = world.raycast(
        robot.pose[:2], np.array([-1.0, 0.0]), 20.0, ignore=robot
    )

    assert hit is not None
    assert isinstance(hit.entity, Wall)


def test_robot_cannot_cross_wall_at_max_speed() -> None:
    world = make_t_corridor()
    robot = world.robots[0]
    bottom_wall = min(world.walls, key=lambda wall: wall.pose[1])
    floor_y = bottom_wall.pose[1] + bottom_wall.shape.height / 2.0

    for _ in range(1000):
        world.step(0.01, {robot: Action(vx=0.0, vy=-100.0)})

    resting_y = floor_y + robot.radius
    assert robot.y >= resting_y - 1e-6
    assert robot.y <= resting_y + 0.06


def test_robot_slides_along_wall_with_diagonal_action() -> None:
    world = make_t_corridor()
    robot = world.robots[0]
    robot.set_pose(-0.5, -6.0, 0.0)
    start_y = robot.y

    for _ in range(100):
        world.step(0.01, {robot: Action(vx=-5.0, vy=5.0)})

    assert robot.x == pytest.approx(-1.0 + robot.radius, abs=1e-6)
    assert robot.y > start_y + 1.0


def test_missing_action_holds_robot_still() -> None:
    world = make_t_corridor()
    robot = world.robots[0]
    start_pose = robot.pose.copy()

    world.step(0.1)

    np.testing.assert_allclose(robot.pose, start_pose)
    np.testing.assert_allclose(robot.velocity, [0.0, 0.0])


def test_world_overlap_query_and_collision_resolution() -> None:
    world = World()
    wall = Wall(Box(2.0, 2.0), (0.0, 0.0, 0.0))
    robot = Robot(0.5, (-1.2, 0.0, 0.0))
    world.add(wall)
    world.add(robot)

    contacts = world.overlaps(robot)

    assert len(contacts) == 1
    other, contact = contacts[0]
    assert other is wall
    assert contact.penetration == pytest.approx(0.3)

    world.step(0.0)

    assert robot.x == pytest.approx(-1.5)


def test_dynamics_clamps_speed() -> None:
    dynamics = Dynamics(max_speed=2.0)
    robot = Robot(0.3, (0.0, 0.0, 0.0))

    dynamics.step(robot, Action(vx=10.0, vy=0.0), 1.0)

    assert robot.velocity[0] == pytest.approx(2.0)
    assert robot.x == pytest.approx(2.0)
