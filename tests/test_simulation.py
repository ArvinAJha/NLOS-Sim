"""Headless tests for the simulation loop and heading dynamics."""

import numpy as np
import pytest

from nlos_sim.controllers.mock import MockController
from nlos_sim.dynamics import Action, Dynamics
from nlos_sim.entities import Robot
from nlos_sim.simulation import Simulation
from nlos_sim.world import World


def test_simulation_shares_one_controller_with_all_robots():
    world = World()
    nlos_robot = Robot(0.3, np.array((0.0, 0.0, 0.0)), nlos=True)
    second_robot = Robot(0.3, np.array((3.0, 0.0, 0.0)))
    third_robot = Robot(0.3, np.array((6.0, 0.0, 0.0)))
    robots = (nlos_robot, second_robot, third_robot)
    for robot in robots:
        world.add(robot)
    controller = MockController()
    simulation = Simulation(world, controller)
    initial_poses = {robot: robot.pose.copy() for robot in robots}

    simulation.step()

    assert simulation.observations[nlos_robot].nlos is not None
    assert simulation.observations[second_robot].nlos is None
    assert simulation.observations[third_robot].nlos is None
    assert simulation.controller is controller
    assert set(simulation.actions) == set(robots)
    assert set(simulation.observation_poses) == set(robots)
    assert simulation.observation_frame == 0
    assert simulation.observation_id == "obs-000000"
    for robot in robots:
        assert not np.allclose(robot.pose, initial_poses[robot])


def test_simulation_records_each_robots_path_from_initial_pose():
    world = World()
    robot = Robot(0.3, np.array((0.0, 0.0, 0.0)))
    world.add(robot)
    simulation = Simulation(world, MockController())

    simulation.step()

    np.testing.assert_allclose(simulation.paths[robot][0], [0.0, 0.0])
    np.testing.assert_allclose(simulation.paths[robot][-1], robot.pose[:2])
    assert len(simulation.paths[robot]) == 2


def test_heading_follows_fast_velocity():
    robot = Robot(0.3, np.array((0.0, 0.0, 0.0)))

    Dynamics().step(robot, Action(0.0, 2.0), 0.1)

    assert robot.theta == pytest.approx(np.pi / 2.0)


def test_heading_stays_put_below_turn_speed():
    robot = Robot(0.3, np.array((0.0, 0.0, 0.5)))

    Dynamics().step(robot, Action(0.0, 0.01), 0.1)

    assert robot.theta == pytest.approx(0.5)
