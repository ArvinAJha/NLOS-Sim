"""Headless tests for swappable controllers."""

import numpy as np

from nlos_sim.controllers.gradient import GradientController
from nlos_sim.controllers.swarm import SwarmController
from nlos_sim.dynamics import Action
from nlos_sim.entities import Robot, Wall
from nlos_sim.geometry.shapes import Box
from nlos_sim.geometry.types import RayHit
from nlos_sim.observation import NlosSample, NlosScan, Observation


def wall_hit(point=(1.0, 0.0), distance=1.0):
    wall = Wall(Box(1.0, 1.0), np.array((2.0, 0.0, 0.0)))
    return RayHit(distance, np.array(point), np.array([-1.0, 0.0]), wall)


def test_one_wall_hit_pushes_away():
    robot = Robot(0.3, np.array((0.0, 0.0, 0.0)))
    observation = Observation(
        [(np.array([1.0, 0.0]), wall_hit((0.5, 0.0), distance=0.5))]
    )

    action = SwarmController().get_action(robot, observation)

    assert action.vx < 0.0


def test_empty_observation_gives_no_action():
    robot = Robot(0.3, np.array((0.0, 0.0, 0.0)))

    action = SwarmController().get_action(robot, Observation([]))

    assert action == Action()


def test_nlos_sample_on_one_side_changes_action():
    robot = Robot(0.3, np.array((0.0, 0.0, 0.0)))
    los: list[tuple[np.ndarray, RayHit | None]] = [
        (np.array([1.0, 0.0]), wall_hit())
    ]
    nlos = NlosScan(
        [NlosSample(np.array([1.0, 0.0]), np.array([0.0, 0.5]), 2.0)],
        np.zeros(1),
    )

    plain = SwarmController().get_action(robot, Observation(los))
    with_nlos = SwarmController().get_action(robot, Observation(los, nlos))

    assert not np.allclose(
        [plain.vx, plain.vy], [with_nlos.vx, with_nlos.vy]
    )


def test_swarm_controller_uses_per_robot_goals():
    robot_a = Robot(0.3, np.array((0.0, 0.0, 0.0)))
    robot_b = Robot(0.3, np.array((0.0, 0.0, 0.0)))
    controller = SwarmController(goals={
        robot_a: np.array([0.0, 5.0]),
        robot_b: np.array([5.0, 0.0]),
    })

    action_a = controller.get_action(robot_a, Observation([]))
    action_b = controller.get_action(robot_b, Observation([]))

    assert action_a.vy > 0.0
    assert action_b.vx > 0.0


def test_nlos_relay_can_add_guidance_without_los():
    robot = Robot(0.3, np.array((0.0, 0.0, 0.0)))
    nlos = NlosScan(
        [NlosSample(np.array([0.5, 1.0]), np.array([0.5, 1.0]), 3.0)],
        np.ones(1),
        primary_readings=[(np.array([0.0, 1.0]), None)],
    )
    controller = SwarmController(goals={robot: np.array([0.0, 5.0])})

    plain_action = controller.get_action(robot, Observation([]))
    nlos_action = controller.get_action(robot, Observation([], nlos))

    assert nlos_action.vy > 0.0
    assert not np.allclose(
        [plain_action.vx, plain_action.vy],
        [nlos_action.vx, nlos_action.vy],
    )


def test_swarm_repulsion_starts_inside_surface_clearance_buffer():
    robot = Robot(0.3, np.array((0.0, 0.0, 0.0)))
    far_hit = Observation(
        [(np.array([1.0, 0.0]), wall_hit((1.0, 0.0), distance=1.0))]
    )
    near_hit = Observation(
        [(np.array([1.0, 0.0]), wall_hit((0.8, 0.0), distance=0.8))]
    )
    controller = SwarmController()

    assert controller.get_action(robot, far_hit) == Action()
    assert controller.get_action(robot, near_hit).vx < 0.0


def test_gradient_controller_uses_clearance_buffer_and_speed_limit():
    robot = Robot(0.3, np.array((0.0, 0.0, 0.0)))
    observation = Observation(
        [(np.array([1.0, 0.0]), wall_hit((0.5, 0.0), distance=0.5))]
    )

    action = GradientController().get_action(robot, observation)

    assert action.vx < 0.0
    assert np.hypot(action.vx, action.vy) <= GradientController.MAX_SPEED
