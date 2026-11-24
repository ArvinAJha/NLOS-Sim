"""Tests for compact run serialization and observation/frame linkage."""

import json

import numpy as np

from nlos_sim.controllers.mock import MockController
from nlos_sim.entities import Robot, Wall
from nlos_sim.geometry.shapes import Box
from nlos_sim.simulation import Simulation
from nlos_sim.world import World
from nlos_sim.recording import RunRecorder


def test_recording_round_trip_links_frames_to_observations(tmp_path):
    world = World()
    world.add(Wall(Box(0.5, 1.0), np.array((3.25, 0.0, 0.0))))
    world.add(Wall(Box(4.0, 0.5), np.array((3.0, 2.25, 0.0))))
    sensing_robot = Robot(0.3, np.array((0.0, 0.0, 0.0)), nlos=True)
    los_only_robot = Robot(0.3, np.array((-3.0, -3.0, 0.0)))
    world.add(sensing_robot)
    world.add(los_only_robot)

    controller = MockController()
    controller_parameters = {"name": "mock-baseline", "speed_mps": 2.4}
    recorder = RunRecorder(
        world,
        world_name="serialization-test",
        controller=controller,
        controller_parameters=controller_parameters,
        fps=24,
        dt=1 / 24,
        sense_every=6,
    )
    simulation = Simulation(world, controller, recorder)
    for _ in range(7):
        simulation.step()

    output_path = recorder.save(tmp_path)
    run = json.loads(output_path.read_text(encoding="utf-8"))

    assert run["schema_version"] == 1
    assert run["run"]["controller"]["class"] == "MockController"
    assert run["run"]["controller"]["parameters"] == controller_parameters
    assert len(run["world"]["entities"]) == 4
    assert [item["before_step"] for item in run["observations"]] == [0, 6]
    assert len(run["frames"]) == 7
    assert [frame["step_index"] for frame in run["frames"]] == list(range(7))
    assert [frame["observation_id"] for frame in run["frames"]] == [
        "obs-000000",
        "obs-000000",
        "obs-000000",
        "obs-000000",
        "obs-000000",
        "obs-000000",
        "obs-000006",
    ]

    first_observation_robots = run["observations"][0]["robots"]
    by_id = {robot["robot_id"]: robot for robot in first_observation_robots}
    nlos_data = by_id[recorder.robot_ids[sensing_robot]]["nlos"]
    assert len(nlos_data["primary_readings"]) == 5
    assert nlos_data["samples"]
    assert len(nlos_data["histogram_counts"]) == 20
    assert sum(nlos_data["histogram_counts"]) == len(nlos_data["samples"])
    assert by_id[recorder.robot_ids[los_only_robot]]["nlos"] is None

    first_frame_robots = {item["robot_id"]: item for item in run["frames"][0]["robots"]}
    assert first_frame_robots[recorder.robot_ids[sensing_robot]]["requested_action"]
    assert first_frame_robots[recorder.robot_ids[sensing_robot]]["velocity_mps"]
    assert by_id[recorder.robot_ids[sensing_robot]]["pose"] != first_frame_robots[
        recorder.robot_ids[sensing_robot]
    ]["pose"]
