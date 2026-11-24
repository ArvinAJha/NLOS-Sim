"""Live simulation loop: observe, decide, then step the world."""

from __future__ import annotations

import numpy as np
import random
from typing import TYPE_CHECKING

from .controllers.parent_controller import Controller
from .dynamics import Action
from .entities import Robot
from .observation import Observation
from .sensors.los import cast_los_scan
from .sensors.nlos import cast_nlos_scan
from .world import World
if TYPE_CHECKING:
    from .recording import RunRecorder


def observe(world: World, robot: Robot) -> Observation:
    """Collect the robot's LOS scan and, when enabled, its NLOS scan."""
    los = cast_los_scan(world, robot)
    nlos = cast_nlos_scan(world, robot) if robot.nlos else None
    return Observation(los, nlos)


class Simulation:
    """One shared controller and a fixed sensing cadence."""

    def __init__(
        self,
        world: World,
        controller: Controller,
        recorder: RunRecorder | None = None,
    ):
        self.world = world
        self.controller = controller
        self.recorder = recorder
        self.frame = 0
        self.observations: dict[Robot, Observation] = {}
        self.observation_poses: dict[Robot, np.ndarray] = {}
        self.paths: dict[Robot, list[np.ndarray]] = {
            robot: [robot.pose[:2].copy()] for robot in world.robots
        }
        self.observation_frame: int | None = None
        self.observation_id: str | None = None          # THIS SEEMS UNNECESSARY
        self.actions: dict[Robot, Action] = {}

    def sense_and_decide(self) -> None:
        # Observe every robot BEFORE anyone moves, so update order can't bias results.
        self.observations = {}

        for robot in self.world.robots:
            self.observations[robot] = observe(self.world, robot)


        self.actions = {
            robot: 
                self.controller.get_action(robot, self.observations[robot]) if random.random() > 0.05 else Action()
                for robot in self.world.robots
        }

        # recording
        if self.recorder is not None:
            self.observation_frame = self.frame
            self.observation_id = f"obs-{self.frame:06d}"
            self.observation_poses = {
                robot: robot.pose.copy() for robot in self.world.robots
            }
            
            self.recorder.record_observation(
                self.observation_frame,
                self.observation_id,
                self.world,
                self.observations,
                self.observation_poses,
            )

    def step(self, dt, sense_every) -> None:
        if self.frame % sense_every == 0:
            self.sense_and_decide()

        self.world.step(actions=self.actions, dt=dt)

        # recording
        for robot in self.world.robots:
            self.paths[robot].append(robot.pose[:2].copy())
        # if self.recorder is not None:
        #     # self.recorder.record_frame(
        #     #     self.frame,
        #     #     # (self.frame + 1) * DT,
        #     #     self.world,
        #     #     self.actions,
        #     #     self.observation_id,
        #     # )

        # advance
        self.frame += 1
