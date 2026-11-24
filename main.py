"""Live animation: robot + world -> observe -> controller -> Action -> step."""

from __future__ import annotations

import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from pathlib import Path

# from nlos_sim.controllers.controller import Controller
from nlos_sim.controllers.gradient import GradientController
from nlos_sim.controllers.mock import MockController
from nlos_sim.controllers.swarm import SwarmController
from nlos_sim.entities import Robot
from nlos_sim.environments import (
    make_nlos_example,
    make_t_corridor,
)
from nlos_sim.recording import RunRecorder
from nlos_sim.simulation import Simulation
from nlos_sim.viz.draw import (
    draw_frame,
)
from nlos_sim.world import World

FPS = 24
TOTAL_FRAMES = 240
SENSE_EVERY = 4  # frames between sensing + controller calls
DT = 1 / FPS  # seconds per physics frame

NLOS_EXAMPLE_START_A: tuple[float, float] = (-6.35, 0.2)
NLOS_EXAMPLE_START_B: tuple[float, float] = (0.5, 3.87)

def make_world() -> World:
    """A T corridor with three robots; two have NLOS enabled."""
    world = make_t_corridor()
    world.robots[0].nlos = True
    # world.add(Robot(radius=0.3, pose=np.array((3.2, 0.6, -0.2))))
    # world.add(Robot(radius=0.3, pose=np.array((4.0, 1.0, -1.0)), nlos=True))
    return world


def make_world_2() -> World:
    """Build the L-corner example with two NLOS robots facing opposite ends."""
    world = make_nlos_example()
    robot_a = Robot(
        radius=0.3,
        pose=np.array((*NLOS_EXAMPLE_START_A, np.pi*2)),
        nlos=True,
    )
    robot_b = Robot(
        radius=0.3,
        pose=np.array((*NLOS_EXAMPLE_START_B, -np.pi / 2.0)),
        nlos=True,
    )
    world.add(robot_a)
    world.add(robot_b)
    return world


def wall_points(world: World) -> list[np.ndarray]:
    """Corners used to frame the scene."""
    return [
        corner
        for wall in world.walls
        for corner in wall.box.corners(wall.pose)
    ]


def main(output_path: Path | None = None) -> FuncAnimation:
    """Run the simulation live or save its animation as a GIF."""
    # world = make_world()
    # target = JUNCTION_CENTER

    # controller_parameters = {
    #     "target": target,
    #     "repulsion_weight": 1.0,
    #     "repulsion_distance": 0.6,
    #     "target_weight": 0.5,
    # }
    world = make_world_2()
    robot_a, robot_b = world.robots
    goals = {
        robot_a: np.array(NLOS_EXAMPLE_START_B, dtype=float),
        robot_b: np.array(NLOS_EXAMPLE_START_A, dtype=float),
    }
    
    waypoints = {
        robot_a: [(-0.8, 0.0), NLOS_EXAMPLE_START_B],
        robot_b: [(0.0, 0.8), NLOS_EXAMPLE_START_A],
    }

    controller_parameters = {
        "goals": goals,
        "waypoints": waypoints,
        "repulsion_weight": 0.5,
        "repulsion_distance": 0.6,
        "aggregation_weight": 0.0,
        "target_weight": 1.0,
        "nlos_repulsion_weight": 1.8,
        "nlos_guidance_weight": 1.4,
        "nlos_guidance_distance": 3.5,
        "wall_lookahead": 2.5,
    }
    # controller = MockController()
    # controller = GradientController()
    controller = SwarmController(**controller_parameters)
    recorder = RunRecorder(
        world,
        world_name="nlos_example",
        # world_name="t_corridor",
        controller=controller,
        controller_parameters=controller_parameters,
        fps=FPS,
        dt=DT,
        sense_every=SENSE_EVERY,
    )
    sim = Simulation(world, controller, recorder)

    #
    # HISTOGRAM VISUALIZATION SETTINGS
    #
    view_points = wall_points(world)
    nlos_robots = [robot for robot in world.robots if robot.nlos]
    rows = max(1, math.ceil(math.sqrt(len(nlos_robots))))
    columns = max(1, math.ceil(len(nlos_robots) / rows))
    figure = plt.figure(figsize=(15.0, max(7.0, rows * 2.4)))
    layout = figure.add_gridspec(1, 2, width_ratios=(2.0, 1.3))
    ax_scene = figure.add_subplot(layout[0, 0])
    histogram_grid = layout[0, 1].subgridspec(rows, columns)
    histogram_axes = [
        figure.add_subplot(histogram_grid[row, column])
        for row in range(rows)
        for column in range(columns)
    ][: len(nlos_robots)]
    if not histogram_axes:
        histogram_axes = [figure.add_subplot(layout[0, 1])]
        histogram_axes[0].text(0.5, 0.5, "No NLOS-enabled robots", ha="center")
        histogram_axes[0].set_axis_off()


    def update(frame: int) -> None:
        sim.step(dt=DT, sense_every=SENSE_EVERY)
        draw_frame(sim, ax_scene, histogram_axes[: len(nlos_robots)], nlos_robots, view_points)
        figure.tight_layout()

    animation = FuncAnimation(
        figure,
        update,
        frames=TOTAL_FRAMES,
        interval=1000 / FPS,
        repeat=False,
    )
    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output_path, writer=PillowWriter(fps=FPS))
        plt.close(figure)
        print(f"Saved animation to {output_path}")
        
    run_path = recorder.save()
    print(f"Saved run data to {run_path}")
    return animation


if __name__ == "__main__":
    animation = main()