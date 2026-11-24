"""Save a static NLOS scan and path-length histogram in the T corridor."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlos_sim.entities import Material, Robot
from nlos_sim.environments import make_t_corridor
from nlos_sim.sensors.los import cast_los_scan
from nlos_sim.sensors.nlos import cast_nlos_scan
from nlos_sim.sensors.rays_utils import position
from nlos_sim.viz.draw import (
    draw_histogram,
    draw_los_scan,
    draw_nlos_scan,
    draw_robot,
    draw_walls,
    set_equal_view,
)

OUTPUT_PATH = Path("outputs/nlos.png")
OUTPUT_DPI = 150
ROBOT_POSE = (0.0, -2.0, np.pi / 2.0)
SECOND_ROBOT_POSE = np.array((1.5, 1.0, 0.0))

def main() -> None:
    """Build, save, and show one NLOS scan with its histogram."""
    world = make_t_corridor()
    robot = world.robots[0]
    robot.set_pose(*ROBOT_POSE)
    robot.nlos = True
    second_robot = Robot(
        radius=0.3,
        pose=SECOND_ROBOT_POSE,
        material=Material(albedo=0.6, color="#2a9d8f"),
    )
    world.add(second_robot)
    los_readings = cast_los_scan(world, robot)
    scan = cast_nlos_scan(world, robot)

    figure, (scene_ax, histogram_ax) = plt.subplots(
        1, 2, figsize=(13.0, 7.0)
    )
    draw_walls(scene_ax, world.walls)
    draw_los_scan(scene_ax, position(robot), los_readings)
    draw_nlos_scan(scene_ax, position(robot), scan)
    draw_robot(scene_ax, robot)
    draw_robot(scene_ax, second_robot)
    view_points = [
        corner
        for wall in world.walls
        for corner in wall.box.corners(wall.pose)
    ]
    set_equal_view(scene_ax, view_points)
    scene_ax.set_title("NLOS relay paths")
    draw_histogram(histogram_ax, scan.counts)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
