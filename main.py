import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from nlos_sim.controllers.controller import MockController, BaseController, GradientController
from nlos_sim.dynamics import Action
from nlos_sim.entities import Robot
from nlos_sim.environments import make_t_corridor
from nlos_sim.geometry.types import Pose, RayHit, Vector
from nlos_sim.world import World
from nlos_sim.viz.draw import draw_ray_fan, draw_robot, draw_walls, set_equal_view

FPS = 24
MOVEMENT_FRAMES = 5
TOTAL_FRAMES = 240
SENSING_INTERVAL = MOVEMENT_FRAMES + 1
TIME_STEP = 1.0 / FPS
FOV = np.deg2rad(45.0)
RAY_COUNT = 16
MAX_RANGE = 6.0
RAY_COLOR = "#2878b5"
MISS_COLOR = "#9a9a9a"
INITIAL_ROBOT_COLOR = "#aab4bf"
ROBOT_COLORS = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]


def _cast_fan(
    world: World, robot: Robot
) -> tuple[Vector, list[Vector], list[RayHit | None]]:
    angles = np.linspace(robot.theta - FOV / 2, robot.theta + FOV / 2, RAY_COUNT)
    directions = [np.array([np.cos(angle), np.sin(angle)]) for angle in angles]
    origin = robot.pose[:2].copy()
    hits = [
        world.raycast(origin, direction, MAX_RANGE, ignore=robot)
        for direction in directions
    ]
    return origin, directions, hits


def _draw_scene(
    ax: Axes,
    world: World,
    robots: list[Robot],
    poses: list[Pose],
    wall_points: list[tuple[float, float]],
    scans: list[tuple[Vector, list[Vector], list[RayHit | None]]],
    paths: list[list[Pose]] | None = None,
) -> None:
    ax.clear()
    draw_walls(ax, world.walls)
    for scan_index, (origin, directions, hits) in enumerate(scans):
        color = RAY_COLOR 
        draw_ray_fan(
            ax,
            origin,
            directions,
            hits,
            MAX_RANGE,
            [color] * len(directions),
            MISS_COLOR,
        )
    for robot_index, (robot, pose) in enumerate(zip(robots, poses, strict=True)):
        path = None if paths is None else paths[robot_index]
        if path:
            points = np.asarray(path)
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=ROBOT_COLORS[robot_index],
                linewidth=1.5,
            )
            if len(path) > 1:
                draw_robot(ax, robot, pose=path[0], color=INITIAL_ROBOT_COLOR)
        draw_robot(ax, robot, pose=pose, color=robot.material.color)
    set_equal_view(ax, wall_points)


def _simulate_cycle(
    world: World,
    robots: list[Robot],
    controller: BaseController,
    movement_frames: int,
):
    # LOS
    scans = [_cast_fan(world, robot) for robot in robots]

    # add NLOS here


    actions: dict[Robot, Action] = {
        robot: # robot
        
        controller.get_action( # their action
            robot, list(zip(directions, hits, strict=True))
        )
        for robot, (_, directions, hits) in zip(
            robots, scans, strict=True
        )
    }

    poses: list[Pose] = [
        (robot.x, robot.y, robot.theta) for robot in robots
    ]
    frames = [poses]

    for _ in range(movement_frames):
        world.step(TIME_STEP, actions)
        frames.append([(robot.x, robot.y, robot.theta) for robot in robots])

    return frames, scans, actions


def draw_frames() -> None:
    """Animate one scan frame followed by five controller-driven steps."""

    world = make_t_corridor()
    world.add(Robot(radius=0.3, pose=(3.0, 1.0, 0.0)))
    world.add(Robot(radius=0.3, pose=(3.0, 2.0, -1.0)))

    robots = world.robots
    controller = GradientController()
    wall_points = [
        (float(corner[0]), float(corner[1]))
        for wall in world.walls
        for corner in wall.box.corners((wall.x, wall.y, wall.theta))
    ]
    frame_data = []

    for frame_start in range(0, TOTAL_FRAMES, SENSING_INTERVAL):
        movement_frames = min(
            MOVEMENT_FRAMES, TOTAL_FRAMES - frame_start - 1
        )
        cycle_frames, scans, actions = _simulate_cycle(
            world, robots, controller, movement_frames
        )
        frame_data.extend((poses, scans, actions) for poses in cycle_frames)

    figure, (before_ax, current_ax) = plt.subplots(1, 2, figsize=(13.0, 7.0))
    initial_poses, initial_scans, _ = frame_data[0]
    _draw_scene(
        before_ax,
        world,
        robots,
        initial_poses,
        wall_points,
        initial_scans,
    )
    before_ax.set_title("Before: both robots' raycasts")

    def update(frame: int) -> tuple[()]:
        poses, scans, actions = frame_data[frame]
        paths = [
            [snapshot[0][robot_index] for snapshot in frame_data[: frame + 1]]
            for robot_index in range(len(robots))
        ]
        _draw_scene(
            current_ax,
            world,
            robots,
            poses,
            wall_points,
            scans,
            paths,
        )
        movement_index = frame % SENSING_INTERVAL
        if movement_index == 0:
            stage = "Raycast and controller action"
        else:
            stage = f"Mock action {movement_index}/{MOVEMENT_FRAMES}"
        action_labels = ", ".join(
            f"R{index + 1}=({action.vx:.1f}, {action.vy:.1f})"
            for index, action in enumerate(actions.values())
        )
        current_ax.set_title(f"{stage} | velocities: {action_labels} m/s")
        figure.tight_layout()
        return ()

    animation = FuncAnimation(
        figure,
        update,
        frames=range(len(frame_data)),
        interval=1000 / FPS,
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    _ = animation


if __name__ == "__main__":
    draw_frames()