"""Static figure builders for the scene, raycast fan, and overlap cases."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from ..entities import Robot, Wall
from ..geometry.overlap import overlap
from ..geometry.types import GEOMETRY_EPSILON, Pose, RayHit, Vector
from ..world import World
from .draw import (
    CONTACT_COLOR,
    CONTACT_MARKER_SIZE,
    draw_box,
    draw_contact,
    draw_ray_fan,
    draw_robot,
    draw_walls,
    set_equal_view,
)

RAYCAST_DEMO_MAX_RANGE = 6.0
RAYCAST_FAN_COUNT = 61
RAYCAST_FAN_HALF_ANGLE = np.pi / 2.0
MISS_COLOR = "#9a9a9a"
UNKNOWN_HIT_COLOR = "#666666"
OVERLAP_ROBOT_COLOR = "#d62728"

SCENE_FIGURE_SIZE = (10.0, 7.0)
RAYCAST_PANEL_SIZE = (6.0, 7.0)
OVERLAP_PANEL_SIZE = (4.2, 4.5)
OVERLAP_ANNOTATION_POSITION = (0.03, 0.97)
ANNOTATION_FONT_SIZE = 9.0

RAY_COLORMAP = colormaps["tab10"]
RAY_COLOR_COUNT = len(RAY_COLORMAP.colors)


def make_scene_figure(world: World) -> Figure:
    """Draw the world's walls and robots."""
    figure, ax = plt.subplots(figsize=SCENE_FIGURE_SIZE)
    draw_walls(ax, world.walls)
    for robot in world.robots:
        color = (
            OVERLAP_ROBOT_COLOR
            if _is_overlapping(world, robot)
            else robot.material.color
        )
        draw_robot(ax, robot, color=color)
    set_equal_view(ax, _wall_points(world.walls))
    ax.set_title("T-shaped corridor")
    figure.tight_layout()
    return figure


def make_raycast_figure(
    world: World,
    panels: Sequence[tuple[str, Pose]],
    max_range: float = RAYCAST_DEMO_MAX_RANGE,
) -> Figure:
    """Draw side-by-side raycast fans from several robot poses."""
    if not world.robots:
        raise ValueError("Raycast figure requires at least one robot.")
    robot = world.robots[0]
    wall_colors = _wall_colors(world.walls)
    view_points = _wall_points(world.walls)
    figure_width = RAYCAST_PANEL_SIZE[0] * len(panels)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(figure_width, RAYCAST_PANEL_SIZE[1]),
        squeeze=False,
    )
    for ax, (label, pose) in zip(axes[0], panels, strict=True):
        _draw_raycast_panel(
            ax, world, robot, pose, wall_colors, max_range, view_points
        )
        ax.set_title(label)
    figure.suptitle("Raycast fan (gray rays miss or exceed range)")
    figure.tight_layout()
    return figure


def make_overlap_figure(
    wall: Wall,
    robot: Robot,
    panels: Sequence[tuple[str, Pose]],
) -> Figure:
    """Draw several circle-box overlap cases side by side."""
    view_points = list(wall.box.corners(wall.pose))
    for _, pose in panels:
        view_points.extend(_robot_points(robot, pose))
    figure_width = OVERLAP_PANEL_SIZE[0] * len(panels)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(figure_width, OVERLAP_PANEL_SIZE[1]),
        squeeze=False,
    )
    for ax, (label, pose) in zip(axes[0], panels, strict=True):
        _draw_overlap_panel(ax, wall, robot, pose, view_points)
        ax.set_title(label)
    figure.suptitle("Circle-box overlap cases")
    figure.tight_layout()
    return figure


def _draw_raycast_panel(
    ax: Axes,
    world: World,
    robot: Robot,
    pose: Pose,
    wall_colors: dict[Wall, str],
    max_range: float,
    view_points: Sequence[Sequence[float]],
) -> None:
    draw_walls(ax, world.walls)
    origin = np.asarray(pose[:2], dtype=float)
    directions = _fan_directions(pose[2])
    hits = [
        world.raycast(origin, direction, max_range, ignore=world.robots)
        for direction in directions
    ]
    colors = [_hit_color(hit, wall_colors) for hit in hits]
    draw_ray_fan(
        ax,
        origin,
        directions,
        hits,
        max_range,
        colors,
        MISS_COLOR,
    )
    draw_robot(ax, robot, pose=pose)
    set_equal_view(ax, view_points)


def _draw_overlap_panel(
    ax: Axes,
    wall: Wall,
    robot: Robot,
    pose: Pose,
    view_points: Sequence[Sequence[float]],
) -> None:
    draw_box(ax, wall.box, wall.pose, wall.material.color)
    contact = overlap(wall.shape, wall.pose, robot.shape, pose)
    overlapping = (
        contact is not None and contact.penetration > GEOMETRY_EPSILON
    )
    color = OVERLAP_ROBOT_COLOR if overlapping else robot.material.color
    draw_robot(ax, robot, pose=pose, color=color)

    if contact is None:
        _draw_closest_point(ax, wall, pose)
        annotation = "no contact"
    else:
        draw_contact(ax, contact, penetration_origin=np.asarray(pose[:2]))
        annotation = f"penetration = {contact.penetration:.3f} m"
    ax.text(
        *OVERLAP_ANNOTATION_POSITION,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
    )
    set_equal_view(ax, view_points)


def _fan_directions(heading: float) -> np.ndarray:
    angles = np.linspace(
        heading - RAYCAST_FAN_HALF_ANGLE,
        heading + RAYCAST_FAN_HALF_ANGLE,
        RAYCAST_FAN_COUNT,
    )
    angles = np.append(angles, heading)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _wall_colors(walls: Sequence[Wall]) -> dict[Wall, str]:
    return {
        wall: to_hex(RAY_COLORMAP(index % RAY_COLOR_COUNT))
        for index, wall in enumerate(walls)
    }


def _hit_color(hit: RayHit | None, wall_colors: dict[Wall, str]) -> str:
    if hit is None:
        return MISS_COLOR
    return wall_colors.get(hit.entity, UNKNOWN_HIT_COLOR)


def _is_overlapping(world: World, robot: Robot) -> bool:
    return any(
        contact.penetration > GEOMETRY_EPSILON
        for _, contact in world.overlaps(robot)
    )


def _wall_points(walls: Sequence[Wall]) -> list[Vector]:
    return [
        corner
        for wall in walls
        for corner in wall.box.corners(wall.pose)
    ]


def _robot_points(robot: Robot, pose: Pose) -> list[Vector]:
    center = np.asarray(pose[:2], dtype=float)
    radius = np.array([robot.radius, robot.radius])
    return [center - radius, center + radius]


def _draw_closest_point(ax: Axes, wall: Wall, pose: Pose) -> None:
    center = np.asarray(pose[:2], dtype=float)
    point = wall.box.closest_point(center, wall.pose)
    ax.scatter(
        point[0],
        point[1],
        s=CONTACT_MARKER_SIZE,
        facecolors="none",
        edgecolors=CONTACT_COLOR,
        zorder=4,
    )
