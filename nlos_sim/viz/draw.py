"""Reusable drawing helpers that only read their inputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Rectangle

from ..entities import Robot, Wall
from ..geometry.shapes import Box
from ..geometry.types import GEOMETRY_EPSILON, Contact, Pose, RayHit, Vector

WALL_ALPHA = 0.9
WALL_EDGE_COLOR = "#4a4f55"
ROBOT_EDGE_COLOR = "#1a1a1a"
ROBOT_HEADING_COLOR = "#111111"
CONTACT_COLOR = "#1b7837"
PENETRATION_COLOR = "#6a3d9a"

HEADING_ARROW_SCALE = 1.5
NORMAL_ARROW_LENGTH = 0.35
ARROW_LINEWIDTH = 1.1
PENETRATION_ARROW_LINEWIDTH = 1.8
ARROW_MUTATION_SCALE = 9.0
RAY_LINEWIDTH = 0.8
HIT_RAY_ALPHA = 0.75
MISS_RAY_ALPHA = 0.25
HIT_MARKER_SIZE = 9.0
CONTACT_MARKER_SIZE = 35.0
VIEW_MARGIN = 1.0


def draw_box(
    ax: Axes,
    box: Box,
    pose: Pose,
    color: str,
    alpha: float = WALL_ALPHA,
    edgecolor: str = WALL_EDGE_COLOR,
) -> None:
    """Draw an axis-aligned box at a pose."""
    center_x, center_y = float(pose[0]), float(pose[1])
    rectangle = Rectangle(
        (center_x - box.width / 2.0, center_y - box.height / 2.0),
        box.width,
        box.height,
        facecolor=color,
        edgecolor=edgecolor,
        alpha=alpha,
    )
    ax.add_patch(rectangle)


def draw_walls(ax: Axes, walls: Iterable[Wall]) -> None:
    """Draw every wall with its material color."""
    for wall in walls:
        draw_box(ax, wall.box, wall.pose, wall.material.color)


def draw_circle(
    ax: Axes,
    center: Vector,
    radius: float,
    color: str,
    alpha: float = 1.0,
    edgecolor: str = ROBOT_EDGE_COLOR,
) -> None:
    """Draw a filled circle."""
    patch = CirclePatch(
        (float(center[0]), float(center[1])),
        radius,
        facecolor=color,
        edgecolor=edgecolor,
        alpha=alpha,
    )
    ax.add_patch(patch)


def draw_robot(
    ax: Axes,
    robot: Robot,
    pose: Pose | None = None,
    color: str | None = None,
) -> None:
    """Draw a robot circle and its heading arrow."""
    draw_pose = robot.pose if pose is None else pose
    fill_color = robot.material.color if color is None else color
    draw_circle(ax, np.asarray(draw_pose[:2]), robot.radius, fill_color)
    draw_heading(ax, draw_pose, robot.radius)


def draw_heading(
    ax: Axes,
    pose: Pose,
    radius: float,
    color: str = ROBOT_HEADING_COLOR,
) -> None:
    """Draw an arrow from the robot center along its heading."""
    start = np.asarray(pose[:2], dtype=float)
    theta = float(pose[2])
    end = start + radius * HEADING_ARROW_SCALE * np.array(
        [np.cos(theta), np.sin(theta)]
    )
    draw_arrow(ax, start, end, color)


def draw_arrow(
    ax: Axes,
    start: Vector,
    end: Vector,
    color: str,
    linewidth: float = ARROW_LINEWIDTH,
) -> None:
    """Draw a single arrow from start to end."""
    ax.annotate(
        "",
        xy=np.asarray(end, dtype=float),
        xytext=np.asarray(start, dtype=float),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": linewidth,
            "mutation_scale": ARROW_MUTATION_SCALE,
        },
    )


def draw_ray(
    ax: Axes,
    origin: Vector,
    direction: Vector,
    hit: RayHit | None,
    max_range: float,
    color: str,
    miss_color: str,
) -> None:
    """Draw one ray to its hit point or its maximum range."""
    origin = np.asarray(origin, dtype=float)
    unit = np.asarray(direction, dtype=float)
    unit = unit / float(np.linalg.norm(unit))
    end = hit.point if hit is not None else origin + unit * max_range
    line_color = color if hit is not None else miss_color
    line_alpha = HIT_RAY_ALPHA if hit is not None else MISS_RAY_ALPHA
    ax.plot(
        [origin[0], end[0]],
        [origin[1], end[1]],
        color=line_color,
        alpha=line_alpha,
        linewidth=RAY_LINEWIDTH,
    )
    if hit is None:
        return
    ax.scatter(
        hit.point[0],
        hit.point[1],
        s=HIT_MARKER_SIZE,
        color=color,
        zorder=3,
    )
    draw_arrow(ax, hit.point, hit.point + hit.normal * NORMAL_ARROW_LENGTH, color)


def draw_ray_fan(
    ax: Axes,
    origin: Vector,
    directions: Sequence[Vector],
    hits: Sequence[RayHit | None],
    max_range: float,
    colors: Sequence[str],
    miss_color: str,
) -> None:
    """Draw a sequence of rays with matching hit and color sequences."""
    for direction, hit, color in zip(directions, hits, colors, strict=True):
        draw_ray(ax, origin, direction, hit, max_range, color, miss_color)


def draw_contact(
    ax: Axes,
    contact: Contact,
    penetration_origin: Vector | None = None,
) -> None:
    """Draw a contact point, its normal, and its penetration vector."""
    point = np.asarray(contact.point, dtype=float)
    ax.scatter(
        point[0],
        point[1],
        s=CONTACT_MARKER_SIZE,
        color=CONTACT_COLOR,
        zorder=4,
    )
    draw_arrow(
        ax,
        point,
        point + contact.normal * NORMAL_ARROW_LENGTH,
        CONTACT_COLOR,
    )
    if contact.penetration > GEOMETRY_EPSILON:
        start = point if penetration_origin is None else penetration_origin
        draw_arrow(
            ax,
            start,
            start + contact.normal * contact.penetration,
            PENETRATION_COLOR,
            linewidth=PENETRATION_ARROW_LINEWIDTH,
        )


def set_equal_view(
    ax: Axes,
    points: Sequence[Sequence[float]],
    margin: float = VIEW_MARGIN,
) -> None:
    """Set equal meter axes around the supplied points."""
    coordinates = np.asarray(points, dtype=float)
    lower = coordinates.min(axis=0) - margin
    upper = coordinates.max(axis=0) + margin
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

# EXTENSION POINT: animation would call these helpers per frame.
