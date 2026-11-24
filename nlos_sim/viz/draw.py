"""Reusable drawing helpers that only read their inputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Rectangle

from ..entities import Robot, Wall
from ..geometry.shapes import Box
from ..geometry.types import GEOMETRY_EPSILON, Contact, Pose, RayHit, Vector
from ..observation import NlosScan, Observation
from ..sensors.los import LOS_RANGE
from ..sensors.nlos import (
    BIN_EDGES,
    BIN_WIDTH,
    NLOS_MAX_PATH,
    PRIMARY_RAY_COUNT,
    SECONDARY_RAY_COUNT,
)

if TYPE_CHECKING:
    from ..simulation import Simulation

WALL_ALPHA = 0.9
WALL_EDGE_COLOR = "#4a4f55"
ROBOT_EDGE_COLOR = "#1a1a1a"
ROBOT_HEADING_COLOR = "#111111"
CONTACT_COLOR = "#1b7837"
PENETRATION_COLOR = "#6a3d9a"
ROBOT_PATH_COLOR = "#e52a09"

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
NLOS_PRIMARY_COLOR = "#d95f02"
NLOS_SECONDARY_COLOR = "#fc8d62"
NLOS_HIT_COLOR = "#b2182b"
NLOS_MISS_COLOR = "#9a9a9a"
NLOS_MARKER_SIZE = 12.0
LOS_RAY_COLOR = "#2878b5"
NLOS_HISTOGRAM_Y_MAX = PRIMARY_RAY_COUNT + SECONDARY_RAY_COUNT 


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


def draw_path(ax: Axes, points: Iterable[Vector]) -> None:
    """Draw a robot's traveled positions as a line."""
    path = np.asarray(list(points), dtype=float)
    if len(path) < 2:
        return
    ax.plot(path[:, 0], path[:, 1], color=ROBOT_PATH_COLOR, alpha=0.7, linewidth=1.2)


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
        xy=(float(end[0]), float(end[1])),
        xytext=(float(start[0]), float(start[1])),
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
    directions: list[Vector],
    hits: list[RayHit | None],
    max_range: float,
    colors: list[str],
    miss_color: str,
) -> None:
    """Draw a sequence of rays with matching hit and color sequences."""
    for direction, hit, color in zip(directions, hits, colors, strict=True):
        draw_ray(ax, origin, direction, hit, max_range, color, miss_color)


def draw_los_scan(
    ax: Axes,
    origin: Vector,
    readings: list[tuple[Vector, RayHit | None]],
) -> None:
    """Draw the robot's line-of-sight ray fan."""
    directions = [direction for direction, _ in readings]
    hits = [hit for _, hit in readings]
    draw_ray_fan(
        ax,
        origin,
        directions,
        hits,
        LOS_RANGE,
        [LOS_RAY_COLOR] * len(directions),
        NLOS_MISS_COLOR,
    )


def draw_nlos_scan(ax: Axes, origin: Vector, scan: NlosScan) -> None:
    """Draw relay points, faded secondary legs, and final hit points."""
    if scan.primary_readings:
        directions = [direction for direction, _ in scan.primary_readings]
        hits = [hit for _, hit in scan.primary_readings]
        draw_ray_fan(
            ax,
            origin,
            directions,
            hits,
            NLOS_MAX_PATH,
            [NLOS_PRIMARY_COLOR] * len(directions),
            NLOS_MISS_COLOR,
        )
    if not scan.samples:
        return
    origin = np.asarray(origin, dtype=float)
    relays = np.unique(
        [sample.relay_point for sample in scan.samples], axis=0
    )
    _draw_nlos_legs(ax, origin, relays, scan)
    hits = np.array([sample.hit_point for sample in scan.samples])
    ax.scatter(
        relays[:, 0], relays[:, 1], s=NLOS_MARKER_SIZE,
        color=NLOS_PRIMARY_COLOR, zorder=3,
    )
    ax.scatter(
        hits[:, 0], hits[:, 1], s=HIT_MARKER_SIZE,
        color=NLOS_HIT_COLOR, zorder=3,
    )


def _draw_nlos_legs(ax: Axes, origin: Vector, relays, scan: NlosScan) -> None:
    for relay in relays:
        ax.plot(
            [origin[0], relay[0]],
            [origin[1], relay[1]],
            color=NLOS_PRIMARY_COLOR,
            alpha=HIT_RAY_ALPHA,
            linewidth=RAY_LINEWIDTH,
        )
    for sample in scan.samples:
        relay, hit = sample.relay_point, sample.hit_point
        ax.plot(
            [relay[0], hit[0]],
            [relay[1], hit[1]],
            color=NLOS_SECONDARY_COLOR,
            alpha=MISS_RAY_ALPHA,
            linewidth=RAY_LINEWIDTH,
        )


def draw_histogram(ax: Axes, counts: np.ndarray) -> None:
    """Draw path-length counts using the sensor's fixed bins."""
    ax.bar(
        BIN_EDGES[:-1],
        counts,
        width=BIN_WIDTH,
        align="edge",
        color=NLOS_HIT_COLOR,
    )
    ax.set_xlabel("path length [m]")
    ax.set_ylabel("count")
    ax.set_xlim(BIN_EDGES[0], BIN_EDGES[-1])
    ax.set_ylim(0, NLOS_HISTOGRAM_Y_MAX)
    ax.set_yticks(np.linspace(0, NLOS_HISTOGRAM_Y_MAX, 6))
    ax.set_title("NLOS path lengths")


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
    points: list[Vector],
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


def draw_observation(
    ax: Axes,
    robot: Robot,
    observation: Observation,
    observed_pose: Pose,
) -> None:
    """Draw a robot's observation at its observation-time position."""
    origin = np.asarray(observed_pose[:2], dtype=float)
    if observation.nlos is not None:
        draw_nlos_scan(ax, origin, observation.nlos)


def draw_scene(
    sim: Simulation,
    ax: Axes,
    view_points: list[Vector],
) -> None:
    """Draw walls, robot paths, latest observations, and current robots."""
    ax.clear()
    draw_walls(ax, sim.world.walls)
    for robot in sim.world.robots:
        draw_path(ax, sim.paths[robot])
        observation = sim.observations.get(robot)
        if observation is not None:
            draw_observation(
                ax,
                robot,
                observation,
                sim.observation_poses[robot],
            )
        draw_robot(ax, robot)
    set_equal_view(ax, view_points)
    ax.set_title(f"frame {sim.frame}")


def draw_histogram_panel(
    sim: Simulation,
    robot: Robot,
    ax: Axes,
) -> None:
    """Show one robot's latest NLOS histogram."""
    ax.clear()
    observation = sim.observations.get(robot)
    counts = np.zeros(len(BIN_EDGES) - 1)
    if observation is not None and observation.nlos is not None:
        counts = observation.nlos.counts
    draw_histogram(ax, counts)
    robot_index = sim.world.robots.index(robot)
    ax.set_title(f"NLOS: robot {robot_index}")


def draw_frame(
    sim: Simulation,
    ax_scene: Axes,
    histogram_axes: list[Axes],
    nlos_robots: list[Robot],
    view_points: list[Vector],
) -> None:
    """Draw the scene and one histogram panel per NLOS robot."""
    draw_scene(sim, ax_scene, view_points)
    for robot, ax in zip(nlos_robots, histogram_axes, strict=True):
        draw_histogram_panel(sim, robot, ax)

