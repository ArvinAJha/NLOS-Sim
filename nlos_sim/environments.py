"""Environment builders."""

from __future__ import annotations

import numpy as np

from .entities import Robot, Wall
from .geometry.shapes import Box
from .world import World

JUNCTION_CENTER: tuple[float, float] = (0.0, 0.0)
ROBOT_HEADING_UP = np.pi / 2.0
SPAWN_CLEARANCE = 0.1

def make_nlos_example() -> World:
    """Create a sealed L-shaped corridor for an NLOS navigation example."""
    world = World()

    # Horizontal arm has its west end at x=-8 and turns upward at x=0.
    world.add(_wall(9.0, 0.3, -3.5, -1.15))
    world.add(_wall(0.3, 2.0, -8.15, 0.0))
    world.add(_wall(7.0, 0.3, -4.5, 1.15))

    # Vertical arm rises from the corner to y=8.
    world.add(_wall(0.3, 9.0, 1.15, 3.5))
    world.add(_wall(0.3, 7.0, -1.15, 4.5))
    world.add(_wall(2.6, 0.3, 0.0, 8.15))

    return world


def make_t_corridor(
    corridor_width: float = 2.0,
    bar_length: float = 12.0,
    stem_length: float = 8.0,
    wall_thickness: float = 0.5,
    robot_radius: float = 0.3,
) -> World:
    """Build a sealed T-shaped corridor with one robot in the stem.

    Dimensions are interior clear sizes. The bar's lower interior line is
    y = 0 and the junction center is the origin. The robot spawns near the
    bottom of the stem, heading up (+y).
    """
    world = World()
    for wall in _corridor_walls(
        corridor_width, bar_length, stem_length, wall_thickness
    ):
        world.add(wall)

    stem_bottom = JUNCTION_CENTER[1] - stem_length
    spawn_y = stem_bottom + robot_radius + SPAWN_CLEARANCE
    robot = Robot(
        robot_radius,
        np.array((JUNCTION_CENTER[0], spawn_y, ROBOT_HEADING_UP-0.2)),
    )
    world.add(robot)
    return world


def _corridor_walls(
    corridor_width: float,
    bar_length: float,
    stem_length: float,
    wall_thickness: float,
) -> list[Wall]:
    thickness = wall_thickness
    half_bar = bar_length / 2.0
    half_width = corridor_width / 2.0
    bar_bottom = JUNCTION_CENTER[1]
    bar_top = bar_bottom + corridor_width
    stem_bottom = bar_bottom - stem_length

    return [
        _wall(
            bar_length + 2.0 * thickness,
            thickness,
            JUNCTION_CENTER[0],
            bar_top + thickness / 2.0,
        ),
        _wall(
            half_bar + thickness - half_width,
            thickness,
            -(half_bar + thickness + half_width) / 2.0,
            bar_bottom - thickness / 2.0,
        ),
        _wall(
            half_bar + thickness - half_width,
            thickness,
            (half_bar + thickness + half_width) / 2.0,
            bar_bottom - thickness / 2.0,
        ),
        _wall(
            thickness,
            corridor_width + thickness,
            -half_bar - thickness / 2.0,
            bar_bottom + (corridor_width - thickness) / 2.0,
        ),
        _wall(
            thickness,
            corridor_width + thickness,
            half_bar + thickness / 2.0,
            bar_bottom + (corridor_width - thickness) / 2.0,
        ),
        _wall(
            thickness,
            stem_length + thickness,
            -half_width - thickness / 2.0,
            bar_bottom - (stem_length + thickness) / 2.0,
        ),
        _wall(
            thickness,
            stem_length + thickness,
            half_width + thickness / 2.0,
            bar_bottom - (stem_length + thickness) / 2.0,
        ),
        _wall(
            corridor_width + 2.0 * thickness,
            thickness,
            JUNCTION_CENTER[0],
            stem_bottom - thickness / 2.0,
        ),
    ]


def _wall(width: float, height: float, x: float, y: float) -> Wall:
    return Wall(Box(width, height), np.array((x, y, 0.0)))
