"""World entities: materials, static walls, and movable robots."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from .geometry.shapes import Box, Circle, Shape
from .geometry.types import ORIGIN_POSE, Pose, RayHit, Vector


@dataclass(frozen=True)
class Material:
    """Surface appearance values for future sensors and visualization."""

    albedo: float
    color: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.albedo <= 1.0:
            raise ValueError("Albedo must be between 0 and 1.")


DEFAULT_WALL_MATERIAL = Material(albedo=0.2, color="#b8bcc2")
DEFAULT_ROBOT_MATERIAL = Material(albedo=0.6, color="#2a6fdb")


class Entity(ABC):
    """Base class combining a shape, material, and world pose."""

    def __init__(
        self,
        shape: Shape,
        material: Material,
        pose: Sequence[float] = ORIGIN_POSE,
    ) -> None:
        self.shape = shape
        self.material = material
        self._pose = _as_pose(pose)

    @property
    def pose(self) -> Vector:
        """Return the mutable (x, y, theta) pose array."""
        return self._pose

    @pose.setter
    def pose(self, pose: Sequence[float]) -> None:
        self._pose = _as_pose(pose)

    @property
    def x(self) -> float:
        """Return the world x coordinate."""
        return float(self._pose[0])

    @property
    def y(self) -> float:
        """Return the world y coordinate."""
        return float(self._pose[1])

    @property
    def theta(self) -> float:
        """Return the heading in radians."""
        return float(self._pose[2])

    def set_pose(self, x: float, y: float, theta: float) -> None:
        """Place the entity at an arbitrary world pose."""
        self.pose = (x, y, theta)

    def raycast(
        self, origin: Vector, direction: Vector, max_range: float
    ) -> RayHit | None:
        """Cast a ray against this entity and attach the entity to the hit."""
        hit = self.shape.raycast(origin, direction, max_range, self._pose)
        if hit is None:
            return None
        return RayHit(hit.distance, hit.point, hit.normal, self)


class Wall(Entity):
    """A static wall backed by an axis-aligned box."""

    def __init__(
        self,
        box: Box,
        pose: Pose,
        material: Material = DEFAULT_WALL_MATERIAL,
    ) -> None:
        if not isinstance(box, Box):
            raise TypeError("Wall requires a Box shape.")
        super().__init__(box, material, pose)

    @property
    def box(self) -> Box:
        """Return the wall's box shape."""
        return cast(Box, self.shape)


class Robot(Entity):
    """A circular robot with a planar velocity and no sensors or controller."""

    def __init__(
        self,
        radius: float,
        pose: Pose,
        material: Material = DEFAULT_ROBOT_MATERIAL,
        nlos: bool = True,
    ) -> None:
        super().__init__(Circle(radius), material, pose)
        self.velocity = np.zeros(2, dtype=float)

    @property
    def radius(self) -> float:
        """Return the robot circle radius."""
        return cast(Circle, self.shape).radius
    los = None


def _as_pose(pose: Sequence[float]) -> Vector:
    value = np.asarray(pose, dtype=float)
    if value.shape != (3,):
        raise ValueError("Pose must contain (x, y, theta).")
    return value


# EXTENSION POINT: add sensor-bearing robot subclasses or new entity types here.
