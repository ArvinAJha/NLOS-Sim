"""Planar shapes and their raycasting behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .raycast import ray_circle, ray_segment
from .types import ORIGIN_POSE, Pose, RayHit, Segment, Vector


class Shape(ABC):
    """A shape centered on its local origin and positioned by a pose."""

    @abstractmethod
    def raycast(
        self,
        origin: Vector,
        direction: Vector,
        max_range: float,
        pose: Pose = ORIGIN_POSE,
    ) -> RayHit | None:
        """Return the nearest ray hit at the given pose, or None."""


@dataclass(frozen=True)
class Circle(Shape):
    """A circle with the given radius."""

    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("Circle radius must be positive.")

    def raycast(
        self,
        origin: Vector,
        direction: Vector,
        max_range: float,
        pose: Pose = ORIGIN_POSE,
    ) -> RayHit | None:
        """Cast a ray against the translated circle."""
        center = np.asarray(pose[:2], dtype=float)
        return ray_circle(origin, direction, center, self.radius, max_range)


@dataclass(frozen=True)
class Box(Shape):
    """An axis-aligned box; pose rotation is intentionally ignored for now."""

    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Box width and height must be positive.")

    def corners(self, pose: Pose = ORIGIN_POSE) -> np.ndarray:
        """Return the four world-space corners in counter-clockwise order."""
        center_x, center_y = float(pose[0]), float(pose[1])
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        return np.array(
            [
                [center_x - half_width, center_y - half_height],
                [center_x + half_width, center_y - half_height],
                [center_x + half_width, center_y + half_height],
                [center_x - half_width, center_y + half_height],
            ],
            dtype=float,
        )

    def bounds(self, pose: Pose = ORIGIN_POSE) -> tuple[Vector, Vector]:
        """Return the lower-left and upper-right box corners."""
        center = np.asarray(pose[:2], dtype=float)
        half_size = np.array([self.width / 2.0, self.height / 2.0], dtype=float)
        return center - half_size, center + half_size

    def closest_point(self, point: Vector, pose: Pose = ORIGIN_POSE) -> Vector:
        """Return the point on or inside the box closest to a query point."""
        lower, upper = self.bounds(pose)
        return np.clip(np.asarray(point, dtype=float), lower, upper)

    def segments(self, pose: Pose = ORIGIN_POSE) -> tuple[Segment, ...]:
        """Expose the four box edges as segments."""
        corners = self.corners(pose)
        return tuple(
            Segment(corners[index], corners[(index + 1) % len(corners)])
            for index in range(len(corners))
        )

    def raycast(
        self,
        origin: Vector,
        direction: Vector,
        max_range: float,
        pose: Pose = ORIGIN_POSE,
    ) -> RayHit | None:
        """Cast a ray against the nearest of the four box edges."""
        hits = [
            ray_segment(origin, direction, segment, max_range)
            for segment in self.segments(pose)
        ]
        valid_hits = [hit for hit in hits if hit is not None]
        if not valid_hits:
            return None
        return min(valid_hits, key=lambda hit: hit.distance)

# EXTENSION POINT: add rotated boxes, polygons, or other shapes by subclassing Shape.
