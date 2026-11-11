"""Shared value types for geometry queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..entities import Entity

Vector = np.ndarray
Pose = tuple[float, float, float]

ORIGIN_POSE: Pose = (0.0, 0.0, 0.0)
GEOMETRY_EPSILON = 1e-9


@dataclass(frozen=True)
class Segment:
    """A finite line segment between two points."""

    start: Vector
    end: Vector


@dataclass(frozen=True)
class RayHit:
    """A ray intersection with distance, point, unit normal, and optional entity."""

    distance: float
    point: Vector
    normal: Vector
    entity: Entity | None = None


@dataclass(frozen=True)
class Contact:
    """An overlap contact with a unit normal, penetration depth, and point."""

    normal: Vector
    penetration: float
    point: Vector
