"""Low-level ray intersections with circles and line segments."""

from __future__ import annotations

import numpy as np

from .types import GEOMETRY_EPSILON, RayHit, Segment, Vector


def ray_segment(
    origin: Vector, direction: Vector, segment: Segment, max_range: float
) -> RayHit | None:
    """Return the nearest hit on a segment within max_range, or None.

    Parallel or collinear rays miss because they have no single intersection.
    Segments are treated as closed, so hitting an endpoint counts. A ray that
    starts exactly on the segment reports distance zero. The normal faces the
    ray origin.
    """
    origin = np.asarray(origin, dtype=float)
    unit = _unit_direction(direction)
    start = np.asarray(segment.start, dtype=float)
    end = np.asarray(segment.end, dtype=float)
    segment_vector = end - start
    denominator = _cross(unit, segment_vector)
    if abs(denominator) <= GEOMETRY_EPSILON:
        return None

    offset = start - origin
    distance = _cross(offset, segment_vector) / denominator
    parameter = _cross(offset, unit) / denominator
    if not _within(distance, 0.0, max_range):
        return None
    if not _within(parameter, 0.0, 1.0):
        return None

    distance = max(distance, 0.0)
    point = origin + unit * distance
    normal = _perpendicular(segment_vector)
    normal = _facing_origin(normal, point, origin, unit)
    return RayHit(distance, point, normal)


def ray_circle(
    origin: Vector,
    direction: Vector,
    center: Vector,
    radius: float,
    max_range: float,
) -> RayHit | None:
    """Return the nearest hit on a circle within max_range, or None.

    Tangent rays count as hits. A ray that starts on the circle reports
    distance zero even if it points outward. The normal faces the ray origin.
    """
    if radius <= 0.0:
        raise ValueError("Circle radius must be positive.")
    origin = np.asarray(origin, dtype=float)
    unit = _unit_direction(direction)
    center = np.asarray(center, dtype=float)

    offset = origin - center
    projection = float(np.dot(offset, unit))
    discriminant = projection * projection - (
        float(np.dot(offset, offset)) - radius * radius
    )
    if discriminant < -GEOMETRY_EPSILON:
        return None

    root = np.sqrt(max(discriminant, 0.0))
    near_distance = -projection - root
    far_distance = -projection + root
    if near_distance >= -GEOMETRY_EPSILON:
        distance = max(near_distance, 0.0)
    elif far_distance >= -GEOMETRY_EPSILON:
        distance = max(far_distance, 0.0)
    else:
        return None
    if distance > max_range + GEOMETRY_EPSILON:
        return None

    point = origin + unit * distance
    radial = (point - center) / radius
    normal = _facing_origin(radial, point, origin, unit)
    return RayHit(distance, point, normal)


def _unit_direction(direction: Vector) -> Vector:
    vector = np.asarray(direction, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= GEOMETRY_EPSILON:
        raise ValueError("Ray direction must be non-zero.")
    return vector / length


def _cross(first: Vector, second: Vector) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _within(value: float, lower: float, upper: float) -> bool:
    return lower - GEOMETRY_EPSILON <= value <= upper + GEOMETRY_EPSILON


def _perpendicular(vector: Vector) -> Vector:
    length = float(np.linalg.norm(vector))
    return np.array([-vector[1] / length, vector[0] / length], dtype=float)


def _facing_origin(
    normal: Vector, point: Vector, origin: Vector, direction: Vector
) -> Vector:
    to_origin = origin - point
    if float(np.dot(normal, to_origin)) < 0.0:
        return -normal
    if (
        float(np.linalg.norm(to_origin)) <= GEOMETRY_EPSILON
        and float(np.dot(normal, direction)) > 0.0
    ):
        return -normal
    return normal
