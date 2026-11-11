"""Shape-pair overlap queries with a small dispatch registry."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .shapes import Box, Circle, Shape
from .types import GEOMETRY_EPSILON, Contact, Pose, Vector

OverlapHandler = Callable[[Shape, Pose, Shape, Pose], Contact | None]
_OVERLAP_HANDLERS: dict[tuple[type[Shape], type[Shape]], OverlapHandler] = {}


def register_overlap(
    a_type: type[Shape], b_type: type[Shape]
) -> Callable[[OverlapHandler], OverlapHandler]:
    """Register a handler for one ordered pair of shape types."""

    def decorator(handler: OverlapHandler) -> OverlapHandler:
        _OVERLAP_HANDLERS[(a_type, b_type)] = handler
        return handler

    return decorator


def overlap(
    a_shape: Shape, a_pose: Pose, b_shape: Shape, b_pose: Pose
) -> Contact | None:
    """Return the contact between two positioned shapes, or None.

    The normal points from shape A to shape B. Touching shapes produce a
    contact with zero penetration.
    """
    handler = _find_handler(a_shape, b_shape)
    if handler is None:
        if isinstance(a_shape, Box) and isinstance(b_shape, Box):
            raise NotImplementedError("Box-box overlap is not implemented.")
        raise NotImplementedError(
            f"No overlap handler for {type(a_shape).__name__} "
            f"and {type(b_shape).__name__}."
        )
    return handler(a_shape, a_pose, b_shape, b_pose)


def _find_handler(a_shape: Shape, b_shape: Shape) -> OverlapHandler | None:
    for a_type in type(a_shape).__mro__:
        for b_type in type(b_shape).__mro__:
            handler = _OVERLAP_HANDLERS.get((a_type, b_type))
            if handler is not None:
                return handler
    return None


@register_overlap(Circle, Circle)
def _overlap_circle_circle(
    a_shape: Circle, a_pose: Pose, b_shape: Circle, b_pose: Pose
) -> Contact | None:
    """Return a contact with the point midway between the two surfaces."""
    center_a = np.asarray(a_pose[:2], dtype=float)
    center_b = np.asarray(b_pose[:2], dtype=float)
    delta = center_b - center_a
    distance = float(np.linalg.norm(delta))
    radius_sum = a_shape.radius + b_shape.radius
    if distance > radius_sum:
        return None

    penetration = radius_sum - distance
    if distance <= GEOMETRY_EPSILON:
        normal = np.array([1.0, 0.0], dtype=float)
        point = center_a.copy()
    else:
        normal = delta / distance
        point = center_a + normal * (a_shape.radius - penetration / 2.0)
    return Contact(normal, penetration, point)


@register_overlap(Box, Circle)
def _overlap_box_circle(
    box: Box, box_pose: Pose, circle: Circle, circle_pose: Pose
) -> Contact | None:
    """Return a contact whose normal points from the box toward the circle.

    When the circle center is inside the box, the normal points outward through
    the nearest face so moving the circle by the penetration separates it.
    """
    center = np.asarray(circle_pose[:2], dtype=float)
    lower, upper = box.bounds(box_pose)
    closest = box.closest_point(center, box_pose)
    offset = center - closest
    distance = float(np.linalg.norm(offset))

    if distance > GEOMETRY_EPSILON:
        if distance > circle.radius:
            return None
        normal = offset / distance
        return Contact(normal, circle.radius - distance, closest)

    face_distance, normal, point = _nearest_face(center, lower, upper)
    return Contact(normal, circle.radius + face_distance, point)


@register_overlap(Circle, Box)
def _overlap_circle_box(
    circle: Circle, circle_pose: Pose, box: Box, box_pose: Pose
) -> Contact | None:
    contact = _overlap_box_circle(box, box_pose, circle, circle_pose)
    if contact is None:
        return None
    return Contact(-contact.normal, contact.penetration, contact.point)


def _nearest_face(
    center: Vector, lower: Vector, upper: Vector
) -> tuple[float, Vector, Vector]:
    faces = [
        (
            center[0] - lower[0],
            np.array([-1.0, 0.0], dtype=float),
            np.array([lower[0], center[1]], dtype=float),
        ),
        (
            upper[0] - center[0],
            np.array([1.0, 0.0], dtype=float),
            np.array([upper[0], center[1]], dtype=float),
        ),
        (
            center[1] - lower[1],
            np.array([0.0, -1.0], dtype=float),
            np.array([center[0], lower[1]], dtype=float),
        ),
        (
            upper[1] - center[1],
            np.array([0.0, 1.0], dtype=float),
            np.array([center[0], upper[1]], dtype=float),
        ),
    ]
    distance, normal, point = min(faces, key=lambda face: face[0])
    return float(distance), normal, point

# EXTENSION POINT: register new shape-pair overlap handlers here.
