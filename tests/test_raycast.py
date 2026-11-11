"""Tests for low-level and shape-level raycasting."""

import numpy as np
import pytest

from nlos_sim.geometry.raycast import ray_circle, ray_segment
from nlos_sim.geometry.shapes import Box, Circle
from nlos_sim.geometry.types import Segment


def test_ray_segment_hand_computed_hit() -> None:
    segment = Segment(np.array([5.0, -1.0]), np.array([5.0, 1.0]))

    hit = ray_segment(np.zeros(2), np.array([1.0, 0.0]), segment, 10.0)

    assert hit is not None
    assert hit.distance == pytest.approx(5.0)
    np.testing.assert_allclose(hit.point, [5.0, 0.0])
    np.testing.assert_allclose(hit.normal, [-1.0, 0.0])
    assert hit.entity is None


def test_ray_circle_hand_computed_hit() -> None:
    hit = ray_circle(
        np.zeros(2), np.array([1.0, 0.0]), np.array([5.0, 0.0]), 1.0, 10.0
    )

    assert hit is not None
    assert hit.distance == pytest.approx(4.0)
    np.testing.assert_allclose(hit.point, [4.0, 0.0])
    np.testing.assert_allclose(hit.normal, [-1.0, 0.0])


def test_ray_tangent_to_circle_is_a_hit() -> None:
    hit = ray_circle(
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([5.0, 0.0]),
        1.0,
        10.0,
    )

    assert hit is not None
    assert hit.distance == pytest.approx(5.0)
    np.testing.assert_allclose(hit.point, [5.0, 1.0])
    np.testing.assert_allclose(hit.normal, [0.0, 1.0])


def test_ray_circle_misses_when_aimed_beside_it() -> None:
    hit = ray_circle(
        np.array([0.0, 3.0]),
        np.array([1.0, 0.0]),
        np.array([5.0, 0.0]),
        1.0,
        10.0,
    )

    assert hit is None


def test_ray_pointing_away_from_circle_misses() -> None:
    hit = ray_circle(
        np.zeros(2), np.array([1.0, 0.0]), np.array([-5.0, 0.0]), 1.0, 10.0
    )

    assert hit is None


def test_ray_parallel_to_non_collinear_segment_misses() -> None:
    segment = Segment(np.array([2.0, 1.0]), np.array([4.0, 1.0]))

    hit = ray_segment(
        np.zeros(2), np.array([1.0, 0.0]), segment, 10.0
    )

    assert hit is None


def test_ray_collinear_with_segment_misses() -> None:
    segment = Segment(np.array([2.0, 0.0]), np.array([4.0, 0.0]))

    hit = ray_segment(
        np.zeros(2), np.array([1.0, 0.0]), segment, 10.0
    )

    assert hit is None


def test_ray_starting_on_segment_reports_zero_distance() -> None:
    segment = Segment(np.array([5.0, -1.0]), np.array([5.0, 1.0]))

    hit = ray_segment(
        np.array([5.0, 0.0]), np.array([1.0, 0.0]), segment, 10.0
    )

    assert hit is not None
    assert hit.distance == pytest.approx(0.0)
    np.testing.assert_allclose(hit.point, [5.0, 0.0])
    np.testing.assert_allclose(hit.normal, [-1.0, 0.0])


def test_ray_starting_on_circle_reports_zero_distance() -> None:
    hit = ray_circle(
        np.array([4.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([5.0, 0.0]),
        1.0,
        10.0,
    )

    assert hit is not None
    assert hit.distance == pytest.approx(0.0)
    np.testing.assert_allclose(hit.point, [4.0, 0.0])
    np.testing.assert_allclose(hit.normal, [-1.0, 0.0])


def test_hit_exactly_at_max_range_is_included() -> None:
    segment = Segment(np.array([5.0, -1.0]), np.array([5.0, 1.0]))

    hit = ray_segment(np.zeros(2), np.array([1.0, 0.0]), segment, 5.0)

    assert hit is not None
    assert hit.distance == pytest.approx(5.0)


def test_hit_just_beyond_max_range_misses() -> None:
    segment = Segment(np.array([5.0, -1.0]), np.array([5.0, 1.0]))

    hit = ray_segment(
        np.zeros(2), np.array([1.0, 0.0]), segment, 5.0 - 1e-6
    )

    assert hit is None


def test_zero_length_direction_raises_for_segment() -> None:
    segment = Segment(np.array([5.0, -1.0]), np.array([5.0, 1.0]))

    with pytest.raises(ValueError, match="non-zero"):
        ray_segment(np.zeros(2), np.zeros(2), segment, 10.0)


def test_zero_length_direction_raises_for_circle() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        ray_circle(np.zeros(2), np.zeros(2), np.array([5.0, 0.0]), 1.0, 10.0)


def test_zero_length_direction_raises_for_shape_raycast() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        Circle(1.0).raycast(np.zeros(2), np.zeros(2), 10.0)


def test_ray_starting_on_circle_pointing_outward_reports_zero() -> None:
    hit = ray_circle(
        np.array([4.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([5.0, 0.0]),
        1.0,
        10.0,
    )

    assert hit is not None
    assert hit.distance == pytest.approx(0.0)


def test_non_unit_direction_still_returns_world_distance() -> None:
    segment = Segment(np.array([5.0, -1.0]), np.array([5.0, 1.0]))

    hit = ray_segment(np.zeros(2), np.array([2.0, 0.0]), segment, 10.0)

    assert hit is not None
    assert hit.distance == pytest.approx(5.0)


def test_box_raycast_chooses_nearest_face() -> None:
    box = Box(2.0, 2.0)

    hit = box.raycast(
        np.zeros(2), np.array([1.0, 0.0]), 10.0, pose=(5.0, 0.0, 0.0)
    )

    assert hit is not None
    assert hit.distance == pytest.approx(4.0)
    np.testing.assert_allclose(hit.point, [4.0, 0.0])
    np.testing.assert_allclose(hit.normal, [-1.0, 0.0])


def test_box_corner_hit_is_on_an_edge_normal() -> None:
    box = Box(2.0, 2.0)
    direction = np.array([1.0, 1.0]) / np.sqrt(2.0)

    hit = box.raycast(np.array([-2.0, -2.0]), direction, 10.0)

    assert hit is not None
    assert hit.distance == pytest.approx(np.sqrt(2.0))
    np.testing.assert_allclose(hit.point, [-1.0, -1.0])
    assert any(
        np.allclose(hit.normal, candidate)
        for candidate in ([-1.0, 0.0], [0.0, -1.0])
    )


def test_circle_shape_uses_pose_translation() -> None:
    circle = Circle(1.0)

    hit = circle.raycast(
        np.zeros(2), np.array([1.0, 0.0]), 10.0, pose=(5.0, 0.0, 0.0)
    )

    assert hit is not None
    assert hit.distance == pytest.approx(4.0)
    assert hit.entity is None


def test_circle_normal_faces_origin_from_inside() -> None:
    hit = ray_circle(
        np.zeros(2), np.array([1.0, 0.0]), np.zeros(2), 1.0, 10.0
    )

    assert hit is not None
    assert hit.distance == pytest.approx(1.0)
    np.testing.assert_allclose(hit.point, [1.0, 0.0])
    np.testing.assert_allclose(hit.normal, [-1.0, 0.0])
