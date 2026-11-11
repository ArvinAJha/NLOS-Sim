"""Tests for overlap queries and dispatch."""

import numpy as np
import pytest

from nlos_sim.geometry.overlap import overlap
from nlos_sim.geometry.shapes import Box, Circle


def test_circle_circle_separated_returns_none() -> None:
    contact = overlap(
        Circle(1.0), (0.0, 0.0, 0.0), Circle(1.0), (5.0, 0.0, 0.0)
    )

    assert contact is None


def test_circle_circle_touching_has_zero_penetration() -> None:
    contact = overlap(
        Circle(1.0), (0.0, 0.0, 0.0), Circle(1.0), (2.0, 0.0, 0.0)
    )

    assert contact is not None
    assert contact.penetration == pytest.approx(0.0)
    np.testing.assert_allclose(contact.normal, [1.0, 0.0])
    np.testing.assert_allclose(contact.point, [1.0, 0.0])


def test_circle_circle_penetrating_contact() -> None:
    contact = overlap(
        Circle(1.0), (0.0, 0.0, 0.0), Circle(1.0), (1.5, 0.0, 0.0)
    )

    assert contact is not None
    assert contact.penetration == pytest.approx(0.5)
    np.testing.assert_allclose(contact.normal, [1.0, 0.0])
    np.testing.assert_allclose(contact.point, [0.75, 0.0])


def test_circle_circle_reversed_normal_points_other_way() -> None:
    contact = overlap(
        Circle(1.0), (1.5, 0.0, 0.0), Circle(1.0), (0.0, 0.0, 0.0)
    )

    assert contact is not None
    np.testing.assert_allclose(contact.normal, [-1.0, 0.0])


def test_concentric_circles_use_deterministic_normal() -> None:
    contact = overlap(
        Circle(1.0), (0.0, 0.0, 0.0), Circle(2.0), (0.0, 0.0, 0.0)
    )

    assert contact is not None
    assert contact.penetration == pytest.approx(3.0)
    np.testing.assert_allclose(contact.normal, [1.0, 0.0])
    np.testing.assert_allclose(contact.point, [0.0, 0.0])


def test_circle_box_separated_returns_none() -> None:
    contact = overlap(
        Box(2.0, 2.0), (0.0, 0.0, 0.0), Circle(1.0), (-3.0, 0.0, 0.0)
    )

    assert contact is None


def test_circle_box_touching_has_zero_penetration() -> None:
    contact = overlap(
        Box(2.0, 2.0), (0.0, 0.0, 0.0), Circle(1.0), (-2.0, 0.0, 0.0)
    )

    assert contact is not None
    assert contact.penetration == pytest.approx(0.0)
    np.testing.assert_allclose(contact.normal, [-1.0, 0.0])
    np.testing.assert_allclose(contact.point, [-1.0, 0.0])


def test_circle_box_penetrating_contact() -> None:
    contact = overlap(
        Box(2.0, 2.0), (0.0, 0.0, 0.0), Circle(1.0), (-1.5, 0.0, 0.0)
    )

    assert contact is not None
    assert contact.penetration == pytest.approx(0.5)
    np.testing.assert_allclose(contact.normal, [-1.0, 0.0])
    np.testing.assert_allclose(contact.point, [-1.0, 0.0])


def test_circle_box_reversed_arguments_flip_normal() -> None:
    contact = overlap(
        Circle(1.0), (-1.5, 0.0, 0.0), Box(2.0, 2.0), (0.0, 0.0, 0.0)
    )

    assert contact is not None
    assert contact.penetration == pytest.approx(0.5)
    np.testing.assert_allclose(contact.normal, [1.0, 0.0])
    np.testing.assert_allclose(contact.point, [-1.0, 0.0])


def test_circle_center_inside_box_uses_nearest_face() -> None:
    contact = overlap(
        Box(2.0, 2.0), (0.0, 0.0, 0.0), Circle(0.5), (0.3, 0.2, 0.0)
    )

    assert contact is not None
    assert contact.penetration == pytest.approx(1.2)
    np.testing.assert_allclose(contact.normal, [1.0, 0.0])
    np.testing.assert_allclose(contact.point, [1.0, 0.2])


def test_circle_center_inside_box_reversed_normal_is_inward() -> None:
    contact = overlap(
        Circle(0.5), (0.3, 0.2, 0.0), Box(2.0, 2.0), (0.0, 0.0, 0.0)
    )

    assert contact is not None
    np.testing.assert_allclose(contact.normal, [-1.0, 0.0])


def test_circle_center_on_box_boundary_is_pushed_out() -> None:
    contact = overlap(
        Box(2.0, 2.0), (0.0, 0.0, 0.0), Circle(0.5), (1.0, 0.0, 0.0)
    )

    assert contact is not None
    assert contact.penetration == pytest.approx(0.5)
    np.testing.assert_allclose(contact.normal, [1.0, 0.0])
    np.testing.assert_allclose(contact.point, [1.0, 0.0])


def test_box_box_overlap_raises_clear_error() -> None:
    with pytest.raises(NotImplementedError, match="Box-box"):
        overlap(
            Box(1.0, 1.0),
            (0.0, 0.0, 0.0),
            Box(1.0, 1.0),
            (0.0, 0.0, 0.0),
        )
