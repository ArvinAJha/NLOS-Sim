"""What a robot senses in one frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry.types import RayHit, Vector


@dataclass
class NlosSample:
    """One two-bounce path through a relay point."""

    relay_point: np.ndarray
    hit_point: np.ndarray
    path_length: float  # meters. Time of flight = 2 * path_length / c


@dataclass
class NlosScan:
    """Primary fan readings, secondary samples, and path-length histogram."""

    samples: list[NlosSample]
    counts: np.ndarray  # bins are defined in sensors/nlos.py
    primary_readings: list[tuple[Vector, RayHit | None]] = field(default_factory=list)


@dataclass
class Observation:
    """LOS rays and, when the robot has NLOS on, an NLOS scan."""

    los: list[tuple[np.ndarray, RayHit | None]]  # (direction, hit); None = miss
    nlos: NlosScan | None = None  # None = robot has NLOS off
