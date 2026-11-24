"""Compact JSON recording for reproducible simulation frames."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .controllers.parent_controller import Controller
from .dynamics import Action, DEFAULT_MAX_ANGULAR_SPEED
from .entities import Entity, Robot, Wall
from .geometry.types import RayHit, Vector
from .observation import NlosScan, Observation
from .sensors.los import LOS_FOV, LOS_RANGE, LOS_RAY_COUNT
from .sensors.nlos import (
    BIN_EDGES,
    NLOS_MAX_PATH,
    PRIMARY_FOV,
    PRIMARY_RAY_COUNT,
    SECONDARY_FAN,
    SECONDARY_RAY_COUNT,
)
from .world import World


def _array(values: Vector) -> list[float]:
    return [float(value) for value in values]


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


class RunRecorder:
    """Store static run metadata, observation snapshots, and post-step states."""

    def __init__(
        self,
        world: World,
        world_name: str,
        controller: Controller,
        controller_parameters: dict[str, object],
        fps: int,
        dt: float,
        sense_every: int,
        seed: int | None = None,
    ) -> None:
        self.entity_ids: dict[Entity, str] = {}
        self.robot_ids: dict[Robot, str] = {}
        entities = []
        counters = {"wall": 0, "robot": 0}

        for entity in world.entities:
            if isinstance(entity, Wall):
                kind = "wall"
                shape = {
                    "type": "box",
                    "width_m": float(entity.box.width),
                    "height_m": float(entity.box.height),
                }
            elif isinstance(entity, Robot):
                kind = "robot"
                shape = {"type": "circle", "radius_m": float(entity.radius)}
            else:
                raise TypeError(f"Cannot record entity type {type(entity).__name__}.")

            entity_id = f"{kind}-{counters[kind]:03d}"
            counters[kind] += 1
            self.entity_ids[entity] = entity_id
            if isinstance(entity, Robot):
                self.robot_ids[entity] = entity_id

            entity_record = {
                "id": entity_id,
                "kind": kind,
                "shape": shape,
                "initial_pose": _array(entity.pose),
                "material": {
                    "albedo": float(entity.material.albedo),
                    "color": entity.material.color,
                },
            }
            if isinstance(entity, Robot):
                entity_record["nlos_enabled"] = bool(entity.nlos)
            entities.append(entity_record)

        self.data = {
            "schema_version": 1,
            "run": {
                "run_id": uuid4().hex,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "world_name": world_name,
                "seed": seed,
                "fps": int(fps),
                "dt_seconds": float(dt),
                "sense_every_steps": int(sense_every),
                "dynamics": {
                    "max_speed_mps": float(world.dynamics.max_speed),
                    "min_turn_speed_mps": DEFAULT_MAX_ANGULAR_SPEED,
                },
                "controller": {
                    "class": type(controller).__name__,
                    "parameters": _json_value(controller_parameters),
                },
                "sensors": {
                    "los": {
                        "fov_rad": float(LOS_FOV),
                        "ray_count": LOS_RAY_COUNT,
                        "range_m": LOS_RANGE,
                    },
                    "nlos": {
                        "primary_fov_rad": float(PRIMARY_FOV),
                        "primary_ray_count": PRIMARY_RAY_COUNT,
                        "secondary_fan_rad": float(SECONDARY_FAN),
                        "secondary_ray_count": SECONDARY_RAY_COUNT,
                        "max_path_m": NLOS_MAX_PATH,
                        "histogram_bin_edges_m": _array(BIN_EDGES),
                    },
                },
            },
            "world": {"entities": entities},
            "observations": [],
            "frames": [],
        }

    def _serialize_hit(self, hit: RayHit | None) -> dict | None:
        if hit is None:
            return None
        return {
            "entity_id": (
                self.entity_ids.get(hit.entity)
                if hit.entity is not None
                else None
            ),
            "distance_m": float(hit.distance),
            "point": _array(hit.point),
            "normal": _array(hit.normal),
        }

    def _serialize_nlos(self, scan: NlosScan | None) -> dict | None:
        if scan is None:
            return None
        return {
            "primary_readings": [
                {"direction": _array(direction), "hit": self._serialize_hit(hit)}
                for direction, hit in scan.primary_readings
            ],
            "samples": [
                {
                    "relay_point": _array(sample.relay_point),
                    "hit_point": _array(sample.hit_point),
                    "path_length_m": float(sample.path_length),
                }
                for sample in scan.samples
            ],
            "histogram_counts": [int(value) for value in scan.counts],
        }

    def record_observation(
        self,
        before_step: int,
        observation_id: str,
        world: World,
        observations: dict[Robot, Observation],
        poses: dict[Robot, np.ndarray],
    ) -> None:
        robot_records = []
        for robot in world.robots:
            observation = observations[robot]
            robot_records.append(
                {
                    "robot_id": self.robot_ids[robot],
                    "pose": _array(poses[robot]),
                    "los": [
                        {"direction": _array(direction), "hit": self._serialize_hit(hit)}
                        for direction, hit in observation.los
                    ],
                    "nlos": self._serialize_nlos(observation.nlos),
                }
            )
        self.data["observations"].append(
            {
                "id": observation_id,
                "before_step": int(before_step),
                "robots": robot_records,
            }
        )

    def record_frame(
        self,
        step_index: int,
        time_seconds: float,
        world: World,
        actions: dict[Robot, Action],
        observation_id: str | None,
    ) -> None:
        self.data["frames"].append(
            {
                "step_index": int(step_index),
                "time_seconds": float(time_seconds),
                "observation_id": observation_id,
                "robots": [
                    {
                        "robot_id": self.robot_ids[robot],
                        "pose": _array(robot.pose),
                        "velocity_mps": _array(robot.velocity),
                        "requested_action": {
                            "vx_mps": float(actions[robot].vx),
                            "vy_mps": float(actions[robot].vy),
                        },
                    }
                    for robot in world.robots
                ],
            }
        )

    def save(self, output_dir: str | Path = "outputs/runs") -> Path:
        """Write one compact JSON file and return its path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = output_dir / f"run_{timestamp}_{self.data['run']['run_id'][:8]}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.data,
                file,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        return path
