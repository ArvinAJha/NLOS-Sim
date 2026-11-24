"""The world container with queries, collision resolution, and stepping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .dynamics import Action, Dynamics
from .entities import Entity, Robot, Wall
from .geometry.overlap import overlap
from .geometry.types import Contact, RayHit, Vector

class World:
    """Owns entities and coordinates queries, dynamics, and collisions."""

    def __init__(
        self,
        dynamics: Dynamics | None = None,
    ) -> None:

        self.entities: list[Entity] = []
        self.robots: list[Robot] = []
        self.walls: list[Wall] = []
        self.dynamics = dynamics if dynamics is not None else Dynamics()

    def add(self, entity: Entity) -> None:
        """Add an entity and track robots and walls for collision handling."""
        self.entities.append(entity)
        if isinstance(entity, Robot):
            self.robots.append(entity)
        elif isinstance(entity, Wall):
            self.walls.append(entity)

    def raycast(
        self,
        origin: Vector,
        direction: Vector,
        max_range: float,
        ignore: Entity | Iterable[Entity] | None = None,
    ) -> RayHit | None:
        """Return the nearest hit across all entities, optionally ignoring some."""
        ignored = _ignored_entities(ignore)
        nearest: RayHit | None = None
        for entity in self.entities:
            if entity in ignored:
                continue
            hit = entity.raycast(origin, direction, max_range)
            if hit is not None and (
                nearest is None or hit.distance < nearest.distance
            ):
                nearest = hit
        return nearest

    def overlaps(self, entity: Entity) -> list[tuple[Entity, Contact]]:
        """Return contacts between an entity and every other entity."""
        contacts: list[tuple[Entity, Contact]] = []
        for other in self.entities:
            if other is entity:
                continue
            contact = overlap(entity.shape, entity.pose, other.shape, other.pose)
            if contact is not None:
                contacts.append((other, contact))
        return contacts

    def step(
        self,
        dt: float,
        actions: Mapping[Robot, Action] | None = None,
    ) -> None:

        action_map = actions if actions is not None else {}

        for robot in self.robots:
            # latent intent (updated every SENSE_EVERY frames)
            action = action_map.get(robot, Action())        # actions sourced from controller -> simulation -> (on SENSE_EVERY) world -> dynamics

            # active intent (always updated)
            # TODO: apply conversion (controller output (as input) -> converter output)
            uni_action = Dynamics.convert_si_to_uni(action.vx, action.vy, robot.theta)
            # TODO: apply barrier (takes in v, w and adjusts them)

            self.dynamics.step(robot, uni_action, dt)
        
        self._resolve_collisions()  # defunct. replace with functions tracking collisions, not resolving them

    def _resolve_collisions(self) -> None:
        wall_collides = 0
        bot_collides = 0
        for robot in self.robots:
            for wall in self.walls:
                contact = overlap(
                    wall.shape, wall.pose, robot.shape, robot.pose
                )
                if contact is not None:
                    # self._resolve_against_static(robot, contact)
                    wall_collides += 1
        
        for index, first in enumerate(self.robots):
            for second in self.robots[index + 1 :]:
                contact = overlap(
                    first.shape, first.pose, second.shape, second.pose
                )
                if contact is not None:
                    # self._resolve_between_robots(first, second, contact)
                    bot_collides += 1

        if wall_collides > 0: print(f"[COLLISION] Wall: {wall_collides}")
        if bot_collides > 0: print(f"[COLLISION] Robot: {bot_collides}")

    def _resolve_against_static(self, robot: Robot, contact: Contact) -> None:
        """Push the robot out along a wall-to-robot normal."""
        _move_robot(robot, contact.normal * contact.penetration)
        normal_speed = float(np.dot(robot.velocity, contact.normal))
        if normal_speed < 0.0:
            robot.velocity = robot.velocity - (
                1.0 + self.restitution
            ) * normal_speed * contact.normal

    def _resolve_between_robots(
        self, first: Robot, second: Robot, contact: Contact
    ) -> None:
        """Separate two robots and remove their closing normal velocity."""
        half_penetration = contact.penetration / 2.0
        _move_robot(first, -contact.normal * half_penetration)
        _move_robot(second, contact.normal * half_penetration)

        relative_velocity = second.velocity - first.velocity
        normal_speed = float(np.dot(relative_velocity, contact.normal))
        if normal_speed < 0.0:
            impulse = (
                -(1.0 + self.restitution) * normal_speed / 2.0
            )
            first.velocity = first.velocity - impulse * contact.normal
            second.velocity = second.velocity + impulse * contact.normal


def _ignored_entities(
    ignore: Entity | Iterable[Entity] | None,
) -> set[Entity]:
    if ignore is None:
        return set()
    if isinstance(ignore, Entity):
        return {ignore}
    return set(ignore)


def _move_robot(robot: Robot, offset: Vector) -> None:
    x, y, theta = robot.pose
    robot.set_pose(x + offset[0], y + offset[1], theta)


# EXTENSION POINT: sensors can call World.raycast and read Material values.
# EXTENSION POINT: new dynamic entity types need handling in _resolve_collisions.
