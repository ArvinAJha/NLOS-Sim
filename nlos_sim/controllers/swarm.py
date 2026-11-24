"""Obstacle avoidance and flocking with pyswarming behaviors."""

from __future__ import annotations

import numpy as np
import pyswarming.behaviors as pb

from ..dynamics import Action, Dynamics
from ..entities import Robot, Wall
from ..observation import Observation
from .parent_controller import Controller


def to_3d(points):
    """pyswarming works in 3D, so add z = 0. Works for an empty list too."""
    points = np.array(points, dtype=float).reshape(-1, 2)
    return np.hstack([points, np.zeros((len(points), 1))])

class SwarmController(Controller):
    """Repel from sensed surfaces, optionally aggregate or seek a target."""

    def __init__(
        self,
        speed=2.0,
        repulsion_weight=1.0,
        repulsion_distance=0.6,
        aggregation_weight=0.0,
        target=None,
        target_weight=1.0,
        target_tolerance=0.2,
        goals=None,
        nlos_repulsion_weight=1.8,
        nlos_guidance_weight=1.4,
        nlos_guidance_distance=3.5,
        wall_lookahead=2.0,
        waypoints=None,
    ):
        self.speed = speed
        self.repulsion_weight = repulsion_weight
        self.repulsion_distance = repulsion_distance
        self.aggregation_weight = aggregation_weight
        self.target = target  # (x, y) or None
        self.target_weight = target_weight
        self.target_tolerance = target_tolerance
        self.goals = goals if goals is not None else {}
        self.nlos_repulsion_weight = nlos_repulsion_weight
        self.nlos_guidance_weight = nlos_guidance_weight
        self.nlos_guidance_distance = nlos_guidance_distance
        self.wall_lookahead = wall_lookahead
        self.waypoints = {
            robot: list(points)
            for robot, points in (waypoints or {}).items()
        }

    def get_action(self, robot: Robot, observation: Observation) -> Action:
        me = np.array([robot.x, robot.y], dtype=float)
        obstacles, neighbors = self.sort_points(observation)
        push = np.zeros(2, dtype=float)

        goal = self.goals.get(robot, self.target)
        route = self.waypoints.get(robot)
        if route:
            next_point = np.asarray(route[0], dtype=float)
            if np.linalg.norm(next_point - me) <= self.target_tolerance:
                route.pop(0)
            if route:
                goal = np.asarray(route[0], dtype=float)

        if goal is not None:
            goal = np.asarray(goal, dtype=float)
            goal_distance = float(np.linalg.norm(goal - me))
            if goal_distance > self.target_tolerance:
                target_push = pb.target(
                    to_3d([me])[0],
                    to_3d([goal])[0],
                )[:2]
                push += self.target_weight * target_push

        # LOS wall avoidance uses the ray normal, which already points toward the robot.
        wall_risk = 0.0
        for direction, hit in observation.los:
            if hit is None or not isinstance(hit.entity, Wall):
                continue
            clearance = max(0.0, hit.distance - robot.radius)
            if clearance >= self.repulsion_distance:
                continue

            normal = np.asarray(hit.normal, dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-9)
            distance_weight = (
                self.repulsion_distance - clearance
            ) / self.repulsion_distance
            magnitude = self.repulsion_weight * (
                distance_weight ** 2
            ) / max(clearance + 0.1, 0.1)
            push += normal * magnitude
            wall_risk = max(
                wall_risk,
                (self.wall_lookahead - clearance) / self.wall_lookahead,
            )

        if len(neighbors) > 0:
            neighbor_points = to_3d([point for point in neighbors[:, :2]])
            push += (
                self.aggregation_weight
                * pb.aggregation(to_3d([me])[0], neighbor_points)
            )[:2]

            push += 1.8 * pb.collision_avoidance(
                to_3d([me])[0], neighbor_points
            )[:2]

        # point of innovation here. what does nlos data do uniquely in a controller? 
        if observation.nlos is not None:
            nlos_push = self.nlos_push(robot, observation.nlos, goal)
            push += nlos_push

            # If a wall is close, evaluate nearby candidate headings instead of
            # allowing goal attraction to force the robot straight into the wall.
            if wall_risk > 0.15:
                push += self._wall_sliding_push(
                    robot,
                    observation,
                    goal,
                    nlos_push,
                )

        return self.to_action(push)

    def _wall_sliding_push(
        self,
        robot: Robot,
        observation: Observation,
        goal,
        nlos_push: np.ndarray,
    ) -> np.ndarray:
        """Choose a nearby heading that preserves goal progress while avoiding wall rays."""
        me = np.array([robot.x, robot.y], dtype=float)
        reference = np.zeros(2, dtype=float)
        if goal is not None:
            reference = np.asarray(goal, dtype=float) - me
        if np.linalg.norm(reference) <= 1e-9:
            reference = nlos_push
        if np.linalg.norm(reference) <= 1e-9:
            return np.zeros(2, dtype=float)
        reference /= float(np.linalg.norm(reference))

        base_angle = float(np.arctan2(reference[1], reference[0]))
        best_direction = reference
        best_score = -np.inf

        for degrees in np.arange(-150.0, 151.0, 15.0):
            angle = base_angle + np.radians(degrees)
            candidate = np.array([np.cos(angle), np.sin(angle)], dtype=float)
            score = 3.0 * float(np.dot(candidate, reference))

            if np.linalg.norm(nlos_push) > 1e-9:
                nlos_direction = nlos_push / float(np.linalg.norm(nlos_push))
                score += 0.7 * float(np.dot(candidate, nlos_direction))

            for ray_direction, hit in observation.los:
                if hit is None or not isinstance(hit.entity, Wall):
                    continue
                clearance = max(0.0, hit.distance - robot.radius)
                if clearance >= self.wall_lookahead:
                    continue
                risk = (
                    self.wall_lookahead - clearance
                ) / self.wall_lookahead
                heading_toward_wall = max(
                    0.0,
                    float(np.dot(candidate, np.asarray(ray_direction, dtype=float))),
                )
                score -= 5.0 * risk * heading_toward_wall

            if score > best_score:
                best_score = score
                best_direction = candidate

        return 1.8 * best_direction

    def nlos_push(self, robot: Robot, nlos, goal) -> np.ndarray:
        """Use existing relay and histogram data as goal-progress waypoints."""
        me = np.array([robot.x, robot.y], dtype=float)
        relay_points = []
        hidden_push = np.zeros(2, dtype=float)

        for sample in nlos.samples:
            relay_points.append(sample.relay_point)
            offset = me - sample.hit_point
            distance = float(np.linalg.norm(offset))
            if distance <= 1e-9:
                continue
            clearance = max(0.0, distance - robot.radius)
            if clearance < self.nlos_guidance_distance:
                path_scale = 1.0 / (
                    1.0 + max(0.0, sample.path_length - 4.0) / 4.0
                )
                distance_weight = (
                    self.nlos_guidance_distance - clearance
                ) / self.nlos_guidance_distance
                hidden_push += (
                    0.15
                    * distance_weight
                    * path_scale
                    * offset
                    / distance
                )

        for _, relay in nlos.primary_readings:
            if relay is not None:
                relay_points.append(relay.point)

        if goal is None or not relay_points:
            counts = np.asarray(nlos.counts, dtype=float)
            confidence = min(1.0, float(np.sum(counts)) / 5.0)
            confidence = max(confidence, 0.25)
            return hidden_push * confidence

        goal = np.asarray(goal, dtype=float)
        goal_direction = goal - me
        goal_distance = float(np.linalg.norm(goal_direction))
        if goal_distance <= 1e-9:
            return hidden_push

        goal_direction /= goal_distance

        counts = np.asarray(nlos.counts, dtype=float)
        confidence = min(1.0, float(np.sum(counts)) / 5.0)
        confidence = max(confidence, 0.25)

        best_relay = None
        best_score = -np.inf
        unique_relays = np.unique(np.asarray(relay_points), axis=0)
        for relay in unique_relays:
            relay_vector = relay - me
            relay_distance = float(np.linalg.norm(relay_vector))
            if relay_distance <= 1e-9:
                continue

            relay_direction = relay_vector / relay_distance
            alignment = float(np.dot(goal_direction, relay_direction))
            distance_to_goal = float(np.linalg.norm(goal - relay))

            # Short, goal-aligned relays are preferred as corridor corner waypoints.
            score = (
                3.0 * alignment
                - 0.08 * relay_distance
                - 0.20 * distance_to_goal
                + 0.20 * confidence
            )
            if score > best_score:
                best_score = score
                best_relay = relay

        if best_relay is None:
            return hidden_push

        relay_direction = best_relay - me
        relay_distance = float(np.linalg.norm(relay_direction))
        if relay_distance <= 1e-9:
            return hidden_push

        return (
            self.nlos_guidance_weight
            * confidence
            * relay_direction
            / relay_distance
        ) + hidden_push

    def sort_points(self, observation: Observation):
        """Split what the robot sees into wall-like points and other robots."""
        obstacles, neighbors = [], []
        for direction, hit in observation.los:
            if hit is None:
                continue
            # hit.entity stands in for an RGBD class label (robot vs wall).
            if isinstance(hit.entity, Robot):
                neighbors.append(hit.point)
            else:
                obstacles.append(hit.point)
        return to_3d(obstacles), to_3d(neighbors)

    def to_action(self, push):
        """Use only the direction of `push`; the speed is fixed."""
        length = np.hypot(push[0], push[1])
        if length < 1e-9:
            return Action()
        return Action(
            self.speed * push[0] / length,
            self.speed * push[1] / length,
        )

        Dynamics.convert()

