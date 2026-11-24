"""Controller interface: a controller sees a robot and its observation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..dynamics import Action
from ..entities import Robot
from ..observation import Observation


class Controller(ABC):
    """Base class for swappable controllers."""

    @abstractmethod
    def get_action(self, robot: Robot, observation: Observation) -> Action:
        """Return an Action. The controller never sees the World.

        observation.nlos may be None.
        """