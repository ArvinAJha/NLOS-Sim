"""Save and show static circle-box overlap cases."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlos_sim.entities import Robot, Wall
from nlos_sim.geometry.shapes import Box
from nlos_sim.geometry.types import Pose
from nlos_sim.viz.figures import make_overlap_figure

OUTPUT_PATH = Path("outputs/overlap.png")
OUTPUT_DPI = 150
HEADING_UP = np.pi / 2.0
DEMO_PANELS: tuple[tuple[str, Pose], ...] = (
    ("Separated", (-3.0, 0.0, HEADING_UP)),
    ("Touching", (-1.5, 0.0, HEADING_UP)),
    ("Penetrating", (-1.2, 0.0, HEADING_UP)),
    ("Center inside", (0.3, 0.2, HEADING_UP)),
)


def main() -> None:
    """Build, save, and show the overlap figure."""
    wall = Wall(Box(2.0, 2.0), (0.0, 0.0, 0.0))
    robot = Robot(0.5, (0.0, 0.0, 0.0))
    figure = make_overlap_figure(wall, robot, DEMO_PANELS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
