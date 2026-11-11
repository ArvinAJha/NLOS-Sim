"""Save and show static raycast fans from three robot poses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlos_sim.environments import make_t_corridor
from nlos_sim.geometry.types import Pose
from nlos_sim.viz.figures import make_raycast_figure

OUTPUT_PATH = Path("outputs/raycast.png")
OUTPUT_DPI = 150
DEMO_PANELS: tuple[tuple[str, Pose], ...] = (
    ("In the stem", (0.0, -4.0, np.pi / 2.0)),
    ("At the junction", (0.0, 0.0, 0.0)),
    ("In the bar", (4.0, 1.0, 0.0)),
)


def main() -> None:
    """Build, save, and show the raycast figure."""
    world = make_t_corridor()
    figure = make_raycast_figure(world, DEMO_PANELS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
