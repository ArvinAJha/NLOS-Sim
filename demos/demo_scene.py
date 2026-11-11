"""Save and show the static T-corridor scene."""

from pathlib import Path

import matplotlib.pyplot as plt

from nlos_sim.environments import make_t_corridor
from nlos_sim.viz.figures import make_scene_figure

OUTPUT_PATH = Path("outputs/scene.png")
OUTPUT_DPI = 150


def main() -> None:
    """Build, save, and show the scene figure."""
    world = make_t_corridor()
    figure = make_scene_figure(world)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
