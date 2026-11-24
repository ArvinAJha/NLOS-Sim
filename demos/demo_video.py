"""Save the main simulation animation as a GIF."""

from pathlib import Path

from main import main as run_simulation

OUTPUT_PATH = Path("outputs/simulation.gif")


def main() -> None:
    """Run the main simulation and save its animation."""
    run_simulation(output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()