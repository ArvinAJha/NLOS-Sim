"""Tests for replaying serialized simulation frames."""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from nlos_sim.controllers.mock import MockController
from nlos_sim.entities import Robot
from nlos_sim.recording import RunRecorder
from nlos_sim.simulation import Simulation
from nlos_sim.sensors.los import LOS_RAY_COUNT
from nlos_sim.sensors.nlos import BIN_EDGES
from nlos_sim.viz.draw import NLOS_HISTOGRAM_Y_MAX, draw_histogram, draw_path
from nlos_sim.viz.test_renderer import create_animation, select_frames
from nlos_sim.world import World


def test_select_frames_uses_inclusive_start_and_exclusive_end():
    frames = [{"step_index": index} for index in range(8)]

    selected = select_frames(frames, 2, 5)

    assert [frame["step_index"] for frame in selected] == [2, 3, 4]
    with pytest.raises(ValueError):
        select_frames(frames, 7, 9)


def test_histograms_share_a_fixed_y_scale():
    counts = np.zeros(len(BIN_EDGES) - 1)
    figure, axes = plt.subplots(1, 2)
    try:
        draw_histogram(axes[0], counts)
        counts[0] = 12
        draw_histogram(axes[1], counts)

        assert axes[0].get_ylim() == axes[1].get_ylim()
        assert axes[0].get_ylim() == (0.0, float(NLOS_HISTOGRAM_Y_MAX))
    finally:
        plt.close(figure)


def test_draw_path_plots_robot_positions():
    figure, axis = plt.subplots()
    try:
        draw_path(axis, [np.array((0.0, 1.0)), np.array((2.0, 3.0))], "red")

        np.testing.assert_allclose(axis.lines[0].get_xdata(), [0.0, 2.0])
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [1.0, 3.0])
    finally:
        plt.close(figure)


def test_renderer_replays_saved_states_and_each_nlos_histogram():
    world = World()
    robots = [
        Robot(0.3, np.array((0.0, 0.0, 0.0)), nlos=True),
        Robot(0.3, np.array((3.0, 0.0, 0.0)), nlos=True),
        Robot(0.3, np.array((6.0, 0.0, 0.0))),
    ]
    for robot in robots:
        world.add(robot)
    recorder = RunRecorder(
        world,
        world_name="renderer-test",
        controller=MockController(),
        controller_parameters={"name": "mock"},
        fps=24,
        dt=1 / 24,
        sense_every=6,
    )
    simulation = Simulation(world, MockController(), recorder)
    for _ in range(7):
        simulation.step()

    figure, animation = create_animation(recorder.data, start_frame=1, end_frame=4)
    try:
        assert len(list(animation.new_frame_seq())) == 3
        for frame_index in range(3):
            animation._func(frame_index)
            figure.canvas.draw()
        assert len(figure.axes) == 3  # scene and one histogram for each NLOS robot
        assert len(figure.axes[0].lines) >= len(robots) * LOS_RAY_COUNT
        assert figure.axes[1].get_title().startswith("NLOS:")
        assert figure.axes[2].get_title().startswith("NLOS:")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
