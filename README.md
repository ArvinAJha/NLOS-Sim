# NLOS-Sim: 2D multi-robot sensing, control, and animation

This project is a small 2D multi-robot simulator. It provides shapes,
raycasting, overlap and collision handling, a T-shaped corridor, line-of-sight
(LOS) and optional two-bounce non-line-of-sight (NLOS) sensing, and swappable
controllers. A live matplotlib animation shows the simulation as it runs.

## Coordinate frame and units

The world frame has **x pointing right**, **y pointing up**, and angles measured
**counter-clockwise from +x**. Units are SI: meters, seconds, and radians.

## Setup

Python 3.10 or newer is required. With conda:

```bash
conda create -n nlos-sim python=3.11
conda activate nlos-sim
pip install -r requirements.txt
```

The project was developed and verified in the `nlos-sim` conda environment.
`pyswarming` is pinned because the swarm controller depends on its function
signatures.

## Run

```bash
python main.py                  # live animation; saves a JSON run on close
python -m nlos_sim.viz.test_renderer "outputs/runs/<run-file>.json"
python -m demos.demo_nlos       # static NLOS scan + histogram, saved to outputs/nlos.png
python -m demos.demo_scene      # static scene, saved to outputs/scene.png
python -m demos.demo_raycast    # static raycast fans, saved to outputs/raycast.png
python -m demos.demo_overlap    # static overlap cases, saved to outputs/overlap.png
python -m demos.demo_video      # simulation animation, saved to outputs/simulation.gif
```

The animation runs for 10 seconds. `Simulation` senses and asks one shared
controller for each robot's action every 6 physics frames. On a sensing frame,
it observes all robots before any move, computes every action, then advances
the world. The action is reused for the following 5 steps. At 24 FPS each
physics step is `1/24` second. The animation displays the post-step state and
the latest scan for each robot, along with each robot's traveled path.

Select the controller instance in `main.py`; the same instance is used for all
robots in that run:

```python
controller = MockController()
# controller = GradientController()
# controller = SwarmController(target=(-3.0, 1.0), repulsion_weight=1.0)
sim = Simulation(world, controller)
```

NLOS histograms are shown for every NLOS-enabled robot in the live animation.
When the window closes, the run is saved as a compact JSON file under
`outputs/runs/`.

Replay a run or an inclusive-start/exclusive-end frame slice:

```bash
python -m nlos_sim.viz.test_renderer "outputs/runs/<run-file>.json"
python -m nlos_sim.viz.test_renderer "outputs/runs/<run-file>.json" --start-frame 24 --end-frame 96
```

Replace the placeholder with a run filename printed by `main.py`.

## Run the tests

```bash
python -m pytest -q
```

The tests are headless; no demo is used to move the robot.

## Package layout

```text
nlos_sim/
  geometry/       shapes, low-level raycasts, overlap dispatch
  entities.py     Material, Entity, Wall, Robot
  dynamics.py     Action, heading-follows-velocity kinematics
  world.py        queries, collision resolution, step
  environments.py make_t_corridor()
  observation.py  Observation, NlosScan, NlosSample
  sensors/        shared ray helpers, LOS fan, optional NLOS two-bounce scan
  controllers/    Controller interface, MockController, SwarmController
  simulation.py   observe -> decide -> world.step loop and recording hooks
  recording.py    compact run JSON serialization
  viz/            draw helpers, static figures, and test_renderer replay
demos/            static demo entry points
tests/            headless sensor, controller, dynamics, and world tests
```

## How to add a controller

Subclass `Controller` in `nlos_sim/controllers/controller.py` and implement
`get_action(robot, observation)`. Return an `Action`; the controller sees only
the robot and its observation, never the `World`.

```python
from nlos_sim.controllers.controller import Controller
from nlos_sim.dynamics import Action


class ForwardController(Controller):
    def get_action(self, robot, observation):
        return Action(1.0, 0.0)
```

Then import it in `main.py` and choose one instance for the whole run:

```python
controller = ForwardController()
sim = Simulation(world, controller)
```

## Built-in controller behavior

`MockController` is a small LOS-only baseline. It checks the middle LOS
reading, stops if that ray hits within `0.6` m, and otherwise requests a
`2.4` m/s velocity along the robot's current heading.

`GradientController` restores the earlier normal-based avoidance controller
using the current `Controller` and `Observation` interfaces. It uses LOS hits
only, weights robot hits by `1.0` and wall hits by `0.3`, applies inverse-square
repulsion inside a `0.6` m surface-clearance buffer, and caps requested speed at
`1.0` m/s. If no hit is within that buffer, it requests a slow forward speed of
`0.1` m/s. It does not seek a target or consume NLOS samples.

`SwarmController` combines sensed points into a 2D steering vector (the
`pyswarming` library expects 3D positions, so the controller appends `z = 0`):

- LOS hit points whose entity is a `Robot` are treated as neighbors; all other
  LOS hit points are treated as obstacles.
- Every NLOS sample's final `hit_point` is treated as an obstacle. Its relay
  point, path length, and histogram counts do not currently affect the
  controller. NLOS samples do not carry entity identity, so a hidden robot hit
  cannot currently be distinguished from a wall hit.
- `repulsion_weight` scales the clearance-based force from nearby obstacle
  and neighbor points.
- `repulsion_distance` is the minimum surface clearance (meters) at which
  swarm repulsion begins. The controller subtracts the controlled robot's
  radius from the distance to each observed point, and only points inside this
  buffer contribute. The repulsive force grows as clearance shrinks.
- `aggregation_weight` scales the library's average directional pull toward
  the neighbor points. It defaults to zero, so aggregation is off unless
  enabled.
- If `target` is set and farther away than `target_tolerance` (0.2 m by
  default), a normalized direction toward it is added, scaled by `target_weight`.
  This keeps target influence from growing with target distance and stops
  adding target attraction once the robot is within tolerance.
- The final vector is normalized to the fixed `speed` (2 m/s by default).
  A near-zero vector returns a zero-velocity `Action`.

`pyswarming` is still used for neighbor aggregation. Swarm repulsion is
clearance-based so points outside `repulsion_distance` do not affect that
component. Repeated nearby hit points still contribute repeatedly, so the
result can depend on sensor ray/sample density.

This controller reacts to the visible hit-point samples; it does not construct
or maintain an obstacle map, and it does not consume the NLOS histogram. LOS
neighbors are represented by their ray-hit surface points, not their robot
centers. These choices keep the example small but matter when tuning a swarm or
combining sensors with different ray/sample counts.

## Simulation data flow

`Simulation` owns a `World` and one controller instance. On a sensing frame it
creates an `Observation` for every robot (LOS is always collected; NLOS is
collected only when `robot.nlos` is true), then calls
`get_action(robot, observation)` once per robot. The resulting
`Robot -> Action` mapping is passed to `World.step`.

An `Observation` contains:

- `los`: one `(direction, hit)` pair per LOS ray; `hit` is the nearest
  `RayHit` or `None` for a miss.
- `nlos`: `None` when disabled, otherwise an `NlosScan` containing the
  collected `NlosSample` objects and fixed-bin path-length counts.

Each `NlosSample` has a primary relay point, a final secondary-hit point, and
the combined primary-plus-secondary path length in meters. The NLOS sensor
keeps every successful secondary-ray hit, but each individual ray still
returns only its nearest hit. Its histogram bins use fixed edges so counts
from different frames can be compared. The live display and run recorder keep a
separate histogram for every NLOS-enabled robot.

## Run Recording

`RunRecorder` writes one compact JSON file per live run. Static world entity
geometry, initial poses, materials, sensor/controller settings, and timing are
stored once. Observation records are written only on sensing frames and hold
each robot's observation-time pose, LOS readings, optional NLOS primary
readings/samples/histogram, and an observation ID. Each post-step frame stores
the resulting robot poses and velocities, requested actions, timestamp, and
the ID of the observation that drove the action. This allows the renderer to
recreate a displayed state without rerunning physics or controllers.

`nlos_sim/viz/test_renderer.py` loads the JSON and replays all frames or a frame
slice. It renders from recorded LOS/NLOS values, displays one histogram per
NLOS-enabled robot, and overlays the recorded requested action. It does not
call sensors, controllers, or `World.step`.

## How to turn on NLOS

NLOS is off by default. Pass `nlos=True` when creating a robot, or set the
attribute before creating the `Simulation`:

```python
world.add(Robot(radius=0.3, pose=(3.0, 1.0, 0.0), nlos=True))
world.robots[0].nlos = True
```

When on, `observe()` adds an `NlosScan` to that robot's `Observation`. The scan
contains primary readings, relay points, final hit points, path lengths, and a
fixed-bin histogram. `main.py` displays a histogram for every NLOS-enabled
robot. The standalone example is:

```bash
python -m demos.demo_nlos
```

## How to extend the geometry

### Add a shape

Subclass `Shape` in `nlos_sim/geometry/shapes.py` and implement
`raycast(origin, direction, max_range, pose)`. If the shape participates in
overlap queries, add a handler in `nlos_sim/geometry/overlap.py` and register it
with `@register_overlap(NewShape, OtherShape)`. Add a drawing helper only if the
new shape needs one.

### Add a robot subclass

Subclass `Robot` in `nlos_sim/entities.py`. `World.add` recognizes `Robot`
subclasses, so the new robot automatically participates in dynamics and
collision resolution. No changes to `World` or `Dynamics` are needed.

## Notes

- Boxes are axis-aligned; pose `theta` is ignored for box geometry until rotated
  boxes are added.
- Touching shapes return a contact with zero penetration.
- `Dynamics` turns the heading toward velocity when speed is above
  `MIN_TURN_SPEED`, so the LOS fan follows the direction of travel.
- `# EXTENSION POINT:` comments mark the remaining intended seams for sensors,
  controllers, new shapes, and new entity types.
