# Codebase Memory

Read this before changing sensor, geometry, entity, controller, or simulation code. This is a technical contract guide; it is separate from the user-facing [README.md](README.md).

## Shared Types

Defined in `nlos_sim/geometry/types.py`:

- `Vector` and `Pose` are both aliases for `numpy.ndarray`. A `Pose` is a float array with shape `(3,)`: `[x, y, theta]`. `theta` is radians.
- `RayHit(distance, point, normal, entity)` contains distance in meters, a 2D world-space hit point, a 2D unit normal facing the ray origin, and the actual hit `Entity` (or `None` for a primitive-only shape query).
- `Contact(normal, penetration, point)` describes an overlap. The normal points from the first input shape toward the second.
- `Segment(start, end)` stores 2D endpoints as `Vector`s.

World coordinates use +x right, +y up, and counter-clockwise angles from +x. Use NumPy arrays for poses; tuples may work at runtime through `np.asarray`, but the declared `Pose` type is an array.

## Entities and Geometry

`nlos_sim/entities.py`:

- `Entity` combines a `Shape`, `Material`, and mutable pose. `set_pose(x, y, theta)` replaces the pose array. `x`, `y`, and `theta` return floats.
- `Wall` is an `Entity` backed by a `Box`.
- `Robot` is an `Entity` backed by a `Circle`; it has a 2D velocity array and `nlos: bool`. LOS is always sensed; `nlos` only enables the optional NLOS scan.
- `Material` currently supplies albedo/color metadata; ray intensity is not modeled.

`nlos_sim/geometry/shapes.py` implements `Shape.raycast(...)`, `Circle`, and `Box`. Circle pose uses x/y translation. Boxes are axis-aligned; theta is ignored. A box raycast checks each of its four edges and returns its nearest edge hit.

`nlos_sim/geometry/overlap.py` dispatches by shape-type pair. Circle-circle, box-circle, and circle-box are implemented. Box-box overlap raises `NotImplementedError`. `World.step` uses these contacts for robot-wall and robot-robot collision resolution.

## One Ray

The query path is:

```text
sensor chooses origin/direction/range
 -> World.raycast
 -> Entity.raycast for each eligible entity
 -> Shape.raycast
 -> ray_circle or ray_segment
 -> nearest RayHit (or None)
```

`ray_segment` and `ray_circle` normalize direction, so hit distance and max range are in meters. Zero direction raises `ValueError`. Parallel/collinear segment rays miss; segment endpoints count. Tangent circle hits count. Normals are oriented toward the origin.

`World.raycast(origin, direction, max_range, ignore=None)` visits all entities except those ignored and returns only the nearest hit for that ray. `ignore` accepts one entity, an iterable, or `None`. It does not return all intersections along a ray.

An entity-level hit gets its `entity` field attached by `Entity.raycast`. Sensor rays originate at the robot center: `sensors/rays_utils.py::position(robot)` returns `[robot.x, robot.y]`. LOS and NLOS primary rays ignore the casting robot itself.

## LOS Data

`nlos_sim/sensors/los.py::cast_los_scan(world, robot)` always runs when `simulation.observe` is called:

- 15 rays (`LOS_RAY_COUNT`, odd, so one is exactly centered on heading).
- 45-degree total FOV, centered on `robot.theta`.
- 6-meter maximum range.
- Returns `list[tuple[Vector, RayHit | None]]`; each pair is `(unit_direction, nearest_hit_or_None)`.

LOS hits preserve entity identity through `RayHit.entity`. A miss is `None`. The controller receives directions and hits, not the `World`.

## NLOS Data

`nlos_sim/sensors/nlos.py::cast_nlos_scan(world, robot)` is called only if `robot.nlos` is true. It returns an `NlosScan`.

1. It casts a separate primary fan: 5 rays over 8 degrees, centered on the current heading, from the robot center. This primary direct raycast produces the relay hit. It does **not** reuse or select from `Observation.los`; it is a separate, narrower fan, even though its first hit is geometrically direct/LOS to the relay.
2. For every primary hit, `secondary_fan` casts 15 rays across a 180-degree fan centered on the relay's surface-normal angle. The ray budget is `10 m - primary_distance`.
3. Secondary rays ignore both the robot and the relay entity, so they do not immediately re-hit their starting surface. Each secondary ray contributes at most its nearest hit. Misses are dropped; all successful secondary hits across all primary relays are kept.
4. Every primary ray is retained in `NlosScan.primary_readings` as `(direction, RayHit | None)`, including misses and relays that produce no secondary returns. Each successful secondary hit creates `NlosSample(relay_point, hit_point, path_length)`. `path_length` is the primary distance plus the secondary distance. The sample does not store hit entity, ray direction, or time.
5. `make_histogram(samples)` bins every sample's combined path length into fixed 0.5-meter bins from 0 to 10 meters. The counts describe one scan; they are not accumulated over time. Empty scans produce zero counts.

Thus the primary relay is included as the start of each sample's path length and is displayed as a relay point, but is not a standalone histogram entry. Histogram entries count successful secondary returns, not primary relays. `NlosSample` lacks entity identity, so consumers cannot tell whether its final hit was a wall or robot.

## Observation and Controller

`nlos_sim/observation.py` defines:

```python
Observation(
    los: list[tuple[Vector, RayHit | None]],
    nlos: NlosScan | None,
)
```

`nlos is None` means NLOS is disabled for that robot. `NlosScan` contains `primary_readings: list[tuple[Vector, RayHit | None]]`, `samples: list[NlosSample]`, and `counts: np.ndarray`.

A controller implements `Controller.get_action(robot, observation) -> Action` in `nlos_sim/controllers/controller.py`. It must not query `World`; all sensor information it may use must be in `Observation`.

- `MockController` uses the middle LOS hit to stop or move forward.
- `GradientController` uses nearby LOS hit normals to repel; it does not read NLOS.
- `SwarmController` classifies LOS hits of type `Robot` as neighbors and other LOS hits as obstacles. It treats every NLOS `hit_point` as an obstacle because NLOS has no entity identity. It currently uses NLOS samples, not histogram counts, to steer.

`Action(vx, vy)` is desired world-frame velocity in meters/second.

## Simulation and State Flow

`nlos_sim/simulation.py` owns a `World`, one shared `Controller`, current observations, and actions:

```text
frame % 6 == 0:
  observe every robot first (LOS; NLOS when enabled)
  call the shared controller once for each robot's Observation
  save returned Robot -> Action mapping
Every frame:
  World.step(1/24, actions)
  increment frame
```

All observations and observation-time poses are collected before any robot moves, avoiding robot iteration order affecting same-frame sensing. The controller returns one action per robot. `World.step` applies those actions and resolves collisions.

`Simulation.paths` stores each robot's initial 2D position and each post-step
position for live path visualization. It is not part of the run recording.

When a `RunRecorder` is attached, `Simulation` writes one observation entry at each sensing boundary, then one frame entry after every physics step. Each frame references the latest observation ID. Observation poses and sensor data are from before movement; frame poses and velocities are after the step and collision resolution.

`nlos_sim/dynamics.py::Dynamics.step` clamps velocity magnitude to its maximum, integrates x/y by `velocity * dt`, and points heading along velocity above `MIN_TURN_SPEED`. `nlos_sim/world.py::World.step` then resolves robot-wall and robot-robot overlap.

## Worlds, Main, and Visualization

`nlos_sim/environments.py`:

- `make_t_corridor(...)` builds eight walls and one default robot, facing +y from the stem.
- `make_maze(seed=None)` builds maze walls only; it adds no robots. Current maze constants describe a 13×13 grid with 8-meter cells.

`main.py` adds experiment-specific robots and selects a world with `main(maze: bool)`. `maze=True` calls `make_maze_like_world()` (20 unique-cell robots, all with NLOS enabled) and targets `MAZE_CENTER`. `maze=False` calls `make_world()` (the T corridor with three robots; robots 0 and 2 have NLOS enabled) and targets `JUNCTION_CENTER`. One `SwarmController` instance is shared by all robots in the run.

`Simulation.observations` is keyed by robot. In `main.py`, `draw_scene` draws each robot's latest observation from its saved observation-time pose. The histogram grid contains one panel per NLOS-enabled robot; counts are not combined.

## Run Data and Replay

`nlos_sim/recording.py::RunRecorder` writes a versioned, compact JSON document under `outputs/runs/`:

- `run`: run ID/time, world name, optional seed, FPS/timestep/sensing interval, shared controller class and parameters, sensor settings/histogram edges, and dynamics limits.
- `world.entities`: stable IDs, entity kind/shape/dimensions, initial pose, material, and each robot's NLOS flag.
- `observations`: one entry per sensing boundary. Each robot entry stores observation-time pose, LOS readings, optional NLOS primary readings/samples/histogram, and stable entity IDs for LOS hits.
- `frames`: one entry per completed physics step, including step index/time, reference to the latest observation ID, and each robot's post-step pose, actual velocity, and requested action.

`nlos_sim/viz/test_renderer.py` replays the JSON without invoking sensors, controllers, or physics. It reconstructs the static world, draws recorded observations/actions, shows one histogram per NLOS-enabled robot, and accepts a half-open frame-index slice `[start, end)`.

`nlos_sim/viz/draw.py` draws LOS fans, relay/secondary NLOS paths, and NLOS histograms. `nlos_sim/viz/figures.py` builds static scene, raycast, and overlap figures. The `demos/` modules run those static plots; they are separate from the live `main.py` simulation.

## Type and Extension Rules

- Keep `Pose` as a NumPy `(3,)` array and `Vector` as a NumPy 2D array; use explicit aliases in annotations where possible.
- Visualization helpers consume point/direction collections as `list[Vector]`. `draw_ray_fan` takes parallel lists of directions, hits, and colors; `_fan_directions` returns a list of 2D vectors. `set_equal_view` takes a list of 2D point vectors, not poses. Convert NumPy arrow endpoints to 2-float tuples at the Matplotlib `annotate` boundary.
- Preserve the observation-to-controller boundary: add sensor results to `Observation`, then teach the controller to consume them. A local `scans` list does not affect actions unless passed through `get_action`.
- Keep LOS and NLOS as separate observation fields. Do not append NLOS fans as extra top-level robot scans; per-robot association must remain explicit.
- If NLOS returns need classification, add identity/metadata to `NlosSample` at the sensor boundary; do not infer entity type from a point alone.
- If a new query needs every intersection along one ray, extend the ray/world API; current raycasts return only the nearest hit.
- Tests are in `tests/test_raycast.py`, `test_overlap.py`, `test_world.py`, `test_sensors.py`, `test_controllers.py`, `test_simulation.py`, `test_recording.py`, and `test_renderer.py`.
