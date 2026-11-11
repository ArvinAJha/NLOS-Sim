# NLOS-Sim: geometry and world foundation

This project is the first layer of a 2D multi-drone simulator. It provides
shapes, raycasting, overlap and collision handling, a world containing a robot in
a T-shaped corridor, and static matplotlib visualizations. There are deliberately
no sensors, controllers, motion demos, or animation yet.

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

## Run the demos

Run any of these from the project root:

```bash
python -m demos.demo_scene
python -m demos.demo_raycast
python -m demos.demo_overlap
```

Each demo creates a static matplotlib figure, saves it under `outputs/`, and
shows it. The output files are:

- `outputs/scene.png`
- `outputs/raycast.png`
- `outputs/overlap.png`

## Run the tests

```bash
pytest
```

The tests are headless; no demo is used to move the robot.

## Package layout

```text
nlos_sim/
  geometry/       shapes, low-level raycasts, overlap dispatch
  entities.py     Material, Entity, Wall, Robot
  dynamics.py     Action and kinematic Dynamics
  world.py        queries, collision resolution, step
  environments.py make_t_corridor()
  viz/            static draw helpers and figure builders
demos/            three static demo entry points
tests/            headless geometry and world tests
```

## How to extend

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

### Where a sensor plugs in

A future sensor can call `World.raycast(origin, direction, max_range,
ignore=robot)` and read the returned `RayHit` plus `Material.albedo` and
`Material.color`. Sensors can live in a `Robot` subclass or a future simulation
loop. The natural seam is marked in `nlos_sim/entities.py` and
`nlos_sim/world.py`.

### Where a controller plugs in

A controller produces a mapping of `Robot` to `Action`. Pass it to
`World.step(dt, actions)`; robots without an action hold still. The controller
does not need any new interface. This seam is marked in `World.step` in
`nlos_sim/world.py`.

## Notes

- Boxes are axis-aligned; pose `theta` is ignored for box geometry until rotated
  boxes are added.
- Touching shapes return a contact with zero penetration.
- The raycast demo uses a finite range, so some rays miss and are drawn faded
  gray.
- `# EXTENSION POINT:` comments mark the intended seams for sensors, controllers,
  new shapes, new entity types, and animation.
