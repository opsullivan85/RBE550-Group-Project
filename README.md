# README

## Code Organization
- `.devcontainer`: Docker related files.
- `src`: All the source code.
  - `dogmeetsworld.py`: The main script for running an end-to-end motion
    planning test in an environment.
  - `obstacles.py`: Contains classes and functions for generating an obstacle environment.
  - `path_vis.py`: Contains the `PathVisualizer` class PathVisualizer used to
    visualize a global, multi-segment path of the robot.
  - `search/`: Global path planning code for the quadruped trunk.
  - `example-*`: Example code of specific components, mainly for sub-system testing purposes.
  - `quadruped_drake/`: The main reference code used, which includes TOWR. The
    only modifications made to this reference code base are listed below.
    - `planners/towr_of_powr.py`: Mid-level trunk planner used to loop through
      the segments of the global path and start/end states to TOWR. This extends
      the originally existing functionality of the `TowrTrunkPlanner` class.
    - `towr/include/towr/terrain/Grid.h`: Contains the Grid class used to
      calculate terrain height and derivative values.

## Running our project

### Startup Docker Container

Be sure you have docker installed locally on your computer. Then you can start it as shown below, or by using the devcontainer functionality of VSCode.

First build the docker image:
``` shell
cd .devcontainer
docker compose build sim
```

Startup a docker container of the image:
``` shell
cd .devcontainer

# start up the container
docker compose up --detach

# open a shell in the container
docker exec -it rbe-550-sim-1 bash

# navigate to the directory mapped to the project directory
cd ws

# do things in the container...

# stop the container once done
docker compose stop
```

All other instructions are assumed to be in the docker container with, unless otherwise stated.

# Build TOWR
Move into the `quadruped_drake` directory from the `ws` directory.

``` shell
cd src/quadruped_drake/
```

Setup the build profile for conan (C++ package manager):

``` shell
conan profile detect --force
```

Install dependencies through Conan (this creates a `build` directory):

``` shell
conan install . --output-folder=build --build=missing
```

Configure the build:
``` shell
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="conan_toolchain.cmake" -DCMAKE_BUILD_TYPE=Release

```

Compile C\+\+ code (includes TOWR and custom LCM bindings for interface with drake):
```
make
```

# Run the main Python Script
In any webbrowser open the address `localhost:7000`. Nothing will appear yet,
but it will be used for environment visualization so be ready to refresh it.


From the `ws` directory, move into the `src` directory and run the `dogmeetsworld.py` python script:

``` shell
cd src
python3 dogmeetsworld.py
```

The script will initially show output related to TOWR performing optimization
computations. This will happen for several seconds until it stops with the
following text, at which point the webbrowser page should be refreshed.

``` shell
Full multi-trajectory plan complete!
INFO:drake:Meshcat listening for connections at http://localhost:7000
```

The rough terrain environment should appear with a visualization of the planned,
global trunk path, as well as the robot attempting to follow it. The robot will
initially collapse due to torque limits, followed by a visualization of the
planned trunk as a transparent black box, and foot placements as black balls.

Other python scripts in this directory can be run in a similar way.
