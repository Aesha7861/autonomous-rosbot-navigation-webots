# Autonomous Robots – Webots Controllers

This submission contains one Python controller for each of the five maze environments. The Webots world files are not included; use the corresponding course/examiner-provided world.

## Directory layout

| Environment | Controller file                                           | Webots`controller` field |
| ----------- | --------------------------------------------------------- | -------------------------- |
| Maze 1      | `src/Maze 1/controllers/frontier/frontier.py`           | `frontier`               |
| Maze 2      | `src/Maze 2/controllers/my_controller/my_controller.py` | `my_controller`          |
| Maze 3      | `src/Maze 3/controllers/my_controller/my_controller.py` | `my_controller`          |
| Maze 4      | `src/Maze 4/controllers/my_controller/my_controller.py` | `my_controller`          |
| Maze 5      | `src/Maze 5/controllers/my_controller/my_controller.py` | `my_controller`          |

Each maze has its own tuned controller.

## Configuration

1. Install the external Python packages with the same Python interpreter used by Webots:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

   Webots provides the `controller` Python module; do not install a separate package named `controller`.
2. For the maze being tested, copy its controller folder into the `controllers` directory of the Webots project containing the corresponding world:

   - Maze 1: copy the `frontier` folder.
   - Mazes 2–5: copy the respective maze's `my_controller` folder.
3. Open the corresponding `.wbt` world in Webots. Select the RosBot node and set its `controller` field to the value shown in the table above.
4. The controller expects the original RosBot devices used in the supplied environments, including the four wheel motors, wheel encoders, `laser`, `camera rgb`, and `camera depth`. The robot and environment must not be modified, and Supervisor access is not used.

## Running

1. Reset the simulation to the initial state.
2. Press **Run** in Webots. Do not start the Python file directly from a terminal.
3. The robot starts autonomously, searches for and reaches the blue pillar, then plans and follows a path to the yellow pillar.
4. The run is complete when the console prints that the mission is complete and the controller state is `done`.

A pygame window may open to display the live occupancy map. The controller also writes `map.pgm` and `inflated_map.pgm` as diagnostic map outputs when it exits normally.

## Maze Videos

The demonstration videos for all maze environments are available on GitLab:

[https://git.oth-aw.de/9826/autonomous-robot-2026	](https://git.oth-aw.de/9826/autonomous-robot-2026)

## Dependencies

- Webots with Python controller support
- Python 3
- NumPy
- pygame, for the external map visualization

## Important note

The five controllers were tuned separately for their corresponding maze directories. Run each controller only with its matching maze world and start every recorded test from a clean simulation reset.
