from controller import Robot, Keyboard
import os
# Prefer X11/XWayland for pygame because Wayland SDL windows can appear transparent
# when Webots/OpenGL is rendering behind them. This is still safe on normal Linux
# desktops; if X11 is unavailable pygame will report the error and mapping continues.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
if "DISPLAY" in os.environ and "SDL_VIDEODRIVER" not in os.environ:
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
import math
import heapq
from collections import deque
import numpy as np

# Optional live map visualization. The controller still runs if pygame is not
# installed in the Webots Python environment.
# These SDL hints avoid transparent / unpainted pygame windows on some Linux
# Wayland/XWayland desktops used with Webots. They must be set before pygame is
# imported.
os.environ.setdefault("SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR", "0")
os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
try:
    import pygame
except Exception:
    pygame = None

class RosbotExplorer:
    def __init__(self):
        # 1. Initialize Webots Robot (Supervisor is not allowed)
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # 2. Map Constants

        # The previous 8m x 8m map covered only -4m to +4m around the
        # robot's initial position. The maze extends outside that area.
        self.MAP_SIZE = 12.0

        # Keep the same resolution so obstacle sizes, inflation distances,
        # path planning and green-region dimensions do not change.
        self.MAP_RES = 0.02

        # 12m / 0.02m = 600 cells.
        self.GRID_SIZE = int(round(self.MAP_SIZE / self.MAP_RES))

        # Keep the initial robot position at the center of the grid.
        self.ORIGIN_X = -self.MAP_SIZE / 2.0
        self.ORIGIN_Y = -self.MAP_SIZE / 2.0
        
        # Cell Labels
        self.UNKNOWN = -1
        self.FREE = 0
        self.OCC = 1
        
        # Log-Odds Constants
        self.P_OCC_TH = 0.65
        self.P_FREE_TH = 0.35
        self.L_OCC = self.p_to_logodds(0.9)
        self.L_FREE = self.p_to_logodds(0.35)
        self.L_PRIOR = self.p_to_logodds(0.5)
        self.L_MIN, self.L_MAX = -6.0, 6.0
        self.HIT_EPS = self.MAP_RES * 0.5
        # Lidar hits farther than this carve free space but do NOT mark walls:
        # tangential error grows with range and far hits paint thick smeared
        # black. Walls are drawn only from close, accurate readings; clear
        # areas stay clean so the planner always sees where to go.
        self.MAPPING_OCC_MAX_RANGE_M = 2.0

        # Confirmation thresholds for converting log-odds -> discrete display_state.
        # FREE should update quickly for planning, but OCC should be more conservative
        # to avoid spurious occupied speckles between sparse lidar rays.
        self.OCC_CONFIRM_TH = 2
        self.FREE_CONFIRM_TH = 1

        # Planning: costmap-style inflation
        # - Hard inflation: cells within this radius of an obstacle are forbidden
        # - Soft inflation: cells within this larger radius are allowed but get extra cost
        # Tune these if you're scraping walls or refusing corridors.
        # RosBot footprint is ~0.20 x 0.235 m: half-width ~0.10, and the body
        # sweeps ~0.15 m from center when turning in place. 0.11 m hard
        # inflation let A* run paths the body physically cannot follow around
        # corners — the direct cause of wall clipping in narrow passages.
        self.HARD_INFLATION_RADIUS_M = 0.13
        self.SOFT_INFLATION_RADIUS_M = 0.24
        self.HARD_INFLATION_RADIUS_CELLS = int(math.ceil(self.HARD_INFLATION_RADIUS_M / self.MAP_RES))
        self.SOFT_INFLATION_RADIUS_CELLS = int(math.ceil(self.SOFT_INFLATION_RADIUS_M / self.MAP_RES))
        self.SOFT_INFLATION_WEIGHT = 150.0
        # Backwards-compat: old code reads this
        self.INFLATION_RADIUS_CELLS = self.HARD_INFLATION_RADIUS_CELLS

        # PHYSICAL passability floor. The RosBot body+wheels span ~0.24m, so
        # its center can never pass closer than ~0.12m to a wall regardless of
        # what margin the planner uses. Escape planners may relax the normal
        # (comfort) inflation down to THIS value but never below it: an escape
        # at 2 cells (4cm) planned paths through gaps narrower than the robot
        # itself and wedged it between walls.
        self.MIN_PASSABLE_CLEARANCE_CELLS = int(math.ceil(0.12 / self.MAP_RES))

        # 3. Grid Data Structures
        self.grid = [[self.L_PRIOR for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.display_state = [[self.UNKNOWN for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.confirm_counters = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.last_updated_scan = [[-1 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        # Tracking variables
        self.scan_id = 0
        self.last_pose = {'x': None, 'y': None, 'yaw': None}
        self.MOVE_THRESHOLD = 0.02
        # Allow mapping updates while rotating in place.
        self.YAW_THRESHOLD = 0.04

        # Supervisor-free pose estimate used by mapping/planning.
        # Starts at the robot initial pose in the robot-local odometry frame.
        # All mapped objects/pillars are estimated in this same frame.
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.z_est = 0.0
        self.pose_initialized = False
        self.last_enc = None
        self.wheel_radius = 0.043
        self.odom_axle_track = 0.26
        self.wheel_sensors = {}
        self.inertial_unit = None
        self.compass = None

        # 4. Device Setup
        self.init_devices()
        self.current_goal = None      # (i, j) or None
        self.need_new_goal = True
        self.goal_reached_threshold = 1.0  # grid cells
        self.visited_goals = set()   # stores (i, j) grid goals we've already reached
        self.path = None
        self.path_index = 0
        self.path_goal = None   # (gi, gj) that current path targets
        self.failed_goal_counts = {}  # (i,j) -> consecutive planning failures
        self.frontier_failure_count = 0  # Track consecutive failures to find valid frontiers

        # Waypoint-following: look ahead along the path and drive to the farthest
        # collision-free cell. This reduces corner-cutting when turning near wall ends.
        # With MAP_RES=0.02, a lookahead of 1 makes the robot crawl cell-by-cell.
        self.WAYPOINT_LOOKAHEAD = 20
        self._hard_blocked_cache = None
        self._hard_blocked_cache_scan_id = None

        # Local safety (anti-stuck while turning near walls)
        # These are simple reactive checks using the lidar front sector.
        self.FRONT_SECTOR_HALF_ANGLE_RAD = 0.35  # ~20 deg
        self.SAFETY_STOP_DIST_M = 0.12          # if closer than this, back off
        self.SAFETY_SLOW_DIST_M = 0.25          # if closer than this, slow turning
        self.RECOVERY_BACKUP_SPEED = 1.5

        # Angular steering limits: leave these unchanged for corner safety.
        self.MAX_W = 4.5
        self.MAX_W_CLOSE = 2.0

        # Main driving-speed parameters.
        self.K_LINEAR = 14.0
        self.K_ANGULAR = 4.0

        # Normal straight-driving wheel speed.
        # Previous value was effectively 3.5 rad/s.
        self.CRUISE_SPEED = 6.0

        # Maximum baseline wheel speed.
        # Never exceed the actual Webots motor limit.
        self.MAX_V = min(12.0, float(getattr(self, "MOTOR_MAX_V", 10.0)))

        # Camera ROI for color detection/visualization (fraction of image kept, centered)
        self.CAMERA_ROI_FRAC = 0.5
        # ROI center position (0.0=top, 1.0=bottom). Shift down to see floor.
        self.CAMERA_ROI_CENTER_Y_FRAC = 0.6
        self.CAMERA_ROI_BORDER_PX = 3

        # --- Color Object Detection State ---
        self.initial_scan_done = False
        self.initial_scan_start_yaw = None
        self.initial_scan_accumulated = 0.0
        self.initial_scan_last_yaw = None
        
        self.blue_found = False
        self.blue_coords = None  # (world_x, world_y)
        self.blue_reached = False
        
        self.yellow_found = False
        # yellow_pillar_coords: estimated pillar position
        # yellow_coords: navigation target (a small standoff in front of the pillar)
        self.yellow_pillar_coords = None  # (world_x, world_y)
        self.yellow_coords = None         # (world_x, world_y)
        self.yellow_reached = False

        # --- Yellow lock-in smoothing (reduce run-to-run variance) ---
        # Only save yellow coordinates after several consistent detections.
        self.YELLOW_LOCK_SAMPLES = 7
        self.YELLOW_LOCK_MAX_SPREAD_M = 0.35
        self._yellow_lock_samples = deque(maxlen=self.YELLOW_LOCK_SAMPLES)
        self._yellow_lock_debug_step = 0
        
        # Track green floor regions that have been marked as obstacles
        self.marked_green_regions = set()  # stores (gi, gj) center cells of marked regions

        # While navigating to pillars, treat green as a poison zone and block a larger area.
        # Stored as coarse region keys to avoid re-marking the same area every frame.
        self.poisoned_green_regions = set()

        # Track the currently-triggered green region key to avoid re-trigger loops
        self.green_scan_pending_region_key = None

        # Cooldown to avoid repeatedly re-triggering green behaviors in a loop.
        # After a green scan completes (or green floor is marked), ignore green triggers
        # for a short period.
        self.GREEN_COOLDOWN_S = 12.0
        self._green_cooldown_until_time = 0.0
        
        # Z-axis elevation monitoring for green platform detection
        self.initial_z_position = None  # Will be set on first pose reading
        self.Z_ELEVATION_THRESHOLD = 0.02  # 2cm elevation difference triggers blocking (green platform is elevated)
        
        # --- Green Region Scanning State Machine ---
        # States: None (not scanning), 'approach', 'position', 'scan_left', 'scan_right', 'mark'
        self.green_scan_state = None
        self.green_scan_saved_mission_state = None  # Save mission state to resume after
        self.green_scan_saved_goal = None
        self.green_scan_start_yaw = None
        self.green_scan_left_yaw = None  # Leftmost extent where green was visible
        self.green_scan_right_yaw = None  # Rightmost extent where green was visible
        self.green_scan_center_yaw = None  # Center yaw towards green
        self.green_scan_distance = 0.50  # Distance when green is at bottom of ROI
        self.GREEN_SIZE_M = 0.5  # Known green region size (0.5m x 0.5m)
        self.GREEN_TRIGGER_COVERAGE = 0.05  # 5% coverage triggers scanning
        self.GREEN_CLOSE_COVERAGE = 0.10  # 10% coverage means close enough

        # When we detect the yellow pillar, navigate to a point slightly BEFORE it
        # (along the camera ray), so the goal is more likely to be FREE.
        # Increased to 0.50m to ensure goal is well in front of pillar, not past it
        self.YELLOW_NAV_STANDOFF_M = 0.10

        # When we detect the blue pillar, navigate to a point slightly BEFORE it
        # (towards the robot) so the goal isn't exactly on the pillar cell.
        self.BLUE_NAV_STANDOFF_M = 0.10
        
        # State machine: 'initial_scan', 'go_to_blue', 'explore', 'go_to_yellow', 'done'
        self.mission_state = 'initial_scan'
        
        # Camera field of view for coordinate estimation (radians)
        self.CAMERA_FOV_H = 1.047  # ~60 degrees, adjust if different

        # --- Debug visualization (does NOT affect mapping/planning data) ---
        # Draw the currently planned A* path on the MapDisplay.
        self.DEBUG_DRAW_PATH_OVERLAY = True
        self._overlay_last_cells = set()
        
        # Draw detected frontiers on the MapDisplay
        self.DEBUG_DRAW_FRONTIERS = True
        self._current_frontiers = []  # Store current frontier cells for visualization

        # Debug-only: visualize where green is detected on the map display.
        # This does NOT modify the occupancy grid / planning; it only draws overlay pixels.
        self.DEBUG_DRAW_GREEN_DETECTIONS = True
        self._debug_green_cells = set()

        # When green is detected, also mark those cells as OCC in the real map (for debugging).
        # This WILL affect planning because display_state is used by A*.
        self.GREEN_DETECTION_MARK_OCCUPIED = True
        self.forced_occupied_cells = set()

        # --- Depth-camera LOW floating-wall detection (Maze3) ---
        # Maze3 has two floating-wall types; the lidar scan plane (~0.18m) sorts
        # most of them out on its own (computed from each wall's actual axis-angle
        # rotation in worlds/Maze3.wbt): slabs with 27.5/36.5cm gaps sit fully
        # ABOVE the plane (invisible → driven under → correct), the 16.5cm-gap
        # slab crosses the plane (seen → avoided → correct). The one dangerous
        # case is the ~8.5cm-gap slab: it sits fully BELOW the plane — invisible
        # to lidar, too low to drive under → collision. The depth camera fills
        # exactly that hole: mark cells whose lowest depth-hit height falls in
        # the robot-body band as obstacles. ROBOT_CLEARANCE_HEIGHT=0.22 sits
        # between the too-low tier (8.5/16.5cm) and passable tier (27.5/36.5cm).
        self.FLOATING_WALL_ENABLED = True
        self.FLOATING_WALL_POLL_EVERY_N_STEPS = 2
        self._floating_wall_step_counter = 0
        self.DEPTH_OBSTACLE_MIN_DIST = 0.15
        # The Astra depth camera's minRange is 0.6m — it is BLIND closer than
        # that. With a 0.8m cap the detector only worked in a 0.6-0.8m shell,
        # crossed in a fraction of a second; planks approached at an angle were
        # never marked and the robot drove under them. 1.3m gives real reaction
        # distance. The accuracy loss at range is handled by the persistence
        # gate below (FLOATING_WALL_CONFIRM_POLLS), NOT by shrinking the range:
        # noise pixels scatter and rarely hit the same cell repeatedly, while a
        # real plank flags the same cells on every poll.
        # 0.9m: marks are PERMANENT, so accuracy beats reach — at 1.3m the
        # projection error plus pose drift smeared permanent black over free
        # corridors and killed paths mid-run. 0.6 (camera minRange) to 0.9
        # still locks every plank before the robot commits.
        self.DEPTH_OBSTACLE_MAX_DIST = 0.90
        self.FLOATING_WALL_CONFIRM_POLLS = 3
        self._low_wall_hit_counts = {}
        self.FLOATING_WALL_MIN_HEIGHT = 0.05  # below = floor reflections, ignore
        # Maze2 retune: this maze's floating planks hang with bottom edges at
        # ~0.20-0.25m (WallShort(18) z=0.45, WallTiny(3)/(6) z=0.5, tilted
        # planks near the blue pillar). The RosBot + lidar mast is ~0.23m tall,
        # so nothing here is safely passable underneath. 0.22 classified those
        # planks as "drive under" and left them free — the collision case.
        self.ROBOT_CLEARANCE_HEIGHT = 0.30
        self.FLOATING_WALL_CAMERA_HEIGHT_M = 0.17
        self.DEPTH_PIXEL_STRIDE = 2
        self.FLOATING_WALL_FORWARD_OFFSET_M = 0.08
        self.FLOATING_WALL_STOP_BEFORE_ROBOT_M = 0.10

        # Yellow approach: avoid stopping forever at a single snapped cell
        self._yellow_goal_blacklist = set()

        # Debug: motion/planning traces (throttled)
        self.DEBUG_MOTION = True
        self._debug_motion_step = 0
        self._debug_last_target_wp = None
        self._debug_last_best_index = None

        # --- Pygame live occupancy-map visualization ---
        # Shows only the occupancy map generated from lidar/odometry. This is
        # a visualization layer; it does not change mapping or planning.
        self.ENABLE_PYGAME_MAP = True

        # Updating every simulation step is unnecessarily expensive.
        # Four steps gives a sufficiently smooth live map without slowing Webots heavily.
        self.PYGAME_UPDATE_EVERY_N_STEPS = 4

        # Keep one map cell equal to one display pixel.
        # GRID_SIZE is 400, so the map becomes 400x400 instead of 600x600.
        self.PYGAME_CELL_SCALE = 1.0

        # Empty border around the map.
        self.PYGAME_MAP_MARGIN_PX = 10
        self._pygame_enabled = False
        self._pygame_screen = None
        self._pygame_font = None
        self._pygame_step_counter = 0
        self._pygame_last_error_printed = False


        # Open the pygame map viewer only after all attributes used by the
        # renderer exist. Opening it inside init_devices() caused a blank black
        # window because the first render happened before fields such as
        # initial_z_position, path, and pygame state were initialized.
        self.init_pygame_map()

    def init_devices(self):
        # Motors
        self.motors = [
            self.robot.getDevice('fl_wheel_joint'), self.robot.getDevice('fr_wheel_joint'),
            self.robot.getDevice('rl_wheel_joint'), self.robot.getDevice('rr_wheel_joint')
        ]
        for m in self.motors:
            m.setPosition(float('inf'))
            m.setVelocity(0.0)


        # Read the real velocity limit from the robot's motor configuration.
        # This avoids commanding velocities that the RosBot motor cannot accept.
        try:
            valid_limits = [
                float(m.getMaxVelocity())
                for m in self.motors
                if float(m.getMaxVelocity()) > 0.0
            ]
            self.MOTOR_MAX_V = min(valid_limits) if valid_limits else 10.0
        except Exception:
            self.MOTOR_MAX_V = 10.0

        print(f"[Motion] Motor maximum velocity: {self.MOTOR_MAX_V:.2f} rad/s")    

        # Wheel encoders for Supervisor-free odometry.
        # These names match the RosBot controller setup used in the main project.
        sensor_names = {
            'fl': ['front left wheel motor sensor', 'fl_wheel_joint_sensor', 'fl_wheel_sensor'],
            'fr': ['front right wheel motor sensor', 'fr_wheel_joint_sensor', 'fr_wheel_sensor'],
            'rl': ['rear left wheel motor sensor', 'rl_wheel_joint_sensor', 'rl_wheel_sensor'],
            'rr': ['rear right wheel motor sensor', 'rr_wheel_joint_sensor', 'rr_wheel_sensor'],
        }
        self.wheel_sensors = {}
        for key, names in sensor_names.items():
            for name in names:
                try:
                    dev = self.robot.getDevice(name)
                    if dev is not None:
                        dev.enable(self.timestep)
                        self.wheel_sensors[key] = dev
                        break
                except Exception:
                    continue
        if len(self.wheel_sensors) != 4:
            print(f"[Pose] WARNING: wheel encoders found {list(self.wheel_sensors.keys())}; odometry may not update correctly.")

        # IMU yaw source. If missing, yaw falls back to differential wheel odometry.
        self.inertial_unit = None
        for name in ['imu inertial_unit', 'inertial unit', 'inertial_unit']:
            try:
                dev = self.robot.getDevice(name)
                if dev is not None:
                    dev.enable(self.timestep)
                    self.inertial_unit = dev
                    break
            except Exception:
                continue

        self.compass = None
        for name in ['imu compass', 'compass']:
            try:
                dev = self.robot.getDevice(name)
                if dev is not None:
                    dev.enable(self.timestep)
                    self.compass = dev
                    break
            except Exception:
                continue

        # Lidar
        self.lidar = self.robot.getDevice("laser")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Camera (RGB + Depth) - additional devices
        self.camera_rgb = self.robot.getDevice("camera rgb")
        if self.camera_rgb:
            self.camera_rgb.enable(self.timestep)

        self.camera_depth = self.robot.getDevice("camera depth")
        if self.camera_depth:
            self.camera_depth.enable(self.timestep)

        # Display only. No Supervisor node / DEF lookup is used.
        self.display = self.robot.getDevice("MapDisplay")
        if self.display:
            self.display.setColor(0x888888)
            self.display.fillRectangle(0, 0, self.GRID_SIZE, self.GRID_SIZE)

        # Optional camera display (add a Display device named "CameraDisplay" in the robot)
        self.camera_display = self.robot.getDevice("CameraDisplay")

        # Pygame map viewer is opened at the end of __init__, after all state fields exist.

    # --- PYGAME LIVE MAP VISUALIZATION ---
    def init_pygame_map(self):
        """Open a pygame window for the live occupancy map only.

        This viewer intentionally uses direct pygame drawing instead of numpy
        frombuffer/surfarray conversion. Direct drawing is slower, but it is
        more reliable inside Webots on Ubuntu/Wayland/XWayland.
        """
        if not getattr(self, "ENABLE_PYGAME_MAP", True):
            return

        if pygame is None:
            print("[pygame map] pygame is not installed; live map window disabled.")
            return

        try:
            pygame.init()
            pygame.font.init()

            scale = max(
                0.1,
                float(getattr(self, "PYGAME_CELL_SCALE", 1.0))
            )

            # Only one status line remains above the map.
            self._pygame_header_h = 28

            margin = int(
                getattr(self, "PYGAME_MAP_MARGIN_PX", 10)
            )

            # Requested map-display size.
            requested_side = int(
                round(self.GRID_SIZE * scale)
            )

            # Determine available desktop resolution.
            try:
                display_info = pygame.display.Info()
                desktop_w = int(display_info.current_w)
                desktop_h = int(display_info.current_h)
            except Exception:
                desktop_w = requested_side + 100
                desktop_h = requested_side + 150

            if desktop_w <= 0:
                desktop_w = requested_side + 100

            if desktop_h <= 0:
                desktop_h = requested_side + 150

            # Leave space for desktop panels and window borders.
            max_window_w = max(320, desktop_w - 80)
            max_window_h = max(320, desktop_h - 120)

            max_map_w = max_window_w - (2 * margin)
            max_map_h = (
                max_window_h
                - self._pygame_header_h
                - (2 * margin)
            )

            # Use the requested map size if it fits.
            # Otherwise scale the whole map down without cutting it.
            side = max(
                200,
                min(
                    requested_side,
                    max_map_w,
                    max_map_h
                )
            )

            self._pygame_map_side_px = int(side)

            width = (
                self._pygame_map_side_px
                + (2 * margin)
            )

            height = (
                self._pygame_header_h
                + self._pygame_map_side_px
                + (2 * margin)
            )

            # This line actually creates the Pygame window.
            self._pygame_screen = pygame.display.set_mode(
                (width, height),
                0,
                32
            )

            pygame.display.set_caption(
                "RosBot live occupancy map"
            )

            self._pygame_font = pygame.font.SysFont(
                "Arial",
                14
            )

            # No large title is displayed anymore.
            self._pygame_big_font = None

            self._pygame_enabled = True
            self._pygame_last_error_printed = False

            print(
                f"[pygame map] opened "
                f"{width}x{height} live occupancy map window"
            )

            self.update_pygame_map(force=True)

        except Exception as exc:
            self._pygame_enabled = False
            self._pygame_screen = None
            print(f"[pygame map] disabled: {exc}")

    def close_pygame_map(self):
        """Close the pygame map window cleanly."""
        if pygame is None:
            return
        try:
            if getattr(self, "_pygame_enabled", False):
                pygame.quit()
        except Exception:
            pass
        self._pygame_enabled = False

    def _map_to_pygame_xy(self, i, j):
        """Convert map cell (i,j) to pygame pixel coordinates on the map surface."""
        return int(i), int(self.GRID_SIZE - 1 - j)

    def _draw_marker_on_surface(self, surface, cell, color, radius=3):
        if cell is None:
            return
        try:
            i, j = int(cell[0]), int(cell[1])
        except Exception:
            return
        if not self.inside_map(i, j):
            return
        x, y = self._map_to_pygame_xy(i, j)
        pygame.draw.circle(surface, color, (x, y), max(1, int(radius)))

    def _make_pygame_map_surface(self):
        """Build a 1-pixel-per-cell pygame Surface from display_state.

        Colors:
            gray  = UNKNOWN
            white = FREE
            black = OCC / forced obstacle
        Overlays:
            green   = current path
            red     = frontiers
            cyan    = blue marker
            yellow  = yellow marker
            magenta = robot pose and heading
        """
        surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), 0, 32)

        # First draw the actual occupancy grid. This is the live map generated
        # by lidar + odometry while the robot moves.
        unknown_col = surface.map_rgb((115, 115, 115))
        free_col = surface.map_rgb((245, 245, 245))
        occ_col = surface.map_rgb((0, 0, 0))

        try:
            px = pygame.PixelArray(surface)
            for j in range(self.GRID_SIZE):
                y = self.GRID_SIZE - 1 - j
                row = self.display_state[j]
                for i in range(self.GRID_SIZE):
                    st = row[i]
                    if st == self.FREE:
                        px[i, y] = free_col
                    elif st == self.OCC:
                        px[i, y] = occ_col
                    else:
                        px[i, y] = unknown_col
            del px
        except Exception:
            # Very conservative fallback. Slower but almost impossible to fail.
            surface.fill((115, 115, 115))
            for j in range(self.GRID_SIZE):
                y = self.GRID_SIZE - 1 - j
                row = self.display_state[j]
                for i in range(self.GRID_SIZE):
                    st = row[i]
                    if st == self.FREE:
                        surface.set_at((i, y), (245, 245, 245))
                    elif st == self.OCC:
                        surface.set_at((i, y), (0, 0, 0))

        # Overlay frontiers in red.
        try:
            for cell in getattr(self, "_current_frontiers", []) or []:
                i, j = int(cell[0]), int(cell[1])
                if self.inside_map(i, j):
                    x, y = self._map_to_pygame_xy(i, j)
                    surface.set_at((x, y), (255, 0, 0))
        except Exception:
            pass

        # Overlay current path in green.
        try:
            path = getattr(self, "path", None)
            path_index = int(getattr(self, "path_index", 0) or 0)
            if path:
                path_cells = []
                for k in range(max(0, path_index), len(path)):
                    i, j = int(path[k][0]), int(path[k][1])
                    if self.inside_map(i, j):
                        path_cells.append(self._map_to_pygame_xy(i, j))
                if len(path_cells) >= 2:
                    pygame.draw.lines(surface, (0, 210, 0), False, path_cells, 2)
                for x, y in path_cells[::max(1, len(path_cells)//80)]:
                    surface.set_at((x, y), (0, 255, 0))
                if path_index < len(path):
                    self._draw_marker_on_surface(surface, path[path_index], (0, 255, 0), radius=3)
        except Exception:
            pass

        # Pillar/nav markers.
        try:
            if getattr(self, "blue_coords", None) is not None:
                self._draw_marker_on_surface(
                    surface,
                    self.world_to_grid(self.blue_coords[0], self.blue_coords[1]),
                    (0, 200, 255),
                    radius=4,
                )
            if getattr(self, "yellow_coords", None) is not None:
                self._draw_marker_on_surface(
                    surface,
                    self.world_to_grid(self.yellow_coords[0], self.yellow_coords[1]),
                    (255, 190, 0),
                    radius=4,
                )
            if getattr(self, "yellow_pillar_coords", None) is not None:
                self._draw_marker_on_surface(
                    surface,
                    self.world_to_grid(self.yellow_pillar_coords[0], self.yellow_pillar_coords[1]),
                    (255, 255, 0),
                    radius=2,
                )
        except Exception:
            pass

        # Robot marker and heading.
        try:
            rx, ry, yaw, _ = self.get_pose()
            ri, rj = self.world_to_grid(rx, ry)
            if self.inside_map(ri, rj):
                x, y = self._map_to_pygame_xy(ri, rj)
                pygame.draw.circle(surface, (255, 0, 255), (x, y), 5)
                hx = int(round(x + 14 * math.cos(yaw)))
                hy = int(round(y - 14 * math.sin(yaw)))
                pygame.draw.line(surface, (255, 0, 255), (x, y), (hx, hy), 2)
        except Exception:
            pass

        return surface

    def _draw_pygame_label(self, text, x, y, big=False):
        if not self._pygame_screen:
            return
        font = getattr(self, "_pygame_big_font", None) if big else getattr(self, "_pygame_font", None)
        if font is None:
            return
        try:
            label = font.render(str(text), True, (245, 245, 245))
            self._pygame_screen.blit(label, (x, y))
        except Exception:
            pass

    def _pygame_map_status_text(self):
        try:
            state = np.asarray(self.display_state, dtype=np.int16)
            unknown = int(np.count_nonzero(state == self.UNKNOWN))
            free = int(np.count_nonzero(state == self.FREE))
            occ = int(np.count_nonzero(state == self.OCC))
            rx, ry, yaw, _ = self.get_pose()
            return f"free={free}  occ={occ}  unknown={unknown}  pose=({rx:.2f}, {ry:.2f}, {math.degrees(yaw):.0f} deg)"
        except Exception:
            return "map status unavailable"

    def update_pygame_map(self, force=False):
        """Refresh the live pygame occupancy-map window."""
        if not getattr(self, "_pygame_enabled", False):
            return
        if pygame is None or self._pygame_screen is None:
            return

        self._pygame_step_counter += 1
        update_every = max(1, int(getattr(self, "PYGAME_UPDATE_EVERY_N_STEPS", 1)))
        if not force and (self._pygame_step_counter % update_every) != 0:
            return

        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._pygame_enabled = False
                    return

            header_h = int(getattr(self, "_pygame_header_h", 28))
            margin = int(getattr(self, "PYGAME_MAP_MARGIN_PX", 10))

            screen_w, screen_h = self._pygame_screen.get_size()

            # Use the size selected when the window was created.
            requested_side = int(
                getattr(
                    self,
                    "_pygame_map_side_px",
                    min(screen_w, screen_h - header_h)
                )
            )

            # Recheck against the real window surface. This guarantees that
            # the complete map fits even if the window manager changes its size.
            available_w = screen_w - 2 * margin
            available_h = screen_h - header_h - 2 * margin

            side = max(
                1,
                min(requested_side, available_w, available_h)
            )

            # Center the complete map inside the Pygame window.
            map_x = (screen_w - side) // 2
            map_y = header_h + (screen_h - header_h - side) // 2

            # Clear the complete screen.
            self._pygame_screen.fill((20, 20, 20))

            # Generate the complete occupancy-map surface.
            map_surface = self._make_pygame_map_surface()

            # Scale the complete surface to the available square.
            # Both dimensions are scaled together, so no side is cut.
            if map_surface.get_size() != (side, side):
                map_surface = pygame.transform.scale(
                    map_surface,
                    (side, side)
                )

            # Draw the complete centered map.
            self._pygame_screen.blit(
                map_surface,
                (map_x, map_y)
            )

            # Draw the border around the complete map.
            pygame.draw.rect(
                self._pygame_screen,
                (220, 220, 220),
                pygame.Rect(map_x, map_y, side, side),
                1
            )

            # Keep only the free/occupied/unknown/pose status line.
            self._draw_pygame_label(
                self._pygame_map_status_text(),
                6,
                6
            )

            # flip() updates the full window and is more reliable than update()
            # for software surfaces under Webots.
            pygame.display.flip()
            pygame.event.pump()
        except Exception as exc:
            self._pygame_enabled = False
            if not getattr(self, "_pygame_last_error_printed", False):
                print(f"[pygame map] disabled after render error: {exc}")
                self._pygame_last_error_printed = True

    # --- MATH HELPERS ---
    def p_to_logodds(self, p): return math.log(p / (1.0 - p))
    def logodds_to_p(self, L): return 1.0 / (1.0 + math.exp(-L))
    
    def world_to_grid(self, x, y):
        i = int((x - self.ORIGIN_X) / self.MAP_RES)
        j = int((y - self.ORIGIN_Y) / self.MAP_RES)
        return i, j

    def grid_to_world(self, i, j):
        x = self.ORIGIN_X + i * self.MAP_RES
        y = self.ORIGIN_Y + j * self.MAP_RES
        return x, y

    def grid_to_world_center(self, i, j):
        x = self.ORIGIN_X + (i + 0.5) * self.MAP_RES
        y = self.ORIGIN_Y + (j + 0.5) * self.MAP_RES
        return x, y

    def inside_map(self, i, j):
        return 0 <= i < self.GRID_SIZE and 0 <= j < self.GRID_SIZE

    # --- SENSING & POSE ---
    def step(self):
        """Advance Webots and update the Supervisor-free odometry pose."""
        result = self.robot.step(self.timestep)
        if result != -1:
            self.update_pose()
        return result

    def _read_wheel_encoders(self):
        if not isinstance(self.wheel_sensors, dict) or len(self.wheel_sensors) != 4:
            return None
        try:
            return {
                'fl': float(self.wheel_sensors['fl'].getValue()),
                'fr': float(self.wheel_sensors['fr'].getValue()),
                'rl': float(self.wheel_sensors['rl'].getValue()),
                'rr': float(self.wheel_sensors['rr'].getValue()),
            }
        except Exception:
            return None

    def _read_imu_yaw(self):
        try:
            if self.inertial_unit is not None:
                _, _, yaw = self.inertial_unit.getRollPitchYaw()
                return float(yaw)
        except Exception:
            pass

        # Optional fallback if only a compass is available.
        try:
            if self.compass is not None:
                north = self.compass.getValues()
                if north is not None and len(north) >= 2:
                    return math.atan2(float(north[0]), float(north[1]))
        except Exception:
            pass

        return None

    def _read_imu_roll_pitch(self):
        """Return (roll, pitch) in rad from the IMU, or None if unavailable."""
        try:
            if self.inertial_unit is not None:
                roll, pitch, _ = self.inertial_unit.getRollPitchYaw()
                return float(roll), float(pitch)
        except Exception:
            pass
        return None

    def update_pose(self):
        """Update x/y/yaw without Webots Supervisor.

        Translation comes from wheel encoder deltas. Yaw comes from the inertial
        unit when available, otherwise from differential-drive encoder odometry.
        The pose starts at (0, 0, yaw0) in an odometry-local frame. That is enough
        for mapping because lidar hits, camera detections, and path planning all
        use the same local frame.
        """
        enc = self._read_wheel_encoders()
        yaw_meas = self._read_imu_yaw()

        if yaw_meas is not None and not self.pose_initialized:
            self.theta = yaw_meas

        if enc is None:
            if yaw_meas is not None:
                self.theta = yaw_meas
            return

        if not self.pose_initialized or self.last_enc is None:
            self.last_enc = enc
            self.pose_initialized = True
            if yaw_meas is not None:
                self.theta = yaw_meas
            print("[Pose] Supervisor-free odometry active (wheel encoders + optional IMU yaw).")
            return

        d_fl = (enc['fl'] - self.last_enc['fl']) * self.wheel_radius
        d_fr = (enc['fr'] - self.last_enc['fr']) * self.wheel_radius
        d_rl = (enc['rl'] - self.last_enc['rl']) * self.wheel_radius
        d_rr = (enc['rr'] - self.last_enc['rr']) * self.wheel_radius
        self.last_enc = enc

        dist_left = 0.5 * (d_fl + d_rl)
        dist_right = 0.5 * (d_fr + d_rr)
        dist = 0.5 * (dist_left + dist_right)

        # Suppress tiny encoder drift during near-pure rotation.
        if abs(dist_left + dist_right) < 0.0005:
            dist = 0.0

        # Anti-slip: if something is pressed right against the front yet the
        # encoders report FORWARD motion, the wheels are slipping — the robot
        # is not actually advancing. Integrating that phantom distance drags
        # the pose (and every lidar hit) through the wall and shifts the map.
        # Backward motion is kept so recovery back-ups still register.
        if dist > 0.0:
            min_front = self._min_lidar_distance_in_front()
            if min_front is not None and min_front < self.SAFETY_STOP_DIST_M:
                dist = 0.0

        if yaw_meas is None:
            self.theta += (dist_right - dist_left) / max(float(self.odom_axle_track), 1e-6)
            self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        else:
            self.theta = yaw_meas

        self.x += dist * math.cos(self.theta)
        self.y += dist * math.sin(self.theta)

    def get_pose(self):
        """Return the current Supervisor-free pose: x, y, yaw, z.

        z is not observable from the standard RosBot sensors used here, so it is
        fixed at 0.0. Any old green-platform logic that depended on Supervisor z
        height will therefore not trigger from elevation alone.
        """
        if self.initial_z_position is None:
            self.initial_z_position = 0.0
            print("Initial Z position recorded: 0.0000m (Supervisor-free odometry, no absolute z sensor)")

        return float(self.x), float(self.y), float(self.theta), float(self.z_est)

    # --- MAPPING LOGIC ---
    def update_cell(self, i, j, logodds_delta):
        if not self.inside_map(i, j):
            return

        # Never allow forced OCC debug cells to be flipped back.
        if (i, j) in getattr(self, 'forced_occupied_cells', set()):
            return

        # Temporal filtering MUST happen before we modify log-odds/counters.
        # Otherwise repeated updates within the same scan accumulate log-odds
        # without updating display_state, which can over-inflate obstacles.
        if self.last_updated_scan[j][i] == self.scan_id:
            return
        self.last_updated_scan[j][i] = self.scan_id

        self.grid[j][i] = max(min(self.grid[j][i] + logodds_delta, self.L_MAX), self.L_MIN)
        
        p = self.logodds_to_p(self.grid[j][i])
        
        if p >= self.P_OCC_TH:
            self.confirm_counters[j][i] = max(1, self.confirm_counters[j][i] + 1) if self.confirm_counters[j][i] >= 0 else 1
        elif p <= self.P_FREE_TH:
            self.confirm_counters[j][i] = min(-1, self.confirm_counters[j][i] - 1) if self.confirm_counters[j][i] <= 0 else -1
        else:
            self.confirm_counters[j][i] = 0

        # Update Display State
        new_state = self.UNKNOWN
        if self.confirm_counters[j][i] >= self.OCC_CONFIRM_TH:
            new_state = self.OCC
        elif self.confirm_counters[j][i] <= -self.FREE_CONFIRM_TH:
            new_state = self.FREE
        
        if new_state != self.display_state[j][i] and new_state != self.UNKNOWN:
            self.display_state[j][i] = new_state
            self.draw_pixel(i, j, new_state)

    def force_occupy_cell(self, i, j):
        """Immediately set a cell to OCC in the real map (grid + display_state)."""
        if not self.inside_map(i, j):
            return
        self.grid[j][i] = self.L_MAX
        self.confirm_counters[j][i] = int(self.OCC_CONFIRM_TH)
        if self.display_state[j][i] != self.OCC:
            self.display_state[j][i] = self.OCC
            self.draw_pixel(i, j, self.OCC)
        self.forced_occupied_cells.add((i, j))

    def draw_pixel(self, i, j, state):
        if not self.display: return
        col = 0x000000 if state == self.OCC else 0xFFFFFF
        self.display.setColor(col)
        self.display.drawPixel(i, self.GRID_SIZE - 1 - j)

    def _draw_overlay_pixel(self, i, j, color):
        if not self.display:
            return
        if not self.inside_map(i, j):
            return
        self.display.setColor(color)
        self.display.drawPixel(i, self.GRID_SIZE - 1 - j)

    def _redraw_base_pixel(self, i, j):
        """Redraw a pixel from the underlying map state (UNKNOWN/FREE/OCC)."""
        if not self.display:
            return
        if not self.inside_map(i, j):
            return

        st = self.display_state[j][i]
        if st == self.UNKNOWN:
            col = 0x888888
        elif st == self.OCC:
            col = 0x000000
        else:
            col = 0xFFFFFF

        self.display.setColor(col)
        self.display.drawPixel(i, self.GRID_SIZE - 1 - j)

    def _clear_path_overlay(self):
        if not self._overlay_last_cells:
            return
        for (i, j) in self._overlay_last_cells:
            self._redraw_base_pixel(i, j)
        self._overlay_last_cells.clear()

    def _draw_frontiers_overlay(self):
        """Draw detected frontier cells as red pixels on the map."""
        if not self.DEBUG_DRAW_FRONTIERS:
            return
        if not self.display:
            return
        if not self._current_frontiers:
            return
        
        # Draw each frontier cell in red
        for (i, j) in self._current_frontiers:
            self._draw_overlay_pixel(i, j, 0xFF0000)  # Red color
            self._overlay_last_cells.add((i, j))
    
    def _draw_color_markers(self):
        """Draw blue and yellow detected coordinates as larger markers on the map."""
        if not self.display:
            return
        
        # Draw BLUE marker (cyan color 0x00FFFF) as a 3x3 cross
        if self.blue_coords is not None:
            bi, bj = self.world_to_grid(self.blue_coords[0], self.blue_coords[1])
            # Draw a 3x3 cross pattern for better visibility
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if abs(di) + abs(dj) <= 2:  # Diamond shape
                        ni, nj = bi + di, bj + dj
                        if self.inside_map(ni, nj):
                            self._draw_overlay_pixel(ni, nj, 0x00FFFF)  # Cyan for blue marker
                            self._overlay_last_cells.add((ni, nj))
        
        # Draw YELLOW marker (orange color 0xFFA500) as a 3x3 cross
        if self.yellow_coords is not None:
            yi, yj = self.world_to_grid(self.yellow_coords[0], self.yellow_coords[1])
            # Draw a 3x3 cross pattern for better visibility
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if abs(di) + abs(dj) <= 2:  # Diamond shape
                        ni, nj = yi + di, yj + dj
                        if self.inside_map(ni, nj):
                            self._draw_overlay_pixel(ni, nj, 0xFFA500)  # Orange for yellow marker
                            self._overlay_last_cells.add((ni, nj))
        
        # Also draw the actual pillar position if available (as a smaller marker)
        if self.yellow_pillar_coords is not None:
            pi, pj = self.world_to_grid(self.yellow_pillar_coords[0], self.yellow_pillar_coords[1])
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = pi + di, pj + dj
                    if self.inside_map(ni, nj):
                        self._draw_overlay_pixel(ni, nj, 0xFFFF00)  # Pure yellow for pillar position
                        self._overlay_last_cells.add((ni, nj))

    def _draw_green_debug_overlay(self):
        """Draw green detections as black pixels (debug only)."""
        if not getattr(self, 'DEBUG_DRAW_GREEN_DETECTIONS', False):
            return
        if not self.display:
            return
        cells = getattr(self, '_debug_green_cells', None)
        if not cells:
            return

        for (i, j) in cells:
            if self.inside_map(i, j):
                self._draw_overlay_pixel(i, j, 0x000000)
                self._overlay_last_cells.add((i, j))

    def _debug_draw_path_overlay(self, start_cell, goal_cell, path, path_index):
        """Overlay the planned path (green), start (blue), goal (yellow)."""
        if not self.DEBUG_DRAW_PATH_OVERLAY:
            return
        if not self.display:
            return

        # Clear the previous overlay and redraw the new one.
        self._clear_path_overlay()
        
        # Draw frontiers first (so path can overlay on top)
        self._draw_frontiers_overlay()

        # Draw green detections (debug) as occupied-looking pixels
        self._draw_green_debug_overlay()
        
        # Draw blue and yellow detected coordinates
        self._draw_color_markers()

        # Draw remaining path from current index.
        if path:
            for k in range(max(0, int(path_index)), len(path)):
                i, j = path[k]
                self._draw_overlay_pixel(i, j, 0x00AA00)
                self._overlay_last_cells.add((i, j))

            # Highlight the next waypoint.
            if 0 <= int(path_index) < len(path):
                ni, nj = path[int(path_index)]
                self._draw_overlay_pixel(ni, nj, 0x00FF00)
                self._overlay_last_cells.add((ni, nj))

        # Draw start and goal markers on top.
        if start_cell is not None:
            si, sj = int(start_cell[0]), int(start_cell[1])
            self._draw_overlay_pixel(si, sj, 0x0000FF)
            self._overlay_last_cells.add((si, sj))

        if goal_cell is not None:
            gi, gj = int(goal_cell[0]), int(goal_cell[1])
            self._draw_overlay_pixel(gi, gj, 0xFFFF00)
            self._overlay_last_cells.add((gi, gj))

        # Keep the external pygame viewer in sync with path/goal overlays.
        # self.update_pygame_map()

    def debug_record_green_detection(self, detection_info, radius_m=0.10):
        """Record green detections for display on the map (debug only).

        This does NOT set map OCC; it only draws overlay pixels so you can see
        what the camera thinks is green.
        """
        if not getattr(self, 'DEBUG_DRAW_GREEN_DETECTIONS', False):
            return
        green_det = self._get_detection_for_color(detection_info, 'green')
        if green_det is None:
            return
        if green_det.get('coverage', 0.0) < 0.02:
            return  # too small to be a trustworthy hazard reading

        # Camera-only depth at the EXACT centroid pixel — no lidar, no window-min,
        # no fabricated fallback. Lidar is structurally blind to these flush floor
        # hazards (its scan plane passes over them and reports the wall BEYOND),
        # the shared estimator's min-of-window clips nearer wall edges, and its
        # depth=1.0m fallback fabricates positions when no reading exists — all
        # three were confirmed sources of green marks appearing where no green is.
        # Range-capped: far estimates scale bearing error into position error.
        cx_g, cy_g = green_det['centroid_px']
        d_g = self._get_depth_at(cx_g, cy_g, window_px=0)
        if d_g is None or d_g < 0.20 or d_g > 1.2:
            return
        rx_g, ry_g, ryaw_g, _ = self.get_pose()
        img_w_g = green_det['img_width']
        cxn_g = (cx_g - img_w_g / 2.0) / (img_w_g / 2.0)
        ang_g = ryaw_g + (-cxn_g * (self._camera_fov_h(img_w_g, green_det['img_height']) / 2.0))
        coords = (rx_g + d_g * math.cos(ang_g), ry_g + d_g * math.sin(ang_g))

        ci, cj = self.world_to_grid(coords[0], coords[1])
        r_cells = int(max(0, float(radius_m) / float(self.MAP_RES)))
        r2 = r_cells * r_cells

        # OCC-mark each coarse 16cm region only ONCE. This method runs every
        # frame while green is visible; without dedup the per-frame estimate
        # jitter unions hundreds of permanent 0.10m disks into a huge black
        # blob that swallows the corridor next to the patch. A few one-shot
        # disks still cover the real 0.5m patch for planning.
        if not hasattr(self, '_green_disk_regions'):
            self._green_disk_regions = set()
        region_key = (ci // 8, cj // 8)
        mark_occ = (getattr(self, 'GREEN_DETECTION_MARK_OCCUPIED', False)
                    and region_key not in self._green_disk_regions)
        if mark_occ:
            self._green_disk_regions.add(region_key)

        if r_cells <= 0:
            if self.inside_map(ci, cj):
                self._debug_green_cells.add((ci, cj))
                if mark_occ:
                    self.force_occupy_cell(ci, cj)
            return

        for dj in range(-r_cells, r_cells + 1):
            for di in range(-r_cells, r_cells + 1):
                if di * di + dj * dj > r2:
                    continue
                ni, nj = ci + di, cj + dj
                if self.inside_map(ni, nj):
                    self._debug_green_cells.add((ni, nj))
                    if mark_occ:
                        self.force_occupy_cell(ni, nj)

    def _get_rgb_image(self):
        if not self.camera_rgb:
            return None
        img = self.camera_rgb.getImage()
        if img is None:
            return None
        w = self.camera_rgb.getWidth()
        h = self.camera_rgb.getHeight()
        arr = np.frombuffer(img, dtype=np.uint8)
        if arr.size != w * h * 4:
            return None
        bgra = arr.reshape((h, w, 4))
        rgb = bgra[:, :, [2, 1, 0]]
        return rgb

    def _rgb_to_hsv(self, rgb):
        """Convert uint8 RGB image to HSV (H in degrees 0-360, S/V in 0-1)."""
        rgb_f = rgb.astype(np.float32) / 255.0
        r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        h = np.zeros_like(cmax)
        mask = delta > 1e-6
        r_eq = (cmax == r) & mask
        g_eq = (cmax == g) & mask
        b_eq = (cmax == b) & mask

        h[r_eq] = (60 * ((g[r_eq] - b[r_eq]) / delta[r_eq]) + 360) % 360
        h[g_eq] = (60 * ((b[g_eq] - r[g_eq]) / delta[g_eq]) + 120) % 360
        h[b_eq] = (60 * ((r[b_eq] - g[b_eq]) / delta[b_eq]) + 240) % 360

        s = np.zeros_like(cmax)
        s[cmax > 1e-6] = delta[cmax > 1e-6] / cmax[cmax > 1e-6]
        v = cmax
        return h, s, v

    def _mask_color(self, h, s, v, color):
        if color == "yellow":
            return (h >= 40) & (h <= 70) & (s >= 0.4) & (v >= 0.4)
        if color == "blue":
            return (h >= 200) & (h <= 250) & (s >= 0.4) & (v >= 0.3)
        if color == "green":
            # Looser S/V than the pillar colors: green is a FLAT floor patch seen
            # at a glancing angle, so its pixels mix with the gray floor and
            # desaturate well below 0.4 at range — that made real green invisible
            # until the robot was almost on top of it. Speckle false positives
            # from the looser thresholds are removed by the erosion filter in
            # process_camera(), not by clamping S/V here.
            return (h >= 80) & (h <= 150) & (s >= 0.30) & (v >= 0.25)
        if color == "red":
            return ((h <= 20) | (h >= 340)) & (s >= 0.4) & (v >= 0.3)
        return None

    def _get_depth_at(self, px, py, window_px=2):
        """Return a robust depth estimate near (px, py) in RGB pixel coords.

        Depth cameras frequently return the background for thin objects (pillars),
        which makes the estimated object position look "too far". To reduce that,
        we sample a small window and return the minimum valid depth.
        """
        if not self.camera_depth:
            return None

        w = self.camera_depth.getWidth()
        h = self.camera_depth.getHeight()
        if w <= 0 or h <= 0:
            return None

        # Map RGB pixel -> depth pixel index space.
        if self.camera_rgb:
            rgb_w = self.camera_rgb.getWidth()
            rgb_h = self.camera_rgb.getHeight()
            if rgb_w > 0 and rgb_h > 0:
                px = px * (w / float(rgb_w))
                py = py * (h / float(rgb_h))

        px_i = int(round(px))
        py_i = int(round(py))
        px_i = int(max(0, min(w - 1, px_i)))
        py_i = int(max(0, min(h - 1, py_i)))

        ranges = self.camera_depth.getRangeImage()
        if not ranges:
            return None

        rmin = self.camera_depth.getMinRange()
        rmax = self.camera_depth.getMaxRange()
        win = int(max(0, window_px))

        best = None
        for dy in range(-win, win + 1):
            yy = py_i + dy
            if yy < 0 or yy >= h:
                continue
            base = yy * w
            for dx in range(-win, win + 1):
                xx = px_i + dx
                if xx < 0 or xx >= w:
                    continue
                d = ranges[base + xx]
                if d <= rmin or d >= rmax:
                    continue
                if best is None or d < best:
                    best = d

        return best

    def _camera_fov_h(self, img_w, img_h):
        """Compute horizontal FOV from the Webots camera vertical FOV when possible."""
        try:
            if self.camera_rgb:
                fov_v = float(self.camera_rgb.getFov())
                if fov_v > 0 and img_w > 0 and img_h > 0:
                    return 2.0 * math.atan(math.tan(fov_v / 2.0) * (float(img_w) / float(img_h)))
        except Exception:
            pass
        return float(getattr(self, "CAMERA_FOV_H", 1.047))

    def _get_lidar_range_at_angle(self, angle_rad, window_rays=2):
        """Return a robust lidar range at a given bearing (rad), positive=left.

        Uses the same angle convention as run_mapping(): angle = fov/2 - k*step.
        Returns the minimum valid range within a small ray window.
        """
        if not self.lidar:
            return None
        ranges = self.lidar.getRangeImage()
        if not ranges:
            return None
        fov = float(self.lidar.getFov())
        n = int(self.lidar.getHorizontalResolution())
        if n <= 1 or fov <= 0:
            return None

        angle_step = fov / (n - 1)
        # Solve for k: angle = fov/2 - k*step
        k_float = (fov / 2.0 - float(angle_rad)) / angle_step
        k0 = int(round(k_float))
        k0 = max(0, min(n - 1, k0))

        rmin = self.lidar.getMinRange()
        rmax = self.lidar.getMaxRange()
        win = int(max(0, window_rays))

        best = None
        for k in range(max(0, k0 - win), min(n - 1, k0 + win) + 1):
            d = ranges[k]
            if d <= rmin or d >= rmax:
                continue
            if best is None or d < best:
                best = d

        return best

    def update_camera_display(self):
        """Draw the RGB camera feed to an optional display named CameraDisplay."""
        if not self.camera_display or not self.camera_rgb:
            return
        img = self.camera_rgb.getImage()
        if img is None:
            return
        w = self.camera_rgb.getWidth()
        h = self.camera_rgb.getHeight()
        handle = self.camera_display.imageNew(img, self.camera_display.BGRA, w, h)
        self.camera_display.imagePaste(handle, 0, 0, False)

        # Draw ROI rectangle overlay (matches process_camera ROI).
        roi_frac = getattr(self, "CAMERA_ROI_FRAC", 0.5)
        side = max(1, int(min(w, h) * roi_frac))
        cy_frac = float(getattr(self, "CAMERA_ROI_CENTER_Y_FRAC", 0.5))
        cy_frac = max(0.0, min(1.0, cy_frac))
        cy = int(cy_frac * h)
        y0 = max(0, min(h - side, cy - (side // 2)))
        x0 = max(0, (w - side) // 2)
        self.camera_display.setColor(0x00FF00)

        # Thicker outline so it's easy to see what ROI is analyzed.
        border = int(getattr(self, "CAMERA_ROI_BORDER_PX", 3))
        border = max(1, min(border, 10))
        for t in range(border):
            xx = max(0, x0 - t)
            yy = max(0, y0 - t)
            ww = min(w - xx, side + 2 * t)
            hh = min(h - yy, side + 2 * t)
            if ww > 0 and hh > 0:
                self.camera_display.drawRectangle(xx, yy, ww, hh)

        self.camera_display.imageDelete(handle)

    def process_camera(self):
        """Detect start (yellow), goal (blue), and blocked zones (red/green).
        Returns dict with 'color', 'coverage', 'centroid_px' or None."""
        # Lazy-init debug fields so this can be safely called from the main loop.
        if not hasattr(self, "_cam_debug_step"):
            self._cam_debug_step = 0
        if not hasattr(self, "DEBUG_CAMERA"):
            self.DEBUG_CAMERA = False

        rgb = self._get_rgb_image()
        if rgb is None:
            return None
        h, s, v = self._rgb_to_hsv(rgb)

        # Use a reduced central ROI for detection to avoid edges/noise.
        orig_img_h, orig_img_w = h.shape
        roi_frac = getattr(self, "CAMERA_ROI_FRAC", 0.5)
        side = max(1, int(min(orig_img_w, orig_img_h) * roi_frac))
        cy_frac = float(getattr(self, "CAMERA_ROI_CENTER_Y_FRAC", 0.5))
        cy_frac = max(0.0, min(1.0, cy_frac))
        cy = int(cy_frac * orig_img_h)
        y0 = max(0, min(orig_img_h - side, cy - (side // 2)))
        x0 = max(0, (orig_img_w - side) // 2)
        y1 = min(orig_img_h, y0 + side)
        x1 = min(orig_img_w, x0 + side)

        h_roi = h[y0:y1, x0:x1]
        s_roi = s[y0:y1, x0:x1]
        v_roi = v[y0:y1, x0:x1]

        roi_h, roi_w = h_roi.shape
        img_area = float(max(1, roi_h * roi_w))

        # Throttled debug stats (every 10 frames)
        self._cam_debug_step += 1
        debug_now = self.DEBUG_CAMERA and (self._cam_debug_step % 10 == 0)

        color_masks = {
            "yellow": self._mask_color(h_roi, s_roi, v_roi, "yellow"),
            "blue": self._mask_color(h_roi, s_roi, v_roi, "blue"),
            "red": self._mask_color(h_roi, s_roi, v_roi, "red"),
        }

        # Decide which color dominates the image (if any) and print it.
        coverage = {}
        centroids = {}
        for name, mask in color_masks.items():
            if mask is None:
                continue
            count = np.count_nonzero(mask)
            coverage[name] = float(count) / img_area
            if count > 0:
                # Centroid of the LARGEST contiguous band of the mask, not of
                # all pixels. A pillar half-hidden behind a floating plank
                # produces a DISJOINT mask (blue above and below the wood); the
                # global centroid then lands ON the plank, and reading depth at
                # that pixel records the pillar at the plank's position — a
                # wrong navigation goal. Select the biggest contiguous row band
                # first (horizontal occluders), then the biggest column band
                # within it (side-by-side blobs).
                sel = mask
                row_counts = sel.sum(axis=1)
                rows = np.where(row_counts > 0)[0]
                rsplits = np.where(np.diff(rows) > 1)[0] + 1
                rbest = max(np.split(rows, rsplits), key=lambda g: int(row_counts[g].sum()))
                sel = np.zeros_like(mask)
                sel[rbest, :] = mask[rbest, :]

                col_counts = sel.sum(axis=0)
                cols = np.where(col_counts > 0)[0]
                csplits = np.where(np.diff(cols) > 1)[0] + 1
                cbest = max(np.split(cols, csplits), key=lambda g: int(col_counts[g].sum()))
                sel2 = np.zeros_like(sel)
                sel2[:, cbest] = sel[:, cbest]

                ys, xs = np.where(sel2)
                # Convert to full image coordinates
                centroids[name] = (x0 + float(np.mean(xs)), y0 + float(np.mean(ys)))

        # GREEN uses its own detection region: full-width BOTTOM band of the image
        # (40% height to the bottom edge) instead of the centered ROI. Green is a
        # FLOOR hazard: as the robot approaches, it slides DOWN in the image and
        # out of the centered ROI's bottom edge — becoming invisible at exactly
        # the moment the robot is about to drive onto it (confirmed root cause of
        # green crossings in testing). Pillars keep the centered ROI above.
        gy0 = int(0.40 * orig_img_h)
        green_mask = self._mask_color(h[gy0:, :], s[gy0:, :], v[gy0:, :], "green")
        if green_mask is not None:
            # 1-px erosion (4-neighbor): antialiased single-pixel speckles at
            # wall/floor edges otherwise count as green and can clear the 0.1%
            # MIN_FRAC gate. A real patch only loses its 1-px rim.
            gm = green_mask
            er = gm.copy()
            er[1:, :] &= gm[:-1, :]
            er[:-1, :] &= gm[1:, :]
            er[:, 1:] &= gm[:, :-1]
            er[:, :-1] &= gm[:, 1:]
            g_count = int(np.count_nonzero(er))
            coverage["green"] = float(g_count) / float(max(1, er.size))
            if g_count > 0:
                # Centroid of the LARGEST contiguous patch, not of all green
                # pixels: with two patches in view, a global centroid lands in
                # the empty gap BETWEEN them and green gets marked at a spot
                # that is not green at all.
                col_counts = er.sum(axis=0)
                cols = np.where(col_counts > 0)[0]
                splits = np.where(np.diff(cols) > 1)[0] + 1
                groups = np.split(cols, splits)
                best_grp = max(groups, key=lambda g: int(col_counts[g].sum()))
                sel = np.zeros_like(er)
                sel[:, best_grp] = er[:, best_grp]
                g_ys, g_xs = np.where(sel)
                centroids["green"] = (float(np.mean(g_xs)), gy0 + float(np.mean(g_ys)))

        # Require some minimum area so tiny speckles don't trigger.
        MIN_FRAC = 0.001  # 0.1% of ROI

        # Build a list of all detections above threshold (sorted by coverage).
        detections = []
        if coverage:
            for name, frac in sorted(coverage.items(), key=lambda kv: kv[1], reverse=True):
                if frac < MIN_FRAC:
                    continue
                centroid = centroids.get(name, (orig_img_w / 2, orig_img_h / 2))
                detections.append({
                    'color': name,
                    'coverage': float(frac),
                    'centroid_px': centroid,
                    'img_width': orig_img_w,
                    'img_height': orig_img_h
                })

        # NOTE: a filter here used to drop ALL green detections whenever a planned
        # path was being followed ("don't interrupt path following"). Its actual
        # effect: while driving — which is almost always — green marking and any
        # green safety reaction were completely blind, which is precisely when the
        # robot drives onto poison. This was confirmed as the primary cause of
        # green crossings in extensive testing. Removed.

        if detections:
            best = detections[0]
            print(f"Camera detected: {best['color']} ({best['coverage']*100.0:.1f}%)")
            # Keep backwards-compatible top-level fields, and also return all detections.
            out = dict(best)
            out['detections'] = detections
            return out

        # No confident detection
        if debug_now and coverage:
            detected, detected_frac = max(coverage.items(), key=lambda kv: kv[1])
            print(f"Camera best: {detected} ({detected_frac*100.0:.2f}%)")
        return None

    def _get_detection_for_color(self, detection_info, color_name):
        """Return the detection dict for a specific color.

        Supports both the legacy single-detection dict and the newer
        multi-detection format returned by process_camera().
        """
        if detection_info is None:
            return None

        try:
            if isinstance(detection_info, dict) and detection_info.get('color') == color_name:
                return detection_info
        except Exception:
            pass

        if isinstance(detection_info, dict):
            dets = detection_info.get('detections')
            if isinstance(dets, list):
                for det in dets:
                    if isinstance(det, dict) and det.get('color') == color_name:
                        return det

        return None

    def raycast_free(self, x0, y0, x1, y1):
        i0, j0 = self.world_to_grid(x0, y0)
        i1, j1 = self.world_to_grid(x1, y1)
        di, dj = abs(i1 - i0), abs(j1 - j0)
        si, sj = (1 if i0 < i1 else -1), (1 if j0 < j1 else -1)
        err = di - dj
        i, j = i0, j0
        while (i != i1 or j != j1):
            # Full-strength carve, matching the Maze4 reference mapping. This is
            # the map's self-cleaning mechanism: any wrongly-blackened cell
            # (range noise, brief pose error) is erased by later beams passing
            # through it, so walls converge THIN. An earlier "stop the carve at
            # confirmed walls" variant made noise cells sticky and thickened
            # walls instead — do not reintroduce it.
            self.update_cell(i, j, self.L_FREE)
            e2 = 2 * err
            if e2 > -dj: err -= dj; i += si
            if e2 < di: err += di; j += sj
        return i1, j1

    def estimate_object_world_coords(self, detection_info, standoff_m=0.0, depth_override=None, use_lidar=False):
        """Estimate world coordinates of detected color object using depth camera.
        
        Args:
            detection_info: dict with 'centroid_px', 'img_width', 'img_height'
        
        Returns:
            (world_x, world_y) or None if depth unavailable
        """
        if detection_info is None:
            return None
        
        cx, cy = detection_info['centroid_px']
        img_w = detection_info['img_width']
        img_h = detection_info['img_height']
        
        # Calculate horizontal angle offset from image center.
        # Positive angle = object is to the left in camera view.
        cx_normalized = (cx - img_w / 2.0) / (img_w / 2.0)  # -1..1
        fov_h = self._camera_fov_h(img_w, img_h)
        angle_offset = -cx_normalized * (fov_h / 2.0)

        # Get depth at centroid (camera depth) and optionally clamp with lidar.
        depth_cam = None
        if depth_override is None:
            depth_cam = self._get_depth_at(cx, cy, window_px=2)
        else:
            try:
                depth_cam = float(depth_override)
            except Exception:
                depth_cam = None

        depth_lidar = self._get_lidar_range_at_angle(angle_offset, window_rays=2) if use_lidar else None

        depth = None
        if depth_cam is not None and depth_lidar is not None:
            # If depth camera accidentally hits background, lidar is usually closer.
            depth = min(depth_cam, depth_lidar)
        elif depth_cam is not None:
            depth = depth_cam
        elif depth_lidar is not None:
            depth = depth_lidar

        if depth is None:
            depth = 1.0  # conservative fallback

        # Apply a standoff so we target a point in front of the object.
        # Keep a minimum so we don't generate a goal on top of the robot.
        standoff_m = float(standoff_m) if standoff_m is not None else 0.0
        depth = max(0.20, depth - max(0.0, standoff_m))
        
        # Get robot pose
        rx, ry, ryaw, rz = self.get_pose()
        
        # Calculate world coordinates
        # Object angle in world frame = robot yaw + camera angle offset
        obj_angle = ryaw + angle_offset
        obj_x = rx + depth * math.cos(obj_angle)
        obj_y = ry + depth * math.sin(obj_angle)
        
        return (obj_x, obj_y)

    def perform_initial_scan(self):
        """Perform 360-degree rotation scan. Returns True when complete."""
        _, _, current_yaw, _ = self.get_pose()
        
        # Initialize scan tracking
        if self.initial_scan_start_yaw is None:
            self.initial_scan_start_yaw = current_yaw
            self.initial_scan_last_yaw = current_yaw
            self.initial_scan_accumulated = 0.0
            print("Starting 360-degree initial scan...")
            return False
        
        # Calculate yaw change since last step
        dyaw = current_yaw - self.initial_scan_last_yaw
        # Normalize to [-pi, pi]
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        
        # Accumulate rotation (only count positive rotation for consistency)
        if dyaw > 0:
            self.initial_scan_accumulated += dyaw
        else:
            # If rotating other direction, still count magnitude
            self.initial_scan_accumulated += abs(dyaw)
        
        self.initial_scan_last_yaw = current_yaw
        
        # Check if we've completed 360 degrees (2*pi radians)
        if self.initial_scan_accumulated >= 2.0 * math.pi:
            print(f"Initial 360-degree scan complete! Accumulated: {math.degrees(self.initial_scan_accumulated):.1f} degrees")
            self.initial_scan_done = True
            self.stop()
            return True
        
        # Continue spinning
        self.spin_in_place(speed=1.5)  # Slower for better detection
        return False

    def handle_color_detection(self, detection_info):
        """Process color detection and update state accordingly.
        
        Returns:
            'go_to_blue' if blue detected and should navigate to it
            'continue' otherwise
        """
        if detection_info is None:
            return 'continue'

        # If multiple colors are detected in the same ROI, process each one.
        # This keeps the existing single-color behavior but avoids dropping e.g. blue
        # when green is also present.
        if isinstance(detection_info, dict) and detection_info.get('detections'):
            for det in detection_info['detections']:
                action = self.handle_color_detection(det)
                if action == 'go_to_blue':
                    return 'go_to_blue'
            return 'continue'
        
        color = detection_info['color']
        coverage = detection_info.get('coverage', 0.0)
        
        # Only record coordinates when pillar has around 30% coverage for better accuracy
        PILLAR_RECORD_COVERAGE_THRESHOLD = 0.05
        
        if color == 'blue' and not self.blue_found:
            # Check if coverage is high enough to record coordinates
            if coverage >= PILLAR_RECORD_COVERAGE_THRESHOLD:
                # Depth from the EXACT centroid pixel — no min-of-window, no lidar.
                # The default estimate takes the MINIMUM depth over a pixel window
                # and mixes in the nearest lidar ray; when the pillar is glimpsed
                # at range or through/near a gap, that minimum clips a NEARER wall
                # edge — bearing stays right but the distance collapses, locking
                # the pillar at a completely wrong position. There is also a
                # depth=1.0m fabrication fallback in the shared estimator when no
                # reading is available. The mask centroid of a convex pillar blob
                # lies ON the pillar, so its exact pixel is the true distance;
                # if it can't be read, wait for a better frame instead of locking
                # in garbage. Range-capped: far estimates scale bearing error
                # into large position error.
                cx_b, cy_b = detection_info['centroid_px']
                d_b = self._get_depth_at(cx_b, cy_b, window_px=0)
                if d_b is None:
                    print("BLUE seen but no depth reading at centroid — waiting for a better view...")
                    return 'continue'
                if d_b > 2.5:
                    print(f"BLUE seen but too far to record accurately ({d_b:.2f}m > 2.5m), waiting...")
                    return 'continue'
                coords_pillar = self.estimate_object_world_coords(
                    detection_info,
                    standoff_m=0.0,
                    depth_override=d_b,
                    use_lidar=False,
                )
                if coords_pillar:
                    rx, ry, _, _ = self.get_pose()
                    px, py = coords_pillar
                    dx = rx - px
                    dy = ry - py
                    dist = math.sqrt(dx * dx + dy * dy)

                    standoff = float(getattr(self, 'BLUE_NAV_STANDOFF_M', 0.30))
                    if dist > 0.01:
                        # Ensure the standoff doesn't overshoot past the robot.
                        standoff = min(standoff, max(0.0, dist - 0.05))
                        coords_nav = (px + (dx / dist) * standoff, py + (dy / dist) * standoff)
                    else:
                        coords_nav = coords_pillar

                    self.blue_coords = coords_nav
                    self.blue_found = True
                    print(
                        f"BLUE object detected at {coverage*100.0:.1f}% coverage! "
                        f"pillar=({coords_pillar[0]:.2f},{coords_pillar[1]:.2f}) "
                        f"nav=({coords_nav[0]:.2f},{coords_nav[1]:.2f}) standoff={standoff:.2f}m"
                    )
                    return 'go_to_blue'
            else:
                print(f"BLUE object detected but coverage too low ({coverage*100.0:.1f}% < {PILLAR_RECORD_COVERAGE_THRESHOLD*100.0:.1f}%), waiting for better view...")
        
        elif color == 'yellow' and not self.yellow_found:
            # Check if coverage is high enough to record coordinates
            if coverage >= PILLAR_RECORD_COVERAGE_THRESHOLD:
                # Estimate pillar position in world coordinates.
                coords_pillar = self.estimate_object_world_coords(
                    detection_info,
                    standoff_m=0.0,
                    depth_override=None,
                    use_lidar=True,
                )

                if coords_pillar:
                    self._yellow_lock_samples.append(coords_pillar)

                # Collect a few consecutive consistent detections before saving.
                if len(self._yellow_lock_samples) < int(getattr(self, "YELLOW_LOCK_SAMPLES", 7)):
                    self._yellow_lock_debug_step += 1
                    if self._yellow_lock_debug_step % 10 == 0:
                        print(f"YELLOW lock-in: collecting samples {len(self._yellow_lock_samples)}/{self.YELLOW_LOCK_SAMPLES}")
                    return 'continue'

                # Check spread of samples; if too large, reset and keep collecting.
                xs = [p[0] for p in self._yellow_lock_samples]
                ys = [p[1] for p in self._yellow_lock_samples]
                mean_x = sum(xs) / len(xs)
                mean_y = sum(ys) / len(ys)
                max_dev = 0.0
                for (sx, sy) in self._yellow_lock_samples:
                    max_dev = max(max_dev, math.hypot(sx - mean_x, sy - mean_y))

                if max_dev > float(getattr(self, "YELLOW_LOCK_MAX_SPREAD_M", 0.35)):
                    self._yellow_lock_samples.clear()
                    return 'continue'

                # Lock in averaged pillar coordinates.
                coords_pillar = (mean_x, mean_y)

                # Navigation target: 0.25m in front of the pillar (towards robot).
                rx, ry, _, _ = self.get_pose()
                px, py = coords_pillar
                dx = rx - px
                dy = ry - py
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0.01:
                    coords_nav = (px + (dx / dist) * self.YELLOW_NAV_STANDOFF_M, py + (dy / dist) * self.YELLOW_NAV_STANDOFF_M)
                else:
                    coords_nav = coords_pillar

                self.yellow_pillar_coords = coords_pillar
                self.yellow_coords = coords_nav
                self.yellow_found = True
                self._yellow_lock_samples.clear()

                print(f"YELLOW object detected at {coverage*100.0:.1f}% coverage! pillar=({coords_pillar[0]:.2f},{coords_pillar[1]:.2f}) nav=({coords_nav[0]:.2f},{coords_nav[1]:.2f})")
                print(f"Navigation target is 0.25m in front of pillar (towards robot position ({rx:.2f},{ry:.2f}))")
                yi, yj = self.world_to_grid(coords_pillar[0], coords_pillar[1])
                yni, ynj = self.world_to_grid(coords_nav[0], coords_nav[1])
                print(f"Yellow grid: pillar=({yi},{yj}) nav=({yni},{ynj})")
                print("Yellow coordinates saved. Will navigate there after blue is reached.")
            else:
                print(f"YELLOW object detected but coverage too low ({coverage*100.0:.1f}% < {PILLAR_RECORD_COVERAGE_THRESHOLD*100.0:.1f}%), waiting for better view...")
        
        return 'continue'

    def get_color_object_goal(self, world_coords):
        """Convert world coordinates to grid goal."""
        if world_coords is None:
            return None
        wx, wy = world_coords
        gi, gj = self.world_to_grid(wx, wy)
        # Clamp to grid bounds
        gi = max(0, min(self.GRID_SIZE - 1, gi))
        gj = max(0, min(self.GRID_SIZE - 1, gj))
        return (gi, gj)

    def get_color_object_goal_free(self, world_coords, search_radius_cells=18, blacklist=None):
        """Convert world coords to a *reachable planning goal*.

        The pillar/object itself often lies on an OCC cell (or within hard inflation),
        which makes A* fail even when a path exists to get near it. This returns the
        nearest nearby FREE cell (and not hard-blocked) within a search radius.
        """
        target = self.get_color_object_goal(world_coords)
        if target is None:
            return None

        ti, tj = target
        if not self.inside_map(ti, tj):
            return None

        hard_blocked = self._compute_hard_blocked()

        bl = blacklist if blacklist is not None else set()

        # If the target cell is already usable, keep it.
        if (ti, tj) not in bl and self.display_state[tj][ti] == self.FREE and not hard_blocked[tj][ti]:
            return (ti, tj)

        best = None
        best_d2 = None

        # Search expanding rings around the target.
        rmax = int(max(1, search_radius_cells))
        for r in range(1, rmax + 1):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    # Only check the perimeter of the square (ring)
                    if abs(di) != r and abs(dj) != r:
                        continue
                    ni, nj = ti + di, tj + dj
                    if not self.inside_map(ni, nj):
                        continue
                    if self.display_state[nj][ni] != self.FREE:
                        continue
                    if hard_blocked[nj][ni]:
                        continue
                    if (ni, nj) in bl:
                        continue

                    d2 = di * di + dj * dj
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = (ni, nj)

            # If we found any candidate in this ring, stop expanding.
            if best is not None:
                break

        if best is not None:
            print(f"Snapped color goal from {target} to nearby FREE cell {best}")
        else:
            print(f"No FREE cell found near target {target} within radius={rmax} cells")

        return best

    def reached_color_object(self, world_coords, threshold_m=0.2):
        """Check if robot has reached the color object location."""
        if world_coords is None:
            return False
        rx, ry, _, _ = self.get_pose()
        ox, oy = world_coords
        dist = math.hypot(ox - rx, oy - ry)
        return dist < threshold_m

    def mark_green_floor_obstacles(self, detection_info):
        """Use RGB camera to detect green floor regions and mark them as obstacles.
        
        When green appears in bottom of ROI, it means green is close (~0.75m from robot).
        Mark all cells within 0.75m radius from robot as blocked and invalidate nearby frontiers.
        Once a region is marked, it won't be re-marked.
        """
        green_det = self._get_detection_for_color(detection_info, 'green')
        if green_det is None:
            return
        
        # Get the camera image dimensions and centroid
        cx, cy = green_det['centroid_px']
        img_w = green_det['img_width']
        img_h = green_det['img_height']
        coverage = green_det.get('coverage', 0.0)
        
        # Only mark as obstacle if green coverage is significant (not just noise)
        if coverage < 0.015:  # 1.5% threshold
            return
        
        # Calculate ROI boundaries to determine if green is in bottom region
        roi_frac = getattr(self, "CAMERA_ROI_FRAC", 0.4)
        side = max(1, int(min(img_w, img_h) * roi_frac))
        cy_frac = float(getattr(self, "CAMERA_ROI_CENTER_Y_FRAC", 0.6))
        cy_frac = max(0.0, min(1.0, cy_frac))
        roi_center_y = int(cy_frac * img_h)
        y0 = max(0, min(img_h - side, roi_center_y - (side // 2)))
        y1 = min(img_h, y0 + side)
        
        # Check if green centroid is in bottom 30% of ROI
        roi_height = y1 - y0
        bottom_threshold_y = y0 + roi_height * 0.7  # Bottom 30% of ROI
        
        green_in_bottom_roi = cy >= bottom_threshold_y
        
        rx, ry, ryaw, rz = self.get_pose()
        ri, rj = self.world_to_grid(rx, ry)
        
        # Calculate elevation difference from initial position
        z_diff = abs(rz - self.initial_z_position) if self.initial_z_position is not None else 0.0
        elevated = z_diff > self.Z_ELEVATION_THRESHOLD
        
        if elevated:
            print(f"ELEVATION DETECTED: Z-diff = {z_diff:.4f}m (threshold: {self.Z_ELEVATION_THRESHOLD}m)")
        
        # Check if we've already marked this robot position area
        region_key = (ri // 10, rj // 10)
        if region_key in self.marked_green_regions:
            return  # Already marked this region
        
        # Trigger blocking when: green detected in bottom ROI AND robot is elevated
        if green_in_bottom_roi and elevated:
            # Robot is climbing onto green platform! Block the region
            print(f"GREEN PLATFORM DETECTED - Robot elevated {z_diff:.4f}m! Blocking ~200 cells ahead and sides!")
            
            # Mark this region as processed
            self.marked_green_regions.add(region_key)

            # Cooldown after marking to prevent repeated triggers
            try:
                self._green_cooldown_until_time = float(self.robot.getTime()) + float(getattr(self, "GREEN_COOLDOWN_S", 12.0))
            except Exception:
                pass
            
            # Block approximately 200 cells in forward direction and sides
            # Forward distance: 2m, Width: ±1m from center
            forward_distance_m = 2.0  # 2m forward
            side_width_m = 1.0  # ±1m to each side
            
            forward_cells = int(forward_distance_m / self.MAP_RES)
            side_cells = int(side_width_m / self.MAP_RES)
            
            cells_marked = 0
            blocked_cells = []
            
            # Create rectangular blocking region in front of robot
            for forward_dist in range(0, forward_cells):
                # Distance in world coordinates
                dist_world = forward_dist * self.MAP_RES
                
                # Sample across the width perpendicular to robot heading
                for side_offset in range(-side_cells, side_cells + 1):
                    side_world = side_offset * self.MAP_RES
                    
                    # Calculate world position (forward along ryaw, offset perpendicular)
                    sample_x = rx + dist_world * math.cos(ryaw) - side_world * math.sin(ryaw)
                    sample_y = ry + dist_world * math.sin(ryaw) + side_world * math.cos(ryaw)
                    
                    gi, gj = self.world_to_grid(sample_x, sample_y)
                    
                    if self.inside_map(gi, gj) and (gi, gj) not in blocked_cells:
                        # Mark as occupied with very high confidence
                        self.update_cell(gi, gj, self.L_OCC * 3.0)
                        blocked_cells.append((gi, gj))
                        cells_marked += 1
            
            print(f"Marked {cells_marked} cells as elevated green platform obstacles")
            
            # Clear current path if it goes through the blocked region
            if self.path is not None and len(self.path) > 0:
                path_invalidated = False
                for (pi, pj) in self.path:
                    if (pi, pj) in blocked_cells:
                        path_invalidated = True
                        break
                
                if path_invalidated:
                    print("Current path goes through elevated green platform - clearing path!")
                    self.path = None
                    self.path_goal = None
                    self.path_index = 0
                    self.need_new_goal = True
            
            # Invalidate frontiers in the blocked region
            if hasattr(self, '_current_frontiers'):
                invalidated_count = 0
                for (fi, fj) in self._current_frontiers:
                    if (fi, fj) in blocked_cells:
                        self.mark_goal_visited((fi, fj))
                        invalidated_count += 1
                
                if invalidated_count > 0:
                    print(f"Invalidated {invalidated_count} frontiers in elevated green danger zone")
            
            return  # Exit after handling elevated platform
        
        # Original behavior: if green in bottom ROI but not elevated yet
        if green_in_bottom_roi:
            # Green is close! Mark front 0.75m distance ahead of robot as blocked
            print(f"GREEN IN BOTTOM ROI - Marking 0.75m in front of robot as blocked!")
            
            # Mark this region as processed
            self.marked_green_regions.add(region_key)
            
            # Block only cells in front of robot (forward direction)
            front_distance_m = 0.75
            max_dist_cells = int(front_distance_m / self.MAP_RES)
            
            # Define forward cone angle (±45 degrees from robot heading)
            cone_half_angle = math.pi / 4  # 45 degrees
            
            cells_marked = 0
            blocked_cells = []
            
            # Mark cells in front of robot within 0.75m and ±45° cone
            for dist_step in range(1, max_dist_cells + 1):
                # Distance in world coordinates
                dist_world = dist_step * self.MAP_RES
                
                # Sample multiple angles in the forward cone
                num_angle_samples = max(3, int(dist_step / 5))  # More samples for farther distances
                for angle_idx in range(num_angle_samples):
                    if num_angle_samples == 1:
                        angle_offset = 0
                    else:
                        # Spread angles across the cone
                        angle_offset = -cone_half_angle + (2 * cone_half_angle * angle_idx / (num_angle_samples - 1))
                    
                    sample_angle = ryaw + angle_offset

                    sample_x = rx + dist_world * math.cos(sample_angle)
                    sample_y = ry + dist_world * math.sin(sample_angle)
                    
                    gi, gj = self.world_to_grid(sample_x, sample_y)
                    
                    if self.inside_map(gi, gj) and (gi, gj) not in blocked_cells:
                        # Mark as occupied with very high confidence
                        self.update_cell(gi, gj, self.L_OCC * 3.0)
                        blocked_cells.append((gi, gj))
                        cells_marked += 1
            
            print(f"Marked {cells_marked} cells (0.75m front) as green floor obstacles ahead of robot")
            
            # Clear current path if it goes through the blocked region
            if self.path is not None and len(self.path) > 0:
                path_invalidated = False
                for (pi, pj) in self.path:
                    if (pi, pj) in blocked_cells:
                        path_invalidated = True
                        break
                
                if path_invalidated:
                    print("Current path goes through green zone - clearing path!")
                    self.path = None
                    self.path_goal = None
                    self.path_index = 0
                    self.need_new_goal = True
            
            # Invalidate frontiers in the blocked region
            if hasattr(self, '_current_frontiers'):
                invalidated_count = 0
                for (fi, fj) in self._current_frontiers:
                    if (fi, fj) in blocked_cells:
                        self.mark_goal_visited((fi, fj))
                        invalidated_count += 1
                
                if invalidated_count > 0:
                    print(f"Invalidated {invalidated_count} frontiers in green danger zone")
        else:
            # Green is further away, mark the distant location
            depth = self._get_depth_at(cx, cy)
            if depth is None:
                depth = 1.5
            
            cx_normalized = (cx - img_w / 2.0) / (img_w / 2.0)
            angle_offset = -cx_normalized * (self.CAMERA_FOV_H / 2.0)
            obj_angle = ryaw + angle_offset
            
            center_x = rx + depth * math.cos(obj_angle)
            center_y = ry + depth * math.sin(obj_angle)
            center_i, center_j = self.world_to_grid(center_x, center_y)
            
            # Mark this region as processed
            self.marked_green_regions.add(region_key)
            
            # Mark ~100 cells around the detected green area
            radius_cells = 6
            cells_marked = 0
            
            for di in range(-radius_cells, radius_cells + 1):
                for dj in range(-radius_cells, radius_cells + 1):
                    dist_sq = di * di + dj * dj
                    if dist_sq > radius_cells * radius_cells:
                        continue
                    
                    gi = center_i + di
                    gj = center_j + dj
                    
                    if self.inside_map(gi, gj):
                        self.update_cell(gi, gj, self.L_OCC * 2.0)
                        cells_marked += 1
            
            print(f"Marked {cells_marked} cells as green floor obstacles at distant location ({center_i}, {center_j})")

    def maybe_update_floating_walls(self):
        """Rate-limited wrapper for depth-camera low-wall detection."""
        if not getattr(self, 'FLOATING_WALL_ENABLED', False):
            return 0
        self._floating_wall_step_counter += 1
        if (self._floating_wall_step_counter % max(1, self.FLOATING_WALL_POLL_EVERY_N_STEPS)) != 0:
            return 0
        return self.detect_and_mark_low_walls()

    def detect_and_mark_low_walls(self):
        """Mark lidar-invisible LOW floating walls using the depth camera.

        Per grid cell with depth hits: if the LOWEST hit height is between
        FLOATING_WALL_MIN_HEIGHT (floor noise) and ROBOT_CLEARANCE_HEIGHT, the
        robot's body would collide → obstacle. Hits entirely above the clearance
        height mean the robot passes under → leave free (lidar also never marks
        those, so nothing needs unblocking). See __init__ for the Maze3 geometry
        that calibrates ROBOT_CLEARANCE_HEIGHT.
        """
        if self.camera_depth is None:
            return 0

        # Only mark while driving straight or standing: during fast turns the
        # camera sweeps sideways and small pose/frame timing offsets place the
        # PERMANENT marks into free space (blacked-out corridors, dead paths).
        try:
            _, _, _yaw_lw, _ = self.get_pose()
            _last_lw = getattr(self, '_lowwall_last_yaw', None)
            self._lowwall_last_yaw = _yaw_lw
            if _last_lw is not None:
                _dy_lw = math.atan2(math.sin(_yaw_lw - _last_lw), math.cos(_yaw_lw - _last_lw))
                if abs(_dy_lw) > 0.03:
                    return 0
        except Exception:
            pass

        try:
            depth_raw = self.camera_depth.getRangeImage()
            if not depth_raw:
                return 0
            dw = int(self.camera_depth.getWidth())
            dh = int(self.camera_depth.getHeight())
            depth_m = np.asarray(depth_raw, dtype=np.float32).reshape((dh, dw))
            fov = float(self.camera_depth.getFov())
            if fov <= 0:
                return 0
        except Exception:
            return 0

        fx = dw / (2.0 * math.tan(fov / 2.0))
        cx_cam, cy_cam = dw / 2.0, dh / 2.0
        stride = max(1, int(self.DEPTH_PIXEL_STRIDE))

        v_idx, u_idx = np.indices(depth_m.shape, dtype=np.float32)
        valid = (
            np.isfinite(depth_m)
            & (depth_m > self.DEPTH_OBSTACLE_MIN_DIST)
            & (depth_m < self.DEPTH_OBSTACLE_MAX_DIST)
            & ((u_idx.astype(np.int32) % stride) == 0)
            & ((v_idx.astype(np.int32) % stride) == 0)
        )
        if not np.any(valid):
            return 0

        D = depth_m[valid]
        u = u_idx[valid]
        v = v_idx[valid]

        # Pinhole back-projection into the robot frame (X fwd, Y left, Z up).
        X_c = D
        Y_c = -D * (u - cx_cam) / fx
        Z_c = -D * (v - cy_cam) / fx
        world_h = self.FLOATING_WALL_CAMERA_HEIGHT_M + Z_c

        X_r = X_c + self.FLOATING_WALL_FORWARD_OFFSET_M
        keep = np.sqrt(X_r * X_r + Y_c * Y_c) > self.FLOATING_WALL_STOP_BEFORE_ROBOT_M
        if not np.any(keep):
            return 0
        X_r, Y_r, world_h = X_r[keep], Y_c[keep], world_h[keep]

        rx, ry, ryaw, _ = self.get_pose()
        cos_h, sin_h = math.cos(ryaw), math.sin(ryaw)
        Xw = rx + cos_h * X_r - sin_h * Y_r
        Yw = ry + sin_h * X_r + cos_h * Y_r

        # Per-cell lowest hit height decides passability.
        cell_min_h = {}
        for i_pt in range(len(Xw)):
            wh = float(world_h[i_pt])
            if wh <= self.FLOATING_WALL_MIN_HEIGHT:
                continue
            gi, gj = self.world_to_grid(float(Xw[i_pt]), float(Yw[i_pt]))
            if not self.inside_map(gi, gj):
                continue
            if (gi, gj) not in cell_min_h or wh < cell_min_h[(gi, gj)]:
                cell_min_h[(gi, gj)] = wh

        newly_locked = 0
        for (gi, gj), min_h in cell_min_h.items():
            # NO pass-under height exception: every floating wall is treated as
            # a normal wall. The old ROBOT_CLEARANCE_HEIGHT test tried to let
            # the robot drive under high slabs, but estimating a plank's bottom
            # edge from depth pixels is unreliable (the edge may be occluded or
            # thin), and a wrong "passable" call means driving into the plank.
            # Anything the depth camera sees above floor level gets marked.
            if self.display_state[gj][gi] == self.OCC:
                continue  # already an obstacle (lidar or previous pass)
            # Persistence gate: force-occupy is PERMANENT, so require the same
            # cell to be flagged on several separate polls first. Real planks
            # re-flag their cells every poll; projection noise does not.
            n = self._low_wall_hit_counts.get((gi, gj), 0) + 1
            self._low_wall_hit_counts[(gi, gj)] = n
            if n < int(getattr(self, 'FLOATING_WALL_CONFIRM_POLLS', 3)):
                continue
            self.force_occupy_cell(gi, gj)
            newly_locked += 1

        if newly_locked:
            self._hard_blocked_cache = None
            self._hard_blocked_cache_scan_id = None
            print(f"[LowWall] locked {newly_locked} lidar-invisible low-wall cells")
            # Replan if the current path now crosses locked cells.
            try:
                if self.path and any(self.display_state[pj][pi] == self.OCC for (pi, pj) in self.path):
                    self.path = None
                    self.path_goal = None
                    self.path_index = 0
                    self.need_new_goal = True
            except Exception:
                pass
        return newly_locked

    def mark_green_poison_zone(self, detection_info, radius_m=1.0):
        """While navigating, block a ~radius_m zone around detected green as OCC."""
        green_det = self._get_detection_for_color(detection_info, 'green')
        if green_det is None:
            return

        coords = self.estimate_object_world_coords(
            green_det,
            standoff_m=0.0,
            depth_override=None,
            use_lidar=True,
        )
        if not coords:
            return

        center_i, center_j = self.world_to_grid(coords[0], coords[1])
        region_key = (center_i // 10, center_j // 10)
        if region_key in self.poisoned_green_regions:
            return
        self.poisoned_green_regions.add(region_key)

        r_cells = int(max(1, float(radius_m) / float(self.MAP_RES)))
        r2 = r_cells * r_cells
        blocked = set()

        for dj in range(-r_cells, r_cells + 1):
            for di in range(-r_cells, r_cells + 1):
                if di * di + dj * dj > r2:
                    continue
                gi = center_i + di
                gj = center_j + dj
                if self.inside_map(gi, gj):
                    self.update_cell(gi, gj, self.L_OCC * 3.0)
                    blocked.add((gi, gj))

        # If our current path crosses the poison zone, force a replan.
        if self.path:
            for cell in self.path:
                if cell in blocked:
                    self.path = None
                    self.path_goal = None
                    self.path_index = 0
                    break

    def start_green_scan(self, detection_info):
        """Start the green region scanning process when 5% green is detected."""
        if self.green_scan_state is not None:
            return  # Already scanning
        
        # Save current mission state to resume later
        self.green_scan_saved_mission_state = self.mission_state
        self.green_scan_saved_goal = self.current_goal
        
        # Get initial heading towards green
        cx, cy = detection_info['centroid_px']
        img_w = detection_info['img_width']
        cx_normalized = (cx - img_w / 2.0) / (img_w / 2.0)
        rx, ry, ryaw, _ = self.get_pose()
        angle_offset = -cx_normalized * (self.CAMERA_FOV_H / 2.0)
        self.green_scan_center_yaw = ryaw + angle_offset
        self.green_scan_start_yaw = ryaw
        
        # Reset scan extents
        self.green_scan_left_yaw = None
        self.green_scan_right_yaw = None
        
        print(f"GREEN SCAN STARTED - 5%+ green detected! Initiating approach and scan...")
        self.green_scan_state = 'approach'
        self.stop()

    def process_green_scan(self, detection_info):
        """Process the green scanning state machine. Returns True if scanning is active."""
        if self.green_scan_state is None:
            return False
        
        green_det = self._get_detection_for_color(detection_info, 'green')
        coverage = green_det.get('coverage', 0.0) if green_det else 0.0
        rx, ry, ryaw, rz = self.get_pose()
        
        # Calculate if green is in bottom of ROI (close)
        green_in_bottom = False
        if green_det:
            cx, cy = green_det['centroid_px']
            img_h = green_det['img_height']
            img_w = green_det['img_width']
            
            roi_frac = getattr(self, "CAMERA_ROI_FRAC", 0.4)
            side = max(1, int(min(img_w, img_h) * roi_frac))
            cy_frac = float(getattr(self, "CAMERA_ROI_CENTER_Y_FRAC", 0.6))
            roi_center_y = int(cy_frac * img_h)
            y0 = max(0, min(img_h - side, roi_center_y - (side // 2)))
            y1 = min(img_h, y0 + side)
            roi_height = y1 - y0
            bottom_threshold_y = y0 + roi_height * 0.7
            green_in_bottom = cy >= bottom_threshold_y
        
        # STATE: APPROACH - Move towards green until 10% coverage or bottom of ROI
        if self.green_scan_state == 'approach':
            if coverage >= self.GREEN_CLOSE_COVERAGE or green_in_bottom:
                print(f"GREEN SCAN: Close enough (coverage={coverage*100:.1f}%, bottom={green_in_bottom})")
                self.green_scan_state = 'position'
                self.stop()
            elif coverage >= 0.01:  # Still see green, keep approaching
                # Turn to face green and move forward slowly
                target_yaw = self.green_scan_center_yaw
                yaw_error = target_yaw - ryaw
                # Normalize to [-pi, pi]
                while yaw_error > math.pi: yaw_error -= 2 * math.pi
                while yaw_error < -math.pi: yaw_error += 2 * math.pi
                
                if abs(yaw_error) > 0.1:
                    # Turn towards green
                    turn_speed = 1.5 if yaw_error > 0 else -1.5
                    for idx, motor in enumerate(self.motors):
                        motor.setVelocity(-turn_speed if idx % 2 == 0 else turn_speed)
                else:
                    # Move forward slowly
                    for motor in self.motors:
                        motor.setVelocity(2.0)
            else:
                # Lost green, abort scan
                print("GREEN SCAN: Lost green during approach, aborting...")
                self.finish_green_scan(abort=True)
            return True
        
        # STATE: POSITION - Turn to face green directly
        elif self.green_scan_state == 'position':
            if green_det:
                cx, cy = green_det['centroid_px']
                img_w = green_det['img_width']
                cx_normalized = (cx - img_w / 2.0) / (img_w / 2.0)
                
                if abs(cx_normalized) < 0.1:  # Green is centered
                    print("GREEN SCAN: Positioned, starting left scan...")
                    self.green_scan_state = 'scan_left'
                    self.green_scan_start_yaw = ryaw
                    self.stop()
                else:
                    # Turn to center the green
                    turn_speed = -1.0 * cx_normalized  # Turn towards green
                    for idx, motor in enumerate(self.motors):
                        motor.setVelocity(-turn_speed if idx % 2 == 0 else turn_speed)
            else:
                print("GREEN SCAN: Lost green during positioning, aborting...")
                self.finish_green_scan(abort=True)
            return True
        
        # STATE: SCAN_LEFT - Rotate left to find edge of green
        elif self.green_scan_state == 'scan_left':
            if coverage >= 0.01:  # Still see green
                self.green_scan_left_yaw = ryaw
                # Keep rotating left
                for idx, motor in enumerate(self.motors):
                    motor.setVelocity(1.0 if idx % 2 == 0 else -1.0)  # Turn left
            else:
                # Lost green on left side, found left edge
                print(f"GREEN SCAN: Left edge found at yaw={math.degrees(self.green_scan_left_yaw):.1f}°")
                self.green_scan_state = 'scan_right'
                # Return to center first
                self.stop()
            
            # Safety: don't rotate more than 90 degrees
            yaw_diff = ryaw - self.green_scan_start_yaw
            while yaw_diff > math.pi: yaw_diff -= 2 * math.pi
            while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
            if abs(yaw_diff) > math.pi / 2:
                print("GREEN SCAN: Max rotation reached on left, switching to right...")
                self.green_scan_state = 'scan_right'
                self.stop()
            return True
        
        # STATE: SCAN_RIGHT - Rotate right to find other edge of green
        elif self.green_scan_state == 'scan_right':
            # First return past center to scan right
            yaw_diff = ryaw - self.green_scan_start_yaw
            while yaw_diff > math.pi: yaw_diff -= 2 * math.pi
            while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
            
            if coverage >= 0.01:  # Still see green
                self.green_scan_right_yaw = ryaw
            
            # Keep rotating right
            for idx, motor in enumerate(self.motors):
                motor.setVelocity(-1.0 if idx % 2 == 0 else 1.0)  # Turn right
            
            # Check if we've scanned far enough right (lost green or max rotation)
            if coverage < 0.01 and yaw_diff < -0.1:
                print(f"GREEN SCAN: Right edge found at yaw={math.degrees(self.green_scan_right_yaw) if self.green_scan_right_yaw else 'N/A'}°")
                self.green_scan_state = 'mark'
                self.stop()
            
            # Safety: don't rotate more than 90 degrees past center
            if yaw_diff < -math.pi / 2:
                print("GREEN SCAN: Max rotation reached on right, marking...")
                self.green_scan_state = 'mark'
                self.stop()
            return True
        
        # STATE: MARK - Calculate and mark the green region
        elif self.green_scan_state == 'mark':
            self.mark_scanned_green_region()
            self.finish_green_scan()
            return True
        
        return False

    def mark_scanned_green_region(self):
        """Mark the scanned green region as blocked based on scan results."""
        rx, ry, ryaw, _ = self.get_pose()
        ri, rj = self.world_to_grid(rx, ry)
        
        # Calculate green region center using scan angles
        center_yaw = self.green_scan_start_yaw
        if self.green_scan_left_yaw is not None and self.green_scan_right_yaw is not None:
            # Average of left and right edges
            center_yaw = (self.green_scan_left_yaw + self.green_scan_right_yaw) / 2
        
        # Green is approximately 0.65m away when in bottom of ROI
        distance = self.green_scan_distance
        
        # Calculate center of green region in world coords
        green_x = rx + distance * math.cos(center_yaw)
        green_y = ry + distance * math.sin(center_yaw)
        center_i, center_j = self.world_to_grid(green_x, green_y)
        
        # Mark the green region as blocked (0.5m x 0.5m)
        region_size_cells = int(self.GREEN_SIZE_M / self.MAP_RES)
        half_size = region_size_cells // 2
        
        cells_marked = 0
        blocked_cells = []
        
        # Mark rectangular region
        for di in range(-half_size, half_size + 1):
            for dj in range(-half_size, half_size + 1):
                gi = center_i + di
                gj = center_j + dj
                
                if self.inside_map(gi, gj):
                    self.update_cell(gi, gj, self.L_OCC * 3.0)
                    blocked_cells.append((gi, gj))
                    cells_marked += 1
        
        # Store as marked region
        region_key = (center_i // 10, center_j // 10)
        self.marked_green_regions.add(region_key)
        
        print(f"GREEN SCAN COMPLETE: Marked {cells_marked} cells at ({center_i}, {center_j})")
        
        # Clear path if it goes through blocked region
        if self.path is not None:
            for (pi, pj) in self.path:
                if (pi, pj) in blocked_cells:
                    print("Path goes through marked green region - clearing!")
                    self.path = None
                    self.path_goal = None
                    self.need_new_goal = True
                    break
        
        # Invalidate frontiers in blocked region
        if hasattr(self, '_current_frontiers'):
            for (fi, fj) in self._current_frontiers:
                if (fi, fj) in blocked_cells:
                    self.mark_goal_visited((fi, fj))

    def finish_green_scan(self, abort=False):
        """End green scanning and restore previous mission state."""
        if abort:
            print("GREEN SCAN: Aborted, resuming previous mission...")
        else:
            print("GREEN SCAN: Completed, resuming previous mission...")
        
        # Restore saved state
        if self.green_scan_saved_mission_state:
            self.mission_state = self.green_scan_saved_mission_state
        if self.green_scan_saved_goal:
            self.current_goal = self.green_scan_saved_goal

        # If we detected a pillar during the scan window, don't lose that intent.
        if self.blue_found and not self.blue_reached:
            self.mission_state = 'go_to_blue'
        elif self.blue_reached and self.yellow_found:
            self.mission_state = 'go_to_yellow'
        
        # Reset scan state
        self.green_scan_state = None
        self.green_scan_saved_mission_state = None
        self.green_scan_saved_goal = None
        self.green_scan_start_yaw = None
        self.green_scan_left_yaw = None
        self.green_scan_right_yaw = None
        self.green_scan_center_yaw = None
        self.green_scan_pending_region_key = None

        # Cooldown after scan completion to prevent immediate re-trigger loops
        try:
            self._green_cooldown_until_time = float(self.robot.getTime()) + float(getattr(self, "GREEN_COOLDOWN_S", 12.0))
        except Exception:
            pass
        
        self.stop()
        
    def detect_frontiers(self):
        frontiers = []
        for j in range(1, self.GRID_SIZE-1):
            for i in range(1, self.GRID_SIZE-1):
                if self.display_state[j][i] != self.FREE:
                    continue

                # check 8-neighborhood
                is_frontier = False
                for dj in [-1,0,1]:
                    for di in [-1,0,1]:
                        if self.display_state[j+dj][i+di] == self.UNKNOWN:
                            is_frontier = True
                            break
                    if is_frontier:
                        break

                if is_frontier:
                    frontiers.append((i,j))
                            
        print("Frontiers:", len(frontiers))
        
        # Store for visualization
        self._current_frontiers = frontiers
                    
        return frontiers
    
    def debug_frontier(self, i, j):
        print("Cell:", self.display_state[j][i])
        for dj in [-1,0,1]:
            for di in [-1,0,1]:
                print(self.display_state[j+dj][i+di], end=" ")
            print()


    def cluster_frontiers(self,frontiers):
        clusters = []
        visited = set()

        for f in frontiers:
            if f in visited:
                continue

            stack = [f]
            cluster = []

            while stack:
                c = stack.pop()
                if c in visited:
                    continue
                visited.add(c)
                cluster.append(c)

                for n in frontiers:
                    if n not in visited:
                        if abs(n[0]-c[0]) <= 1 and abs(n[1]-c[1]) <= 1:
                            stack.append(n)

            clusters.append(cluster)

        return clusters
    
    def frontier_centroid(self,cluster):
        """Return a FREE goal cell representative for this cluster.

        Important: the numeric centroid can land on UNKNOWN/OCC due to rounding.
        We instead pick the frontier cell closest to the centroid, guaranteeing FREE.
        """
        cx = sum(c[0] for c in cluster) / len(cluster)
        cy = sum(c[1] for c in cluster) / len(cluster)
        return min(cluster, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)

    def frontier_nearest_to(self, cluster, ref_cell):
        """Pick a representative frontier cell closest to ref_cell=(i,j).

        This tends to choose goals that are easier/sooner to reach than using
        the cluster centroid (and avoids implicitly preferring large clusters).
        """
        ri, rj = ref_cell
        return min(cluster, key=lambda c: (c[0] - ri) ** 2 + (c[1] - rj) ** 2)

    def find_nearest_reachable_frontier(self, frontiers, start_cell):
        """Return the nearest *reachable* frontier using BFS over FREE cells.

        This uses path-distance (connectivity), not Euclidean distance.
        It also respects hard inflation so we don't pick targets that require
        passing too close to obstacles. Skips frontiers that are already visited.
        """
        if not frontiers:
            return None

        si, sj = start_cell
        if not self.inside_map(si, sj):
            return None
        si, sj = self._planning_start_cell((si, sj))

        frontier_set = set((int(i), int(j)) for (i, j) in frontiers)
        hard_blocked = self._compute_hard_blocked()

        visited = [[False for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        q = deque()
        q.append((si, sj))
        visited[sj][si] = True

        # 4-connected BFS matches our A* connectivity.
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        while q:
            i, j = q.popleft()

            if (i, j) in frontier_set and self.display_state[j][i] == self.FREE:
                # Don't return a goal in hard-inflated space or already visited
                if not hard_blocked[j][i] and not self.is_goal_visited((i, j)):
                    return (i, j)

            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if not self.inside_map(ni, nj):
                    continue
                if visited[nj][ni]:
                    continue
                if self.display_state[nj][ni] != self.FREE:
                    continue
                if hard_blocked[nj][ni]:
                    continue

                visited[nj][ni] = True
                q.append((ni, nj))

        return None

    def find_reachable_frontier_toward_goal(self, frontiers, start_cell, goal_cell):
        """Pick a reachable frontier that moves us toward goal_cell.

        We BFS over FREE cells (respecting hard inflation) to ensure reachability,
        then among the reachable frontier cells we select the one closest to the
        desired goal_cell in grid-space.
        """
        if not frontiers or goal_cell is None:
            return None

        si, sj = start_cell
        if not self.inside_map(si, sj):
            return None
        si, sj = self._planning_start_cell((si, sj))

        gi, gj = int(goal_cell[0]), int(goal_cell[1])
        if not self.inside_map(gi, gj):
            return None

        frontier_set = set((int(i), int(j)) for (i, j) in frontiers)
        hard_blocked = self._compute_hard_blocked()

        visited = [[False for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        q = deque()
        q.append((si, sj))
        visited[sj][si] = True

        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        best = None
        best_d2 = None

        while q:
            i, j = q.popleft()

            if (i, j) in frontier_set and self.display_state[j][i] == self.FREE and not hard_blocked[j][i]:
                d2 = (i - gi) ** 2 + (j - gj) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = (i, j)

            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if not self.inside_map(ni, nj):
                    continue
                if visited[nj][ni]:
                    continue
                if self.display_state[nj][ni] != self.FREE:
                    continue
                if hard_blocked[nj][ni]:
                    continue
                visited[nj][ni] = True
                q.append((ni, nj))

        return best

    
    def _line_of_sight_clear(self, a, b):
        """True if the straight grid line a->b crosses no confirmed OCC cell."""
        i0, j0 = int(a[0]), int(a[1])
        i1, j1 = int(b[0]), int(b[1])
        di, dj = abs(i1 - i0), abs(j1 - j0)
        si, sj = (1 if i0 < i1 else -1), (1 if j0 < j1 else -1)
        err = di - dj
        i, j = i0, j0
        while True:
            if self.inside_map(i, j) and self.display_state[j][i] == self.OCC:
                return False
            if i == i1 and j == j1:
                return True
            e2 = 2 * err
            if e2 > -dj: err -= dj; i += si
            if e2 < di: err += di; j += sj

    def _reachable_distance_map(self, start_cell):
        """BFS over traversable FREE cells (respecting hard inflation).

        Returns {(i, j): path_distance_in_cells} for every reachable cell.
        """
        si, sj = int(start_cell[0]), int(start_cell[1])
        if not self.inside_map(si, sj):
            return {}
        si, sj = self._planning_start_cell((si, sj))
        hard_blocked = self._compute_hard_blocked()
        dist = {(si, sj): 0}
        q = deque([(si, sj)])
        while q:
            i, j = q.popleft()
            d = dist[(i, j)]
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if (ni, nj) in dist or not self.inside_map(ni, nj):
                    continue
                if self.display_state[nj][ni] != self.FREE or hard_blocked[nj][ni]:
                    continue
                dist[(ni, nj)] = d + 1
                q.append((ni, nj))
        return dist

    def select_viewpoint_fallback(self, start_cell, frontiers):
        """Safe fallback when every frontier sits inside the inflated costmap.

        Late in exploration the remaining frontier cells lie in narrow gaps or
        against wall ends, all within the hard-inflation radius, so the normal
        nearest-reachable-frontier search returns None and exploration stalls
        even though unknown space (and possibly the blue pillar) remains.
        Instead of shrinking inflation and squeezing the robot into the gap
        (wall-scrape risk), drive to the nearest SAFELY reachable cell that has
        line of sight to a blocked frontier: from there the lidar sees past the
        gap and turns the unknown area into either free space (opening a real
        frontier) or wall (retiring it for good).

        Returns (goal, path) or None when nothing is left worth looking at.
        """
        if not frontiers:
            return None
        distance_map = self._reachable_distance_map(start_cell)
        if len(distance_map) <= 1:
            return None

        R = int(getattr(self, 'VIEWPOINT_SEARCH_RADIUS_CELLS', 12))
        candidates = []
        for (fi, fj) in frontiers:
            # Expanding square rings around the frontier: nearest reachable,
            # unvisited cell that can actually see it (no wall between them).
            found = None
            for r in range(0, R + 1):
                ring_best = None
                for di in range(-r, r + 1):
                    js = (-r, r) if abs(di) < r else tuple(range(-r, r + 1))
                    for dj in js:
                        v = (fi + di, fj + dj)
                        d = distance_map.get(v)
                        if d is None or self.is_goal_visited(v):
                            continue
                        if not self._line_of_sight_clear(v, (fi, fj)):
                            continue
                        if ring_best is None or d < ring_best[0]:
                            ring_best = (d, v)
                if ring_best is not None:
                    # Prefer viewpoints close to the frontier, then close to us.
                    found = (ring_best[0] + 3 * r, ring_best[1], (fi, fj))
                    break
            if found is not None:
                candidates.append(found)

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0])
        for _, goal, frontier in candidates[:10]:
            path = self.astar(start_cell, goal)
            if path:
                print(f"Viewpoint fallback: frontier {frontier} is inside the inflation "
                      f"zone; driving to nearby observable cell {goal} to look at it.")
                return goal, path
        return None

    def goal_reached(self, goal, threshold_m=0.18):
        if goal is None:
            return False
        rx, ry, _, _ = self.get_pose()
        gx, gy = self.grid_to_world_center(goal[0], goal[1])
        dist = math.hypot(gx - rx, gy - ry)
        return dist < threshold_m



    def run_mapping(self):
        x, y, yaw, z = self.get_pose()

        # Only update if moved
        if self.last_pose['x'] is not None:
            dist = math.sqrt((x - self.last_pose['x'])**2 + (y - self.last_pose['y'])**2)
            dyaw = math.atan2(math.sin(yaw - self.last_pose['yaw']), math.cos(yaw - self.last_pose['yaw']))
            if dist < self.MOVE_THRESHOLD and abs(dyaw) < self.YAW_THRESHOLD:
                return
            # Pose-jump gate: normal mapping updates run every ~2cm / ~2.3deg.
            # A far larger step between updates means wheel slip or a recovery
            # jolt — a pose mid-jump is exactly what stamps a whole scan into
            # the map rotated/shifted (thick smeared walls). Resync and skip
            # this one scan; the next clean update maps normally.
            if dist > 0.15 or abs(dyaw) > 0.35:
                self.last_pose.update({'x': x, 'y': y, 'yaw': yaw})
                return

        # Tilt gate: when a wheel rides up a plank edge, the 2D lidar plane
        # sweeps the floor and walls at wrong ranges — one tilted scan paints
        # giant false wedges (both false OCC and false FREE carving). Skip
        # mapping entirely until the robot is level again (~4 deg tolerance).
        rp = self._read_imu_roll_pitch()
        if rp is not None and (abs(rp[0]) > 0.07 or abs(rp[1]) > 0.07):
            return

        self.scan_id += 1
        self.last_pose.update({'x': x, 'y': y, 'yaw': yaw})

        # Ensure the robot's own cell is treated as free for planning.
        ri, rj = self.world_to_grid(x, y)
        self.update_cell(ri, rj, self.L_FREE)

        ranges = self.lidar.getRangeImage()
        fov = self.lidar.getFov()
        N = self.lidar.getHorizontalResolution()
        angle_step = fov / (N - 1)
        
        # Beyond this range a hit is NOT marked as a wall: tangential error is
        # (angular error x distance), so even ~0.5 deg of yaw/mount error smears
        # a 4m hit across 3-4 cells — that is what paints thick fuzzy walls.
        # Within ~2.5m the same wall maps 1-2 cells thin. Distant beams still
        # carve free space (capped), and the wall itself is mapped crisply when
        # the robot gets close on its way there.
        occ_range_cap = float(getattr(self, 'MAPPING_OCC_MAX_RANGE_M', 2.5))

        for k in range(N):
            d = ranges[k]
            if d <= self.lidar.getMinRange() or d >= self.lidar.getMaxRange(): continue

            angle = -( (-fov/2.0) + k * angle_step)

            if d > occ_range_cap:
                # Too far for an accurate wall position: carve free only up to
                # the cap and leave the rest unknown (no OCC marking).
                fx = x + (occ_range_cap - self.HIT_EPS) * math.cos(yaw + angle)
                fy = y + (occ_range_cap - self.HIT_EPS) * math.sin(yaw + angle)
                self.raycast_free(x, y, fx, fy)
                continue

            # Transform to world
            tx = x + d * math.cos(yaw + angle)
            ty = y + d * math.sin(yaw + angle)

            # Hit point (slightly shortened)
            sh_d = max(d - self.HIT_EPS, 0)
            hx = x + sh_d * math.cos(yaw + angle)
            hy = y + sh_d * math.sin(yaw + angle)

            # Mark the beam path as FREE up to just before the hit.
            # IMPORTANT: do NOT mark the shortened endpoint cell as OCC;
            # that cell is still in free space and would falsely block corridors.
            self.raycast_free(x, y, hx, hy)

            # Mark the actual hit cell as OCC.
            oi, oj = self.world_to_grid(tx, ty)
            self.update_cell(oi, oj, self.L_OCC)

    def _min_lidar_distance_in_front(self):
        """Return minimum valid lidar distance in a front sector (meters).

        If no valid readings exist, returns None.
        """
        if not self.lidar:
            return None

        ranges = self.lidar.getRangeImage()
        if not ranges:
            return None

        fov = self.lidar.getFov()
        n = self.lidar.getHorizontalResolution()
        if n <= 1 or fov <= 0:
            return None

        # Center index corresponds to ~0 radians in our mapping convention.
        mid = n // 2
        half_width = max(1, int((self.FRONT_SECTOR_HALF_ANGLE_RAD / fov) * n))
        lo = max(0, mid - half_width)
        hi = min(n - 1, mid + half_width)

        min_r = None
        rmin = self.lidar.getMinRange()
        rmax = self.lidar.getMaxRange()

        for k in range(lo, hi + 1):
            d = ranges[k]
            if d <= rmin or d >= rmax:
                continue
            if min_r is None or d < min_r:
                min_r = d

        return min_r

    def _move_to_waypoint(self, goal_i, goal_j, threshold_m=0.18):
        """Low-level controller: drive to ONE grid cell (goal_i, goal_j)."""
        curr_x, curr_y, curr_yaw, _ = self.get_pose()
        goal_x, goal_y = self.grid_to_world_center(goal_i, goal_j)

        dx = goal_x - curr_x
        dy = goal_y - curr_y
        dist = math.hypot(dx, dy)

        target_yaw = math.atan2(dy, dx)
        angle_error = target_yaw - curr_yaw
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

        # --- WAYPOINT REACHED ---
        if dist < threshold_m:
            self.stop()
            return True

        # --- SAFETY: avoid getting stuck scraping walls while turning ---
        # If we are extremely close to an obstacle in front, briefly back up.
        min_front = self._min_lidar_distance_in_front()
        if min_front is not None and min_front < self.SAFETY_STOP_DIST_M:
            if getattr(self, "DEBUG_MOTION", False):
                self._debug_motion_step += 1
                if self._debug_motion_step % 20 == 0:
                    print(f"[MOTION] SAFETY_STOP min_front={min_front:.3f}m < {self.SAFETY_STOP_DIST_M:.3f}m -> backing up")
            vL = -self.RECOVERY_BACKUP_SPEED
            vR = -self.RECOVERY_BACKUP_SPEED
            for idx, motor in enumerate(self.motors):
                motor.setVelocity(vL if idx % 2 == 0 else vR)
            return False

        # Motion-controller gains.
        k_lin = float(getattr(self, "K_LINEAR", 14.0))
        k_ang = float(getattr(self, "K_ANGULAR", 4.0))

        # --- TURN-IN-PLACE IF MISALIGNED ---
        # Only turn in place for large angle errors; otherwise drive while turning
        if abs(angle_error) > 0.8:
            # Before turning in place at corners, move forward a little, then turn.
            if not hasattr(self, "_preturn_steps_left"):
                self._preturn_steps_left = 0
            if not hasattr(self, "_was_turning_in_place"):
                self._was_turning_in_place = False

            # The pre-turn forward nudge helps swing wide of corner posts, but
            # nudging with a wall right ahead just closes the last clearance and
            # triggers the back-up reflex — the stuck-jitter-at-wall loop.
            clear_ahead = (min_front is None or min_front > 0.30)

            if not self._was_turning_in_place:
                self._was_turning_in_place = True
                preturn_time_s = 0.15
                steps = int((preturn_time_s * 1000.0) / max(1.0, float(self.timestep)))
                self._preturn_steps_left = max(1, steps) if clear_ahead else 0

            if self._preturn_steps_left > 0:
                self._preturn_steps_left -= 1
                v = 1.0
                w = 0.0
            else:
                # Forward creep while realigning is only safe with room ahead.
                v = 0.5 if clear_ahead else 0.0
                w = k_ang * angle_error
        else:
            if hasattr(self, "_was_turning_in_place"):
                self._was_turning_in_place = False
            if hasattr(self, "_preturn_steps_left"):
                self._preturn_steps_left = 0
            # Scale speed by heading alignment: full speed facing the target,
            # strongly reduced near the turn-in-place threshold. Driving at full
            # speed while 30-45 deg misaligned arcs the body into walls — the
            # main corner-clipping mechanism in narrow corridors.
            align = 1.0 - 0.7 * min(1.0, abs(angle_error) / 0.8)
            v = k_lin * dist * align
            # Cruise floor: k_lin*dist collapses as each waypoint nears, which
            # made whole runs crawl. When roughly aligned with room ahead,
            # keep a brisk minimum speed.
            if abs(angle_error) < 0.35 and (
                min_front is None or min_front > self.SAFETY_SLOW_DIST_M
            ):
                v = max(v, float(getattr(self, "CRUISE_SPEED", 6.0)))
            w = k_ang * angle_error

        # If we're close to obstacles, reduce angular speed to avoid clipping corners.
        w_limit = self.MAX_W
        if min_front is not None and min_front < self.SAFETY_SLOW_DIST_M:
            w_limit = self.MAX_W_CLOSE
            # Also cap forward motion a bit when close to a wall.
            v = min(v, 4.0)

        w = max(min(w, w_limit), -w_limit)
        v = max(min(v, self.MAX_V), 0.0)

        vL = v - w
        vR = v + w

        # v and w may individually be within their limits while v+w is not.
        # Scale both wheel commands proportionally so steering curvature is preserved.
        wheel_limit = float(
            getattr(self, "MOTOR_MAX_V", getattr(self, "MAX_V", 10.0))
        )

        peak = max(abs(vL), abs(vR))
        if peak > wheel_limit and peak > 0.0:
            scale = wheel_limit / peak
            vL *= scale
            vR *= scale

        for idx, motor in enumerate(self.motors):
            motor.setVelocity(vL if idx % 2 == 0 else vR)

        return False

    def _direct_drive_toward(self, goal_cell):
        """Blind-drive fallback toward a goal cell — but never THROUGH a mapped
        obstacle. The lidar safety layer cannot see floating walls, so if the
        straight line to the goal crosses a confirmed OCC cell (e.g. locked
        low-wall cells), stop and wait for the rate-limited replan instead of
        charging into it. Unknown and merely-inflated cells stay allowed:
        healing stale map cells at close range is this fallback's purpose.
        """
        rx, ry, _, _ = self.get_pose()
        ri, rj = self.world_to_grid(rx, ry)
        if self._line_of_sight_clear((ri, rj), (int(goal_cell[0]), int(goal_cell[1]))):
            self._move_to_waypoint(goal_cell[0], goal_cell[1])
        else:
            self.stop()

    def spin_in_place(self, speed=2.0):
        """Rotate in place (call every control step while you want to spin)."""
        vL, vR = -speed, speed  # left backward, right forward
        for idx, motor in enumerate(self.motors):
            motor.setVelocity(vL if idx % 2 == 0 else vR)

    def goal_key(self, goal):
        """Normalize goal to a hashable (i, j) tuple."""
        if goal is None:
            return None
        return (int(goal[0]), int(goal[1]))

    def mark_goal_visited(self, goal):
        """Store a goal as visited."""
        key = self.goal_key(goal)
        if key is not None:
            self.visited_goals.add(key)
            self.failed_goal_counts.pop(key, None)

    def is_goal_visited(self, goal):
        """Check if goal was already visited."""
        key = self.goal_key(goal)
        return key in self.visited_goals if key is not None else False
    
    def closest_target(self, targets):
            """
            Pick the closest (i, j) target from a list of grid cells.
            Returns (i, j) or None if targets is empty.
            """
            if not targets:
                return None

            rx, ry, _, _ = self.get_pose()
            ri, rj = self.world_to_grid(rx, ry)

            # minimize squared distance in grid space
            return min(targets, key=lambda t: (t[0] - ri) ** 2 + (t[1] - rj) ** 2)

    def stop(self):
        for motor in self.motors:
            motor.setVelocity(0.0)

    def _compute_hard_blocked(self):
        """Return a cached hard-inflated blocked grid for the current scan_id."""
        if self._hard_blocked_cache_scan_id == self.scan_id and self._hard_blocked_cache is not None:
            return self._hard_blocked_cache

        r = self.HARD_INFLATION_RADIUS_CELLS
        r2 = r * r
        blocked = [[False for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        for j in range(self.GRID_SIZE):
            for i in range(self.GRID_SIZE):
                if self.display_state[j][i] != self.OCC:
                    continue
                for dj in range(-r, r + 1):
                    for di in range(-r, r + 1):
                        if di * di + dj * dj > r2:
                            continue
                        xi, yj = i + di, j + dj
                        if 0 <= xi < self.GRID_SIZE and 0 <= yj < self.GRID_SIZE:
                            blocked[yj][xi] = True

        self._hard_blocked_cache = blocked
        self._hard_blocked_cache_scan_id = self.scan_id
        return blocked

    def _planning_start_cell(self, start_cell):
        """Snap the planning start OFF the inflation ring.

        Newly locked low-wall / green cells often appear right beside the
        robot, putting the robot's own map cell inside the hard-inflated zone.
        Every BFS/A* then dies at step zero, ALL frontiers and goals report
        "unreachable", and the robot parks forever even though open space is
        centimeters away. The robot itself is physically fine — only its map
        cell is "illegal" — so plan from the nearest traversable FREE cell
        instead (searched outward to ~0.3m). Returns the original cell if no
        traversable cell exists nearby.
        """
        si, sj = int(start_cell[0]), int(start_cell[1])
        if not self.inside_map(si, sj):
            return (si, sj)
        hard_blocked = self._compute_hard_blocked()
        if self.display_state[sj][si] == self.FREE and not hard_blocked[sj][si]:
            return (si, sj)
        for r in range(1, 16):
            best = None
            best_d2 = None
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if abs(di) != r and abs(dj) != r:
                        continue  # ring perimeter only
                    ni, nj = si + di, sj + dj
                    if not self.inside_map(ni, nj):
                        continue
                    if self.display_state[nj][ni] != self.FREE or hard_blocked[nj][ni]:
                        continue
                    # The snapped start must be on the ROBOT'S side of any wall:
                    # pressed against a locked floating plank, the nearest legal
                    # cell can lie on the plank's FAR side, and driving toward a
                    # path that starts there pushes the robot into the plank
                    # (which the lidar safety stop cannot see).
                    if not self._line_of_sight_clear((si, sj), (ni, nj)):
                        continue
                    d2 = di * di + dj * dj
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = (ni, nj)
            if best is not None:
                return best
        return (si, sj)

    def _line_is_clear(self, start_cell, end_cell):
        """Bresenham line check against FREE cells and hard-inflated obstacles."""
        (x0, y0) = start_cell
        (x1, y1) = end_cell

        hard_blocked = self._compute_hard_blocked()

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            if not self.inside_map(x, y):
                return False
            if self.display_state[y][x] != self.FREE:
                return False
            if hard_blocked[y][x]:
                return False

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True


    def astar(self, start, goal):
        """
        A* pathfinding from start=(i,j) to goal=(i,j) on self.display_state.
        Returns list of grid cells [(i0,j0), (i1,j1), ..., goal]
        """
        def heuristic(a, b):
            # Manhattan (L1) distance
            return abs(b[0] - a[0]) + abs(b[1] - a[1])

        # A start cell inside the inflation ring kills the search at step zero
        # even though the robot only needs to roll a few cm to legal space.
        start = self._planning_start_cell(start)

        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
        visited = set()

        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return path

            i, j = current

            # Explore 4-connected neighbors (matches Manhattan heuristic)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj

                # Check bounds
                if ni < 0 or nj < 0 or ni >= self.GRID_SIZE or nj >= self.GRID_SIZE:
                    continue

                # Only plan through known-free space
                if self.display_state[nj][ni] != self.FREE:
                    continue

                # Costmap inflation:
                # - hard radius blocks near obstacles
                # - soft radius adds a cost that biases paths away from walls
                r_hard = self.HARD_INFLATION_RADIUS_CELLS
                r_soft = max(self.SOFT_INFLATION_RADIUS_CELLS, r_hard)
                min_d2 = None

                for ii in range(-r_soft, r_soft + 1):
                    for jj in range(-r_soft, r_soft + 1):
                        xi, yj = ni + ii, nj + jj
                        if 0 <= xi < self.GRID_SIZE and 0 <= yj < self.GRID_SIZE:
                            if self.display_state[yj][xi] == self.OCC:
                                d2 = ii * ii + jj * jj
                                if min_d2 is None or d2 < min_d2:
                                    min_d2 = d2

                soft_penalty = 0.0
                if min_d2 is not None:
                    min_d = math.sqrt(min_d2)

                    if min_d <= r_hard:
                        continue

                    if min_d < r_soft and r_soft > r_hard:
                        # 0 at r_soft, 1 at r_hard (linear falloff)
                        t = (r_soft - min_d) / (r_soft - r_hard)
                        soft_penalty = self.SOFT_INFLATION_WEIGHT * max(0.0, min(1.0, t))

                neighbor = (ni, nj)
                if neighbor in visited:
                    continue

                new_g = g + 1 + soft_penalty
                new_f = new_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))

        return None  # No path found

    def move_to_goal(self, goal_i, goal_j):
        """
        High-level controller: plans with A* on self.display_state and follows waypoints.
        Returns True only when the FINAL goal is reached.
        """
        rx, ry, _, _ = self.get_pose()
        ri, rj = self.world_to_grid(rx, ry)
        final_goal = (int(goal_i), int(goal_j))

        # If goal is not free (unknown/occupied), don't drive straight into it.
        gi, gj = final_goal
        if not self.inside_map(gi, gj) or self.display_state[gj][gi] != self.FREE:
            st = None
            if self.inside_map(gi, gj):
                st = self.display_state[gj][gi]
            print(f"Goal cell not FREE; skipping goal={final_goal} state={st}")
            self.need_new_goal = True
            self.current_goal = None
            self.stop()
            return False

        # Plan (or re-plan if goal changed / path exhausted)
        if self.path_goal != final_goal or not self.path or self.path_index >= len(self.path):
            self.path = self.astar((ri, rj), final_goal)
            self.path_index = 0
            self.path_goal = final_goal

            # skip the first waypoint if it's the start cell
            if self.path and self.path[0] == (ri, rj):
                self.path_index = 1

        # Visualize the currently planned path (if any)
        self._debug_draw_path_overlay((ri, rj), final_goal, self.path, self.path_index)

        # No path found => request a new goal (and avoid retrying forever)
        if not self.path:
            key = self.goal_key(final_goal)
            self.failed_goal_counts[key] = self.failed_goal_counts.get(key, 0) + 1
            print(f"A*: no path start={(ri, rj)} goal={final_goal} failures={self.failed_goal_counts[key]}")

            # After a few failures, blacklist this goal (treat like visited)
            if self.failed_goal_counts[key] >= 3:
                print(f"A*: blacklisting unreachable goal {final_goal}")
                self.visited_goals.add(key)

            self.need_new_goal = True
            self.current_goal = None
            self.stop()
            return False

        # If mapping updated and next waypoint becomes blocked, re-plan
        if self.path_index < len(self.path):
            wp_i, wp_j = self.path[self.path_index]
            if not self.inside_map(wp_i, wp_j) or self.display_state[wp_j][wp_i] != self.FREE:
                self.path = None
                return False

        # Follow the path (lookahead to reduce corner-cutting near walls)
        if self.path_index < len(self.path):
            best_index = self.path_index
            max_index = min(len(self.path) - 1, self.path_index + self.WAYPOINT_LOOKAHEAD)
            start_cell = (ri, rj)

            for k in range(self.path_index, max_index + 1):
                if self._line_is_clear(start_cell, self.path[k]):
                    best_index = k

            wp_i, wp_j = self.path[best_index]
            # For debugging: record which waypoint we are actually targeting.
            self._debug_last_target_wp = (int(wp_i), int(wp_j))
            self._debug_last_best_index = int(best_index)
            # Never steer at a waypoint that sits behind a mapped wall relative
            # to where the robot ACTUALLY is (the path start may have been
            # snapped away from the robot's blocked cell). Floating planks are
            # invisible to the lidar safety stop, so driving "through" them on
            # the map means physically hitting them. Replan instead.
            if not self._line_of_sight_clear((ri, rj), (int(wp_i), int(wp_j))):
                self.path = None
                return False
            reached_wp = self._move_to_waypoint(wp_i, wp_j)
            if reached_wp:
                self.path_index = max(self.path_index, best_index + 1)

        if self.path_index >= len(self.path):
            self.stop()
            return True

        return False

    # --- CONTROL ---
    def manual_drive(self):
        key = self.keyboard.getKey()
        vL, vR = 0.0, 0.0
        s = 5.0
        if key == ord('W'):
            vL, vR = s, s
        elif key == ord('S'):
            vL, vR = -s, -s
        elif key == ord('A'):
            vL, vR = -s, s
        elif key == ord('D'):
            vL, vR = s, -s

        for idx, motor in enumerate(self.motors):
            motor.setVelocity(vL if idx % 2 == 0 else vR)

    def save_map(self, path="map.pgm"):
        with open(path, "wb") as f:
            f.write(f"P5\n{self.GRID_SIZE} {self.GRID_SIZE}\n255\n".encode())
            for j in range(self.GRID_SIZE - 1, -1, -1):
                for i in range(self.GRID_SIZE):
                    st = self.display_state[j][i]
                    val = 127 if st == self.UNKNOWN else (255 if st == self.FREE else 0)
                    f.write(bytes([val]))
        print("Map Saved:", path)

    def save_inflated_map(self, path="inflated_map.pgm"):
        """Save a debug PGM where inflated obstacle buffer is also black.

        Values: FREE=255, UNKNOWN=127, OCC or within inflation radius of OCC=0.
        """
        r = self.HARD_INFLATION_RADIUS_CELLS
        r2 = r * r
        inflated_blocked = [[False for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        for j in range(self.GRID_SIZE):
            for i in range(self.GRID_SIZE):
                if self.display_state[j][i] != self.OCC:
                    continue
                for dj in range(-r, r + 1):
                    for di in range(-r, r + 1):
                        if di * di + dj * dj > r2:
                            continue
                        xi, yj = i + di, j + dj
                        if 0 <= xi < self.GRID_SIZE and 0 <= yj < self.GRID_SIZE:
                            inflated_blocked[yj][xi] = True

        with open(path, "wb") as f:
            f.write(f"P5\n{self.GRID_SIZE} {self.GRID_SIZE}\n255\n".encode())
            for j in range(self.GRID_SIZE - 1, -1, -1):
                
                for i in range(self.GRID_SIZE):
                    st = self.display_state[j][i]
                    if inflated_blocked[j][i]:
                        val = 0
                    else:
                        val = 127 if st == self.UNKNOWN else (255 if st == self.FREE else 0)
                    f.write(bytes([val]))
        print("Inflated map saved:", path)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    controller = RosbotExplorer()

    try:
        while controller.step() != -1:

            # 1. Mapping ALWAYS runs
            controller.run_mapping()
            # Depth-camera detection of LOW floating walls the lidar can't see.
            controller.maybe_update_floating_walls()

            # Draw camera feed if a CameraDisplay device exists
            controller.update_camera_display()

            # Process camera and get detection info
            detection_info = controller.process_camera()

            # Debug: visualize green detections on the map (overlay only).
            # radius 0.16 matches the one-disk-per-16cm-region dedup inside, so
            # marked disks tile the green patch contiguously — 0.10 left
            # unmarked slivers between disks that paths could thread through
            # (seen: robot crossed green just behind the blue pillar).
            controller.debug_record_green_detection(detection_info, radius_m=0.16)

            # External live occupancy-map viewer.
            controller.update_pygame_map()

            # Green cooldown (prevents repeated green-trigger loops)
            now_t = controller.robot.getTime()
            green_cooldown_active = now_t < float(getattr(controller, "_green_cooldown_until_time", 0.0))

            # --- GREEN EMERGENCY ESCAPE (runs in EVERY mission state) ---
            # The green-scan trigger and floor marking below are both disabled
            # during go_to_blue / go_to_yellow — meaning green was completely
            # ignored exactly while navigating to pillars, which is when the robot
            # drove onto poison. This escape has NO state exclusions: very high
            # green coverage in the bottom camera band only happens right in front
            # of the robot, so back away while turning away from the green's side,
            # then let normal planning resume. An active maneuver, not a freeze —
            # a plain stop would leave green in view and re-trigger forever.
            if now_t < float(getattr(controller, '_green_escape_until_t', 0.0)):
                _turn_g = float(getattr(controller, '_green_escape_turn', 1.0))
                _vL_g = -2.5 - 2.0 * _turn_g
                _vR_g = -2.5 + 2.0 * _turn_g
                for idx, motor in enumerate(controller.motors):
                    motor.setVelocity(_vL_g if idx % 2 == 0 else _vR_g)
                continue
            _gd_esc = controller._get_detection_for_color(detection_info, 'green')
            _esc_fire = False
            if _gd_esc is not None and _gd_esc.get('coverage', 0.0) >= 0.15:
                # Coverage measures apparent SIZE, not distance — a patch can fill
                # 15% of the bottom band from over a meter away, well clear of the
                # robot's actual path, and escaping on that alone made the robot
                # refuse to pass anywhere near green (observed in testing). Gate on
                # the measured distance: escape only for genuine near-contact.
                # When depth can't be read (below the sensor's minimum range —
                # i.e. green truly at the robot's feet), very high coverage is the
                # only available close-range signal, so it fires the escape alone.
                _cx_esc, _cy_esc = _gd_esc['centroid_px']
                _d_esc = controller._get_depth_at(_cx_esc, _cy_esc, window_px=0)
                if _d_esc is not None:
                    _esc_fire = _d_esc < 0.45
                else:
                    _esc_fire = _gd_esc.get('coverage', 0.0) >= 0.35
            if _esc_fire:
                print(f"[Green] ESCAPE — coverage {_gd_esc['coverage']*100:.0f}%"
                      f"{'' if _d_esc is None else f', dist {_d_esc:.2f}m'}, "
                      f"backing away and turning (state={controller.mission_state})")
                controller._green_escape_until_t = now_t + 1.0
                try:
                    _cx_g = float(_gd_esc['centroid_px'][0])
                    _w_g = float(_gd_esc['img_width'])
                    controller._green_escape_turn = -1.0 if _cx_g < (_w_g / 2.0) else 1.0
                except Exception:
                    controller._green_escape_turn = 1.0
                controller.path = None
                controller.path_goal = None
                controller.path_index = 0
                controller.need_new_goal = True
                controller.stop()
                continue

            # --- GREEN SCANNING STATE MACHINE (kept only for safe shutdown) ---
            # Check if green scan is in progress
            if controller.green_scan_state is not None:
                controller.process_green_scan(detection_info)
                controller.run_mapping()  # Keep mapping while scanning
                continue

            # LEGACY green-scan TRIGGER disabled: it deliberately drove the robot
            # TOWARD green to sweep-scan it, which now conflicts with the
            # distance-gated escape (approach → escape → approach loop) and is
            # redundant — debug_record_green_detection() already marks green at
            # its true camera-measured position (<=1.2m) passively as the robot
            # moves, no dedicated approach maneuver needed.
            
            # LEGACY mark_green_floor_obstacles disabled: it blocked a 0.75m cone
            # of FREE floor in front of the ROBOT (not at the green's position!)
            # whenever green appeared low in the view — with the bottom-band green
            # detection that fired constantly, walling off open space and killing
            # valid paths ("Marked 143 cells (0.75m front)" while green was still
            # over a meter away, seen in testing). Marking at the hazard's true
            # camera-measured position is handled by debug_record_green_detection.
            # if (not green_cooldown_active) and controller.mission_state not in ('go_to_blue', 'go_to_yellow'):
            #     controller.mark_green_floor_obstacles(detection_info)

            # === STATE MACHINE FOR COLOR OBJECT MISSION ===
            
            # --- STATE: INITIAL SCAN ---
            if controller.mission_state == 'initial_scan':
                # During scan, check for color objects
                action = controller.handle_color_detection(detection_info)
                
                # If blue detected during scan, immediately go to it
                if action == 'go_to_blue':
                    controller.mission_state = 'go_to_blue'
                    controller.stop()
                    controller.initial_scan_done = True  # End scan early
                    print("Interrupting scan - going to BLUE object!")
                    continue
                
                # Perform the scan rotation
                scan_complete = controller.perform_initial_scan()
                
                if scan_complete:
                    # Scan finished - decide next state
                    if controller.blue_found:
                        controller.mission_state = 'go_to_blue'
                        print("Scan complete. Navigating to BLUE object.")
                    else:
                        controller.mission_state = 'explore'
                        print("Scan complete. No blue found - starting exploration.")
                continue
            
            # --- STATE: GO TO BLUE ---
            if controller.mission_state == 'go_to_blue':
                # Check if we've reached blue. Two ways to confirm arrival:
                # 1) distance from the stored estimate — but if that estimate is a
                #    little off, the robot can be physically AT the pillar while
                #    still >0.35m from the stored point, and this alone never
                #    fires (seen in testing: coverage climbed past 50% with no
                #    arrival). So also:
                # 2) camera evidence — blue filling >25% of the view only happens
                #    right next to the pillar.
                _blue_det_now = controller._get_detection_for_color(detection_info, 'blue')
                # Coverage alone stops too early: the pillar already fills >25%
                # of the centered ROI from ~0.6-0.7m away. Require the lidar to
                # confirm something physically close in front as well.
                _front_d_blue = controller._min_lidar_distance_in_front()
                _blue_cam_close = (_blue_det_now is not None
                                   and _blue_det_now.get('coverage', 0.0) > 0.25
                                   and _front_d_blue is not None and _front_d_blue < 0.45)
                if (controller.reached_color_object(controller.blue_coords, threshold_m=0.35)
                        or _blue_cam_close):
                    controller.blue_reached = True
                    print("BLUE object reached!")
                    
                    # If both pillars are found, go directly to yellow
                    if controller.blue_found and controller.yellow_found:
                        controller.mission_state = 'go_to_yellow'
                        print("Both pillars found and blue reached! Navigating to YELLOW object.")
                    # If only blue found, decide next state based on yellow
                    elif controller.yellow_found:
                        # Try to path to yellow first
                        yellow_goal = controller.get_color_object_goal_free(controller.yellow_coords)
                        if yellow_goal:
                            rx, ry, _, _ = controller.get_pose()
                            ri, rj = controller.world_to_grid(rx, ry)
                            direct_path = controller.astar((ri, rj), yellow_goal)
                            
                            if direct_path:
                                controller.mission_state = 'go_to_yellow'
                                print("Yellow found and path available. Navigating to YELLOW object.")
                            else:
                                controller.mission_state = 'explore'
                                print("Yellow found but no path available. Starting exploration.")
                        else:
                            controller.mission_state = 'explore'
                            print("Yellow found but invalid goal. Starting exploration.")
                    else:
                        controller.mission_state = 'explore'
                        print("Yellow not yet found. Continuing exploration.")
                    
                    controller.current_goal = None
                    controller.need_new_goal = True
                    controller.stop()
                    continue

                # LEGACY mark_green_poison_zone(radius_m=1.1) disabled: it blocked a
                # 1.1-METER radius around every green estimate while driving to
                # blue. With a green patch flanking the pillar's only open
                # approach, that giant disk sealed the single remaining corridor
                # and made blue unreachable (seen directly in testing — and the
                # same call was removed from a sister version of this controller
                # for exactly the same reason). The true-position green marking
                # (0.10m disks at <=1.2m range) already keeps paths off poison.

                # Refine the stored blue estimate while approaching. The first
                # sighting may be from up to 2.5m away, where a few degrees of
                # bearing error (plus pose drift) displaces the goal by >0.3m —
                # the robot then parks at the wrong spot and the 0.35m arrival
                # check never fires. Close-range estimates are much more
                # accurate, so re-estimate at most every 1.5s and move the goal
                # only when it shifts meaningfully.
                if (_blue_det_now is not None
                        and _blue_det_now.get('coverage', 0.0) >= 0.05
                        and now_t >= getattr(controller, '_blue_refine_t', 0.0)):
                    controller._blue_refine_t = now_t + 1.5
                    _cxr, _cyr = _blue_det_now['centroid_px']
                    _d_r = controller._get_depth_at(_cxr, _cyr, window_px=0)
                    if _d_r is not None and _d_r <= 2.0:
                        _est = controller.estimate_object_world_coords(
                            _blue_det_now, standoff_m=0.0, depth_override=_d_r, use_lidar=False)
                        if _est is not None:
                            _rx0, _ry0, _, _ = controller.get_pose()
                            _ddx, _ddy = _rx0 - _est[0], _ry0 - _est[1]
                            _dd = math.hypot(_ddx, _ddy)
                            if _dd > 0.01:
                                _so = min(float(getattr(controller, 'BLUE_NAV_STANDOFF_M', 0.10)),
                                          max(0.0, _dd - 0.05))
                                _new_nav = (_est[0] + (_ddx / _dd) * _so, _est[1] + (_ddy / _dd) * _so)
                            else:
                                _new_nav = _est
                            if (controller.blue_coords is None
                                    or math.hypot(_new_nav[0] - controller.blue_coords[0],
                                                  _new_nav[1] - controller.blue_coords[1]) > 0.12):
                                print(f"BLUE estimate refined: nav goal moved to ({_new_nav[0]:.2f},{_new_nav[1]:.2f})")
                                controller.blue_coords = _new_nav
                                controller.path = None      # force replan to the refined goal
                                controller.current_goal = None

                # If we already have a valid path, just follow it — do NOT
                # recompute/re-snap the goal every frame: with the map flickering
                # near the pillar, the snap alternated between two equidistant
                # FREE cells each frame, and every flip invalidated the path,
                # leaving the robot twitching in place instead of driving.
                if (controller.path and controller.current_goal
                        and controller.path_index < len(controller.path)):
                    controller.move_to_goal(*controller.current_goal)
                    continue

                # Need a (new) goal: snap once, keep it while the path lasts.
                blue_goal = controller.get_color_object_goal_free(controller.blue_coords)
                if not blue_goal:
                    blue_goal = controller.get_color_object_goal(controller.blue_coords)
                if not blue_goal:
                    # Keep trying to navigate to blue once coordinates exist.
                    controller.stop()
                    continue

                # Only compute path if we don't have one yet or goal changed.
                # Retry-time gate: don't re-run a failing (exhaustive) A* every
                # 32ms step while the direct-drive fallback is active.
                if (controller.current_goal != blue_goal or not controller.path) and (
                        now_t >= getattr(controller, '_blue_plan_retry_t', 0.0)):
                    rx, ry, _, _ = controller.get_pose()
                    ri, rj = controller.world_to_grid(rx, ry)

                    blue_path = controller.astar((ri, rj), blue_goal)
                    if blue_path:
                        controller._blue_plan_fail_count = 0
                        controller.current_goal = blue_goal
                        controller.need_new_goal = False
                        controller.path = blue_path
                        controller.path_goal = blue_goal
                        controller.path_index = 0
                        if controller.path and controller.path[0] == (ri, rj):
                            controller.path_index = 1
                        print(f"Setting goal to BLUE at grid {blue_goal}, path_len={len(blue_path)}")
                    else:
                        # Same fallback ladder as yellow (the blue approach here is
                        # squeezed to one narrow side by the low floating wall and
                        # the green patch): 1) reduced-inflation A* threads gaps
                        # that normal clearance margins reject; 2) direct drive
                        # lets close-range sensing heal stale cells, with the A*
                        # retry rate-limited (a failing search is the most
                        # expensive kind).
                        saved_hard_b = controller.HARD_INFLATION_RADIUS_CELLS
                        controller.HARD_INFLATION_RADIUS_CELLS = controller.MIN_PASSABLE_CLEARANCE_CELLS
                        controller._hard_blocked_cache = None
                        reduced_b = controller.astar((ri, rj), blue_goal)
                        controller.HARD_INFLATION_RADIUS_CELLS = saved_hard_b
                        controller._hard_blocked_cache = None
                        if reduced_b:
                            print(f"[BlueEscape] Reduced-inflation path found, len={len(reduced_b)}")
                            controller.current_goal = blue_goal
                            controller.need_new_goal = False
                            controller.path = reduced_b
                            controller.path_goal = blue_goal
                            controller.path_index = 0
                            if controller.path and controller.path[0] == (ri, rj):
                                controller.path_index = 1
                            continue
                        print("[BlueDirect] No path even at reduced inflation — "
                              "driving toward blue to let close-range sensing clear stale cells")
                        controller.current_goal = None
                        controller.path = None
                        controller._blue_plan_retry_t = now_t + 1.0
                        # Deadlock breaker: when the mapped approach is truly
                        # walled off (e.g. a floating plank now marked solid),
                        # direct drive is line-of-sight-blocked too and the
                        # robot would otherwise park here forever. After
                        # several failed plan cycles, go EXPLORE to map another
                        # approach; the explore state periodically re-plans to
                        # blue and returns the moment a route exists.
                        controller._blue_plan_fail_count = getattr(controller, '_blue_plan_fail_count', 0) + 1
                        if controller._blue_plan_fail_count >= 8:
                            controller._blue_plan_fail_count = 0
                            controller.mission_state = 'explore'
                            controller.need_new_goal = True
                            controller._blue_explore_retry_t = now_t + 5.0
                            print("[BlueFallback] Blue unreachable on current map — exploring to open another approach.")
                            continue
                        controller._direct_drive_toward(blue_goal)
                        continue

                # Move if we have a path
                if controller.path and controller.current_goal:
                    controller.move_to_goal(*controller.current_goal)
                elif now_t < getattr(controller, '_blue_plan_retry_t', 0.0) and blue_goal:
                    controller._direct_drive_toward(blue_goal)
                continue
            
            # --- STATE: EXPLORE ---
            if controller.mission_state == 'explore':
                # While exploring, continuously scan for colors
                action = controller.handle_color_detection(detection_info)

                # Blue is known but was unreachable when last tried (walled off
                # by a floating plank / green zone). Exploration keeps mapping;
                # re-plan to blue every few seconds and return to go_to_blue
                # the moment a route exists.
                if (controller.blue_found and not controller.blue_reached
                        and now_t >= getattr(controller, '_blue_explore_retry_t', float('inf'))):
                    controller._blue_explore_retry_t = now_t + 5.0
                    _bg_r = controller.get_color_object_goal_free(controller.blue_coords)
                    if _bg_r:
                        _rx_r, _ry_r, _, _ = controller.get_pose()
                        _start_r = controller.world_to_grid(_rx_r, _ry_r)
                        _bp_r = controller.astar(_start_r, _bg_r)
                        if _bp_r:
                            controller.mission_state = 'go_to_blue'
                            controller.current_goal = _bg_r
                            controller.need_new_goal = False
                            controller.path = _bp_r
                            controller.path_goal = _bg_r
                            controller.path_index = 1 if _bp_r[0] == _start_r else 0
                            print("Path to BLUE opened during exploration — returning to go_to_blue.")
                            continue
                
                # If blue found during exploration, go to it immediately
                if action == 'go_to_blue' and not controller.blue_reached:
                    controller.mission_state = 'go_to_blue'
                    controller.current_goal = None
                    controller.need_new_goal = True
                    controller.stop()
                    print("Blue detected during exploration - navigating to BLUE!")
                    continue
                
                # CHECK: If both blue and yellow coordinates are found AND we have a path,
                # stop exploration immediately and follow that path.
                # Rate-limit this heavy re-plan. A* to a currently-unreachable
                # pillar explores the whole free map before failing (~seconds
                # of CPU); running it EVERY control step made each Webots step
                # crawl (robot moved in slow bursts). Once every 2s is plenty —
                # frontier exploration keeps the robot moving in between, and
                # the instant a route opens this switches to go_to_blue/yellow.
                if (controller.blue_found and controller.yellow_found
                        and now_t >= getattr(controller, '_bothfound_retry_t', 0.0)):
                    controller._bothfound_retry_t = now_t + 2.0
                    rx, ry, _, _ = controller.get_pose()
                    ri, rj = controller.world_to_grid(rx, ry)

                    if not controller.blue_reached:
                        blue_goal = controller.get_color_object_goal_free(controller.blue_coords)
                        if not blue_goal:
                            blue_goal = controller.get_color_object_goal(controller.blue_coords)
                        if blue_goal:
                            blue_path = controller.astar((ri, rj), blue_goal)
                            if blue_path:
                                controller.mission_state = 'go_to_blue'
                                controller.current_goal = blue_goal
                                controller.need_new_goal = False
                                controller.path = blue_path
                                controller.path_goal = blue_goal
                                controller.path_index = 0
                                if controller.path and controller.path[0] == (ri, rj):
                                    controller.path_index = 1
                                controller.stop()
                                print("Both pillars found and path exists. Stopping exploration. Going to BLUE...")
                                continue
                    else:
                        yellow_plan_coords = controller.yellow_pillar_coords if controller.yellow_pillar_coords else controller.yellow_coords
                        yellow_goal = controller.get_color_object_goal_free(yellow_plan_coords)
                        if not yellow_goal:
                            yellow_goal = controller.get_color_object_goal(yellow_plan_coords)
                        if yellow_goal:
                            yellow_path = controller.astar((ri, rj), yellow_goal)
                            if yellow_path:
                                controller.mission_state = 'go_to_yellow'
                                controller.current_goal = yellow_goal
                                controller.need_new_goal = False
                                controller.path = yellow_path
                                controller.path_goal = yellow_goal
                                controller.path_index = 0
                                if controller.path and controller.path[0] == (ri, rj):
                                    controller.path_index = 1
                                controller.stop()
                                print("Both pillars found and path exists. Stopping exploration. Going to YELLOW...")
                                continue
                
                # Standard frontier exploration
                if controller.current_goal is None or controller.need_new_goal:
                    frontiers = controller.detect_frontiers()

                    if not frontiers:
                        controller.frontier_failure_count += 1
                        print(f"No frontiers found (failure count: {controller.frontier_failure_count})")
                        
                        # CRITICAL: If blue is reached and yellow IS found, go to yellow immediately!
                        if controller.blue_reached and controller.yellow_found:
                            print("No frontiers but yellow is found! Going to YELLOW...")
                            controller.mission_state = 'go_to_yellow'
                            controller.current_goal = None
                            controller.need_new_goal = True
                            controller.path = None
                            continue
                        
                        # If blue is already reached and no frontiers, check if we should just end or keep trying to find yellow
                        if controller.blue_reached and not controller.yellow_found:
                            print("Blue reached but yellow not found. Spinning to search for yellow...")
                            # Keep spinning to try to detect yellow
                            controller.spin_in_place()
                            
                            # After many failures, consider mission complete if yellow can't be found
                            if controller.frontier_failure_count >= 50:
                                print("Exploration complete. Yellow pillar not found after extensive search.")
                                controller.mission_state = 'done'
                                controller.stop()
                            continue
                        
                        # After 10 consecutive failures, clear visited goals to allow revisiting
                        if controller.frontier_failure_count >= 10:
                            print("Clearing visited goals to enable revisiting areas...")
                            controller.visited_goals.clear()
                            controller.frontier_failure_count = 0
                        
                        controller.spin_in_place()
                        continue

                    rx, ry, _, _ = controller.get_pose()
                    ri, rj = controller.world_to_grid(rx, ry)

                    # Pick nearest reachable frontier by BFS (path distance).
                    chosen = controller.find_nearest_reachable_frontier(frontiers, (ri, rj))
                    if chosen is None:
                        # Frontiers exist but are all inside the inflation zone
                        # (narrow gaps, wall ends). Before shrinking inflation
                        # and squeezing in (wall-scrape risk), drive to a safe
                        # nearby cell that can SEE a blocked frontier so the
                        # lidar resolves the unknown space behind it.
                        vp = controller.select_viewpoint_fallback((ri, rj), frontiers)
                        if vp is not None:
                            vp_goal, vp_path = vp
                            controller.current_goal = vp_goal
                            controller.need_new_goal = False
                            controller.path = vp_path
                            controller.path_goal = vp_goal
                            controller.path_index = 0
                            if vp_path and vp_path[0] == (ri, rj):
                                controller.path_index = 1
                            controller.frontier_failure_count = 0
                            continue

                        controller.frontier_failure_count += 1
                        print(f"No reachable unvisited frontier found (failure count: {controller.frontier_failure_count})")

                        # Escape hatch: when the robot is parked close to a thin
                        # wall, its own cell sits INSIDE the hard-inflation ring
                        # and the reachability search dies at step zero — every
                        # frontier looks "unreachable" and clearing visited goals
                        # can't help (reachability, not visitedness, is the
                        # problem — the robot just spins forever). Retrying at
                        # minimal inflation shrinks the ring so the search can
                        # step off the wall and find a real path out.
                        if controller.frontier_failure_count >= 6:
                            print("[Escape] Searching for any reachable frontier at reduced inflation...")
                            saved_hard_e = controller.HARD_INFLATION_RADIUS_CELLS
                            controller.HARD_INFLATION_RADIUS_CELLS = controller.MIN_PASSABLE_CLEARANCE_CELLS
                            controller._hard_blocked_cache = None
                            escape_goal = controller.find_nearest_reachable_frontier(frontiers, (ri, rj))
                            escape_path = controller.astar((ri, rj), escape_goal) if escape_goal is not None else None
                            controller.HARD_INFLATION_RADIUS_CELLS = saved_hard_e
                            controller._hard_blocked_cache = None
                            if escape_path:
                                print(f"[Escape] Reduced-inflation path found to {escape_goal}")
                                controller.current_goal = escape_goal
                                controller.need_new_goal = False
                                controller.path = escape_path
                                controller.path_goal = escape_goal
                                controller.path_index = 0
                                controller.frontier_failure_count = 0
                                continue

                        # After 10 consecutive failures, clear visited goals
                        if controller.frontier_failure_count >= 10:
                            print("Clearing visited goals to enable revisiting areas...")
                            controller.visited_goals.clear()
                            controller.frontier_failure_count = 0

                        controller.spin_in_place()
                        continue

                    # Plan once here so we can reject goals that still fail A*.
                    chosen_path = controller.astar((ri, rj), chosen)
                    if not chosen_path:
                        controller.frontier_failure_count += 1
                        print(f"Chosen frontier failed A* (failure count: {controller.frontier_failure_count})")
                        
                        # Mark this frontier as visited so we don't try it again
                        controller.mark_goal_visited(chosen)
                        
                        # After 10 consecutive failures, clear visited goals
                        if controller.frontier_failure_count >= 10:
                            print("Clearing visited goals to enable revisiting areas...")
                            controller.visited_goals.clear()
                            controller.frontier_failure_count = 0
                        
                        controller.spin_in_place()
                        continue

                    # Successfully found a valid frontier - reset failure counter
                    controller.frontier_failure_count = 0
                    controller.current_goal = chosen
                    controller.need_new_goal = False

                    # Reuse the path we already computed to avoid replanning immediately.
                    controller.path = chosen_path
                    controller.path_goal = chosen
                    controller.path_index = 0
                    if controller.path and controller.path[0] == (ri, rj):
                        controller.path_index = 1

                    print("New exploration goal selected:", controller.current_goal)
                    gi, gj = controller.current_goal
                    gx, gy = controller.grid_to_world_center(gi, gj)
                    print(f"Goal grid=({gi},{gj}) world=({gx:.2f},{gy:.2f})")

                    # If we picked a goal that's already reached, handle it immediately.
                    if controller.goal_reached(controller.current_goal):
                        print("Goal reached!")
                        controller.mark_goal_visited(controller.current_goal)
                        controller.current_goal = None
                        controller.need_new_goal = True
                        controller.stop()
                        continue

                # --- MOVE ---
                if controller.current_goal:
                    rx, ry, _, _ = controller.get_pose()
                    gi, gj = controller.current_goal
                    gx, gy = controller.grid_to_world_center(gi, gj)
                    d = math.hypot(gx - rx, gy - ry)
                    print(f"Robot=({rx:.2f},{ry:.2f}) dist_to_goal={d:.2f}m")
                    controller.move_to_goal(*controller.current_goal)

                    # --- CHECK ---
                    if controller.goal_reached(controller.current_goal):
                        print("Goal reached!")
                        controller.mark_goal_visited(controller.current_goal)
                        controller.current_goal = None
                        controller.need_new_goal = True
                        controller.stop()
                continue
            
            # --- STATE: GO TO YELLOW ---
            if controller.mission_state == 'go_to_yellow':
                # Check if we've reached yellow (same dual check as blue: stored-
                # estimate distance OR camera filled with yellow at close range).
                yellow_reach_coords = controller.yellow_pillar_coords if controller.yellow_pillar_coords else controller.yellow_coords
                _yel_det_now = controller._get_detection_for_color(detection_info, 'yellow')
                _yel_cam_close = (_yel_det_now is not None
                                  and _yel_det_now.get('coverage', 0.0) > 0.25)
                if (controller.reached_color_object(yellow_reach_coords, threshold_m=0.35)
                        or _yel_cam_close):
                    controller.yellow_reached = True
                    controller.mission_state = 'done'
                    print("YELLOW object reached! Mission complete!")
                    controller.stop()
                    continue

                # While navigating to the yellow object, treat green as poison and block
                # a larger area around it (forces replanning away from green).
                controller.mark_green_poison_zone(detection_info, radius_m=1.0)

                # Navigate to yellow.
                if not controller.yellow_coords:
                    print("Cannot navigate to yellow - invalid goal")
                    controller.mission_state = 'done'
                    continue

                # If we already have a valid path to yellow, just follow it - no replanning!
                if controller.path and controller.current_goal and controller.path_index < len(controller.path):
                    # Just move - no A* call
                    controller.move_to_goal(*controller.current_goal)
                    continue

                # Use the yellow coordinate's own grid cell directly (no nearby FREE snapping).
                # Prefer the more accurate pillar coords if available.
                yellow_plan_coords = controller.yellow_pillar_coords if controller.yellow_pillar_coords else controller.yellow_coords
                
                # Try to get a FREE goal near the yellow pillar
                yellow_goal = controller.get_color_object_goal_free(yellow_plan_coords, blacklist=controller._yellow_goal_blacklist)
                if not yellow_goal:
                    # Fall back to direct grid conversion
                    yellow_goal = controller.get_color_object_goal(yellow_plan_coords)
                
                if not yellow_goal:
                    print("Cannot navigate near yellow - no nearby FREE goal yet")
                    controller.mission_state = 'explore'
                    controller.current_goal = None
                    controller.need_new_goal = True
                    continue

                rx, ry, _, _ = controller.get_pose()
                ri, rj = controller.world_to_grid(rx, ry)

                # Only plan once when we need a new path. The retry-time gate keeps
                # a FAILING A* (the most expensive kind — it exhausts the whole
                # reachable region before giving up) from re-running every 32ms
                # step while the direct-drive fallback below is active.
                if (controller.current_goal != yellow_goal or not controller.path) and (
                        now_t >= getattr(controller, '_yellow_plan_retry_t', 0.0)):
                    direct_path = controller.astar((ri, rj), yellow_goal)
                    if direct_path:
                        controller._yellow_plan_fail_count = 0
                        controller.current_goal = yellow_goal
                        controller.need_new_goal = False
                        controller.path = direct_path
                        controller.path_goal = yellow_goal
                        controller.path_index = 0
                        if controller.path and controller.path[0] == (ri, rj):
                            controller.path_index = 1
                        print(f"Setting goal to YELLOW at grid {yellow_goal}, path_len={len(direct_path)}")
                    else:
                        # Fallback ladder (each step proven necessary in testing):
                        # 1) reduced-inflation A* — squeezes corridors that normal
                        #    clearance margins reject (e.g. the narrow doorway
                        #    under the tall floating wall, flanked by green marks);
                        saved_hard_y = controller.HARD_INFLATION_RADIUS_CELLS
                        controller.HARD_INFLATION_RADIUS_CELLS = controller.MIN_PASSABLE_CLEARANCE_CELLS
                        controller._hard_blocked_cache = None
                        reduced_y = controller.astar((ri, rj), yellow_goal)
                        controller.HARD_INFLATION_RADIUS_CELLS = saved_hard_y
                        controller._hard_blocked_cache = None
                        if reduced_y:
                            print(f"[YellowEscape] Reduced-inflation path found, len={len(reduced_y)}")
                            controller.current_goal = yellow_goal
                            controller.need_new_goal = False
                            controller.path = reduced_y
                            controller.path_goal = yellow_goal
                            controller.path_index = 0
                            if controller.path and controller.path[0] == (ri, rj):
                                controller.path_index = 1
                            continue
                        # 2) direct drive — when even that fails, the corridor is
                        #    blocked by STALE map cells (drift-shifted walls). The
                        #    close-range corrections that erase them (lidar free-
                        #    space raycasts) only run near the corridor, and A*
                        #    refuses to take the robot there: a chicken-and-egg.
                        #    Drive straight toward yellow; the waypoint controller's
                        #    own lidar safety protects, the map heals on approach,
                        #    and the rate-limited A* retry above takes over.
                        print("[YellowDirect] No path even at reduced inflation — "
                              "driving toward yellow to let close-range sensing clear stale cells")
                        controller.current_goal = None
                        controller.path = None
                        controller._yellow_plan_retry_t = now_t + 1.0
                        # Deadlock breaker (same as blue): if yellow stays
                        # unreachable on the current map (its region not yet
                        # connected to the robot's by FREE cells), direct drive
                        # is line-of-sight-blocked and the robot would park
                        # forever. After several failed cycles, go EXPLORE to
                        # map the connection; the explore both-found block
                        # re-plans to yellow and returns here once a route opens.
                        controller._yellow_plan_fail_count = getattr(controller, '_yellow_plan_fail_count', 0) + 1
                        if controller._yellow_plan_fail_count >= 8:
                            controller._yellow_plan_fail_count = 0
                            controller.mission_state = 'explore'
                            controller.need_new_goal = True
                            controller._bothfound_retry_t = now_t + 5.0
                            print("[YellowFallback] Yellow unreachable on current map — exploring to open a route.")
                            continue
                        controller._direct_drive_toward(yellow_goal)
                        continue

                # Move if we have a path
                if controller.path and controller.current_goal:
                    controller.move_to_goal(*controller.current_goal)
                elif now_t < getattr(controller, '_yellow_plan_retry_t', 0.0) and yellow_goal:
                    # Between rate-limited replan attempts: keep direct-driving
                    # toward yellow so close-range sensing keeps healing the map.
                    controller._direct_drive_toward(yellow_goal)
                else:
                    # No path available - try to explore more to find a route
                    print("No path to yellow - exploring to find a route...")
                    controller.mission_state = 'explore'
                    controller.need_new_goal = True
                continue

            # --- STATE: DONE ---
            if controller.mission_state == 'done':
                controller.stop()
                print("Mission complete. Blue and Yellow objects found and reached.")
                # Optionally continue exploring or just idle
                continue

    except KeyboardInterrupt:
        pass
    finally:
        controller.update_pygame_map(force=True)
        controller.close_pygame_map()
        controller.save_map()
        controller.save_inflated_map()
        print("\n=== MISSION SUMMARY ===")
        print(f"Blue found: {controller.blue_found}, coords: {controller.blue_coords}, reached: {controller.blue_reached}")
        print(f"Yellow found: {controller.yellow_found}, coords: {controller.yellow_coords}, reached: {controller.yellow_reached}")
        print(f"Final state: {controller.mission_state}")

