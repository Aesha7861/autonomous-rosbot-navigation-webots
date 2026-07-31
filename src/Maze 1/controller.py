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
        self.MAP_SIZE = 7.0
        self.MAP_RES = 0.02
        self.GRID_SIZE = int(self.MAP_SIZE / self.MAP_RES)
        self.ORIGIN_X = -self.MAP_SIZE / 2
        self.ORIGIN_Y = -self.MAP_SIZE / 2
        
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

        # Confirmation thresholds for converting log-odds -> discrete display_state.
        # FREE should update quickly for planning, but OCC should be more conservative
        # to avoid spurious occupied speckles between sparse lidar rays.
        self.OCC_CONFIRM_TH = 2
        self.FREE_CONFIRM_TH = 1

        # Planning: costmap-style inflation
        # - Hard inflation: cells within this radius of an obstacle are forbidden
        # - Soft inflation: cells within this larger radius are allowed but get extra cost
        # Tune these if you're scraping walls or refusing corridors.
        self.HARD_INFLATION_RADIUS_M = 0.13
        self.SOFT_INFLATION_RADIUS_M = 0.21
        self.HARD_INFLATION_RADIUS_CELLS = int(math.ceil(self.HARD_INFLATION_RADIUS_M / self.MAP_RES))
        self.SOFT_INFLATION_RADIUS_CELLS = int(math.ceil(self.SOFT_INFLATION_RADIUS_M / self.MAP_RES))
        self.SOFT_INFLATION_WEIGHT = 35.0
        # Backwards-compat: old code reads this
        self.INFLATION_RADIUS_CELLS = self.HARD_INFLATION_RADIUS_CELLS

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
        # Translation is estimated from wheel encoders; yaw uses IMU if available.
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

        # Frontier selection when stuck:
        # If we keep failing to pick/plan to a frontier, deliberately choose a FARTHER
        # reachable frontier to break local loops.
        self.FAR_FRONTIER_ON_STUCK = True
        self.FAR_FRONTIER_FAILURE_THRESHOLD = 5

        # Waypoint-following: look ahead along the path and drive to the farthest
        # collision-free cell. This reduces corner-cutting when turning near wall ends.
        # With MAP_RES=0.02, a lookahead of 1 makes the robot crawl cell-by-cell.
        self.WAYPOINT_LOOKAHEAD = 1
        self._hard_blocked_cache = None
        self._hard_blocked_cache_scan_id = None

        # Local safety (anti-stuck while turning near walls)
        # These are simple reactive checks using the lidar front sector.
        self.FRONT_SECTOR_HALF_ANGLE_RAD = 0.35  # ~20 deg
        self.SAFETY_STOP_DIST_M = 0.12          # if closer than this, back off
        self.SAFETY_SLOW_DIST_M = 0.25          # if closer than this, slow turning
        self.RECOVERY_BACKUP_SPEED = 2.4        # wheel-speed units used by this controller
        self.MAX_W = 4.8
        self.MAX_W_CLOSE = 3.2
        self.MAX_V = 10.0

        # Camera ROI for color detection/visualization (fraction of image kept, centered)
        self.CAMERA_ROI_FRAC = 0.4
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

        # Blue behavior:
        # - Start moving toward blue as soon as it is faintly visible.
        # - Only lock/record world coordinates when coverage is high.
        self.BLUE_SEEK_COVERAGE_THRESHOLD = 0.50  # 0.2% (tiny speck)
        self.BLUE_RECORD_COVERAGE_THRESHOLD = 0.60# 10%

        # When blue is detected but not yet locked, bias frontier exploration toward it.
        self.blue_hint_goal_cell = None
        self.BLUE_HINT_TTL_S = 10.0
        self._blue_hint_until_time = 0.0
        
        self.yellow_found = False
        # yellow_pillar_coords: estimated pillar position
        # yellow_coords: navigation target (a small standoff in front of the pillar)
        self.yellow_pillar_coords = None  # (world_x, world_y)
        self.yellow_coords = None         # (world_x, world_y)
        self.yellow_reached = False

        # Yellow behavior (mirrors blue):
        # - Start moving/biasing toward yellow as soon as it is faintly visible.
        # - Only lock/record world coordinates when coverage is high.
        self.YELLOW_SEEK_COVERAGE_THRESHOLD = 0.002  # 0.2% (tiny speck)
        self.YELLOW_RECORD_COVERAGE_THRESHOLD = 0.10 # 10%

        # When yellow is detected but not yet locked, bias frontier exploration toward it.
        self.yellow_hint_goal_cell = None
        self.YELLOW_HINT_TTL_S = 10.0
        self._yellow_hint_until_time = 0.0

        # --- Yellow lock-in smoothing (reduce run-to-run variance) ---
        # Only save yellow coordinates after several consistent detections.
        self.YELLOW_LOCK_SAMPLES = 7
        self.YELLOW_LOCK_MAX_SPREAD_M = 0.35
        self._yellow_lock_samples = deque(maxlen=self.YELLOW_LOCK_SAMPLES)
        self._yellow_lock_debug_step = 0
        
        # Track green floor regions that have been marked as obstacles
        self.marked_green_regions = set()  # stores (gi, gj) center cells of marked regions

        # Red boundary behavior:
        # - When red dominates the ROI (>= threshold), we assume we're facing the boundary
        #   and should turn around + replan away from it.
        # - Additionally, we can *lock* (force OCC) the cells between the robot and the
        #   detected red wall, starting a configurable distance away from the wall.
        self.RED_AVOID_COVERAGE_THRESHOLD = 0.75  # 100% coverage to trigger avoidance
        self.RED_AVOID_TTL_S = 8.0
        self.RED_COOLDOWN_S = 6.0
        self._red_cooldown_until_time = 50.0
        self.red_avoid_state = None  # States: None, 'reverse', 'turn'
        self.red_avoid_target_yaw = None
        self._red_forbidden_dir_world = None  # unit vector (dx, dy)
        self._red_avoid_bias_until_time = 0.0
        self.RED_AVOID_REVERSE_DISTANCE_M = 0.15  # Move backward this much before turning
        self._red_avoid_reverse_start_pos = None  # Track start position for reverse

        # --- Red-front blocking (forced OCC corridor between wall and robot) ---
        # Enable/disable stamping of occupied cells in front of detected red.
        self.RED_FRONT_BLOCK_ENABLED = True
        # Only block when red is the *dominant* ROI detection.
        # This prevents small red speckles from gradually filling the map.
        self.RED_FRONT_BLOCK_REQUIRE_DOMINANT = True
        # Minimum red coverage in ROI to apply the block (avoid tiny speckles).
        self.RED_FRONT_BLOCK_MIN_COVERAGE = 0.100
        # Clearance measured from the detected red wall back toward the robot.
        # Example: 0.30 means we start blocking 0.30m in front of the wall.
        # Set to 0.0 to block from the wall all the way toward the robot.
        self.RED_FRONT_BLOCK_FROM_WALL_M = 0.00
        # Don't mark cells extremely close to the robot as occupied.
        self.RED_FRONT_BLOCK_STOP_BEFORE_ROBOT_M = 1.0
        # Lateral width of the blocked corridor (meters).
        self.RED_FRONT_BLOCK_WIDTH_M = 0.30
        # Cap the maximum block length (meters).
        self.RED_FRONT_BLOCK_MAX_RANGE_M = 0.5

        # Rate-limit stamping so we don't do it every control step.
        self.RED_FRONT_BLOCK_COOLDOWN_S = 30
        self._red_front_block_until_time = 0.0

        # --- Depth-camera floating/low wall mapping (Supervisor-free) ---
        # Projects depth-camera pixels into the occupancy grid and locks points
        # whose height lies inside the robot body band. This catches obstacles
        # that the planar lidar can miss, while ignoring high passable overhead walls.
        self.FLOATING_WALL_ENABLED = True
        self.FLOATING_WALL_POLL_EVERY_N_STEPS = 2
        self._floating_wall_step_counter = 0
        self.DEPTH_OBSTACLE_MIN_DIST = 0.15
        self.DEPTH_OBSTACLE_MAX_DIST = 0.80
        self.FLOATING_WALL_MIN_HEIGHT = 0.05
        self.ROBOT_CLEARANCE_HEIGHT = 0.20
        self.DEPTH_PIXEL_STRIDE = 2
        self.FLOATING_WALL_CAMERA_HEIGHT_M = 0.17
        self.FLOATING_WALL_CAMERA_PITCH_RAD = 0.0
        self.FLOATING_WALL_FORWARD_OFFSET_M = 0.08
        self.FLOATING_WALL_DILATION_CELLS = 1
        self.FLOATING_WALL_STOP_BEFORE_ROBOT_M = 0.10
        self.FLOATING_WALL_LOG_ODDS_LOCK = self.L_MAX
        self.depth_wall_cells_marked = 0

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

        # Yellow approach: avoid stopping forever at a single snapped cell
        self._yellow_goal_blacklist = set()

        # Debug: motion/planning traces (throttled)
        self.DEBUG_MOTION = True
        self._debug_motion_step = 0
        self._debug_last_target_wp = None
        self._debug_last_best_index = None

        # --- Pygame live occupancy-map visualization ---
        # This visualizes the occupancy grid generated by lidar/depth/odometry.
        # It does not affect navigation or planning.
        self.ENABLE_PYGAME_MAP = True
        self.PYGAME_UPDATE_EVERY_N_STEPS = 1
        self.PYGAME_CELL_SCALE = 1.5
        self._pygame_enabled = False
        self._pygame_screen = None
        self._pygame_font = None
        self._pygame_step_counter = 0
        self._pygame_last_error_printed = False

        # Open only after all rendering attributes exist.
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

    # --- PYGAME LIVE MAP VISUALIZATION ---
    def init_pygame_map(self):
        """Open a pygame window for the live occupancy map only.

        This viewer intentionally uses direct pygame drawing instead of numpy
        frombuffer/surfarray conversion. Direct drawing is slower, but it is
        much more reliable inside Webots on Ubuntu/Wayland/XWayland, where the
        previous surface-based renderer could open a transparent/blank window.
        """
        if not getattr(self, "ENABLE_PYGAME_MAP", True):
            return
        if pygame is None:
            print("[pygame map] pygame is not installed; live map window disabled.")
            return

        try:
            pygame.init()
            pygame.font.init()
            scale = max(1.0, float(getattr(self, "PYGAME_CELL_SCALE", 1.0)))
            self._pygame_header_h = 54
            width = int(round(self.GRID_SIZE * scale))
            height = int(round(self.GRID_SIZE * scale)) + self._pygame_header_h

            # No RESIZABLE/SCALED/OPENGL flags here. A plain software window is
            # the most reliable option for Webots controller-side visualization.
            self._pygame_screen = pygame.display.set_mode((width, height), 0, 32)
            pygame.display.set_caption("RosBot live occupancy map")
            self._pygame_font = pygame.font.SysFont("Arial", 14)
            self._pygame_big_font = pygame.font.SysFont("Arial", 16, bold=True)
            self._pygame_enabled = True
            self._pygame_last_error_printed = False
            print(f"[pygame map] opened {width}x{height} live occupancy map window")
            self.update_pygame_map(force=True)
        except Exception as exc:
            self._pygame_enabled = False
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

            scale = max(1.0, float(getattr(self, "PYGAME_CELL_SCALE", 1.0)))
            header_h = int(getattr(self, "_pygame_header_h", 54))
            side = int(round(self.GRID_SIZE * scale))

            # Paint a real opaque background every frame. This prevents the
            # transparent-window effect seen with Webots on some desktops.
            self._pygame_screen.fill((20, 20, 20))

            map_surface = self._make_pygame_map_surface()
            if scale != 1:
                map_surface = pygame.transform.scale(map_surface, (side, side))
            self._pygame_screen.blit(map_surface, (0, header_h))
            pygame.draw.rect(self._pygame_screen, (220, 220, 220), pygame.Rect(0, header_h, side, side), 1)

            self._draw_pygame_label("RosBot live occupancy map", 6, 4, big=True)
            self._draw_pygame_label("gray=unknown | white=free | black=obstacle | magenta=robot | green=path", 6, 25)
            self._draw_pygame_label(self._pygame_map_status_text(), 6, 39)

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
        self.update_pygame_map()

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

        coords = self.estimate_object_world_coords(
            green_det,
            standoff_m=0.0,
            depth_override=None,
            use_lidar=True,
        )
        if not coords:
            return

        ci, cj = self.world_to_grid(coords[0], coords[1])
        r_cells = int(max(0, float(radius_m) / float(self.MAP_RES)))
        r2 = r_cells * r_cells

        if r_cells <= 0:
            if self.inside_map(ci, cj):
                self._debug_green_cells.add((ci, cj))
                if getattr(self, 'GREEN_DETECTION_MARK_OCCUPIED', False):
                    self.force_occupy_cell(ci, cj)
            return

        for dj in range(-r_cells, r_cells + 1):
            for di in range(-r_cells, r_cells + 1):
                if di * di + dj * dj > r2:
                    continue
                ni, nj = ci + di, cj + dj
                if self.inside_map(ni, nj):
                    self._debug_green_cells.add((ni, nj))
                    if getattr(self, 'GREEN_DETECTION_MARK_OCCUPIED', False):
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
            return (h >= 80) & (h <= 140) & (s >= 0.4) & (v >= 0.3)
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
            "green": self._mask_color(h_roi, s_roi, v_roi, "green"),
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
                # Compute centroid in ROI coordinates
                ys, xs = np.where(mask)
                cx_roi = np.mean(xs)
                cy_roi = np.mean(ys)
                # Convert to full image coordinates
                centroids[name] = (x0 + cx_roi, y0 + cy_roi)

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

        # If we're already following a planned path, ignore green completely.
        # This prevents green detections from triggering green-scan / poisoning
        # while we are executing an existing A* path.
        try:
            path = getattr(self, 'path', None)
            path_index = int(getattr(self, 'path_index', 0) or 0)
            path_active = isinstance(path, (list, tuple)) and len(path) > 0 and path_index < len(path)
        except Exception:
            path_active = False

        if path_active and detections:
            detections = [d for d in detections if isinstance(d, dict) and d.get('color') != 'green']

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
            'go_to_yellow' if yellow detected and should navigate to it
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
        
        # Blue: begin approaching early, but only record coords when close enough.
        BLUE_SEEK_COVERAGE_THRESHOLD = float(getattr(self, 'BLUE_SEEK_COVERAGE_THRESHOLD', 0.01))
        BLUE_RECORD_COVERAGE_THRESHOLD = float(getattr(self, 'BLUE_RECORD_COVERAGE_THRESHOLD', 0.10))

        # Yellow: mirror blue thresholds.
        YELLOW_SEEK_COVERAGE_THRESHOLD = float(getattr(self, 'YELLOW_SEEK_COVERAGE_THRESHOLD', 0.01))
        YELLOW_RECORD_COVERAGE_THRESHOLD = float(getattr(self, 'YELLOW_RECORD_COVERAGE_THRESHOLD', 0.10))
        
        if color == 'blue' and not self.blue_found:
            # If blue is faintly detected, store a rough hint direction and keep exploring
            # and start approaching to increase coverage.
            if coverage >= BLUE_SEEK_COVERAGE_THRESHOLD and coverage < BLUE_RECORD_COVERAGE_THRESHOLD:
                self.update_blue_hint_goal(detection_info)
                return 'go_to_blue'

            # Only lock/record coordinates when coverage is high enough.
            if coverage >= BLUE_RECORD_COVERAGE_THRESHOLD:
                # Estimate pillar position, then choose a navigation target a bit BEFORE it
                # (toward the robot) to avoid planning to an occupied cell on the pillar.
                coords_pillar = self.estimate_object_world_coords(
                    detection_info,
                    standoff_m=0.0,
                    depth_override=None,
                    use_lidar=True,
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
                print(
                    f"BLUE object detected but coverage too low to record "
                    f"({coverage*100.0:.1f}% < {BLUE_RECORD_COVERAGE_THRESHOLD*100.0:.1f}%), approaching..."
                )
                self.update_blue_hint_goal(detection_info)
                return 'continue'
        
        elif color == 'yellow' and not self.yellow_found:
            # Always store a rough hint when we see even a speck.
            if coverage > 0.0:
                self.update_yellow_hint_goal(detection_info)

            # If yellow is faintly detected, store a rough hint direction and keep exploring.
            if coverage >= YELLOW_SEEK_COVERAGE_THRESHOLD and coverage < YELLOW_RECORD_COVERAGE_THRESHOLD:
                # Only start a dedicated approach to yellow once blue is reached.
                if getattr(self, 'blue_reached', False):
                    return 'go_to_yellow'
                return 'continue'

            # Only lock/record coordinates when coverage is high enough.
            if coverage >= YELLOW_RECORD_COVERAGE_THRESHOLD:
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
                print(
                    f"YELLOW object detected but coverage too low to record "
                    f"({coverage*100.0:.1f}% < {YELLOW_RECORD_COVERAGE_THRESHOLD*100.0:.1f}%), moving toward it..."
                )
        
        return 'continue'

    def update_blue_hint_goal(self, detection_info):
        """Update a rough grid goal in the direction of the blue detection.

        This is used to bias frontier selection toward blue so the robot keeps
        exploring/mapping instead of rotating in place searching.
        """
        if detection_info is None:
            return

        try:
            cx, cy = detection_info['centroid_px']
            img_w = detection_info['img_width']
            img_h = detection_info['img_height']
        except Exception:
            return

        # Bearing from image center
        cx_normalized = (cx - img_w / 2.0) / (img_w / 2.0)  # -1..1
        fov_h = self._camera_fov_h(img_w, img_h)
        angle_offset = -float(cx_normalized) * (float(fov_h) / 2.0)

        # Range estimate: prefer lidar at that bearing.
        depth = self._get_lidar_range_at_angle(angle_offset, window_rays=3)
        if depth is None:
            depth = self._get_depth_at(cx, cy)
        if depth is None:
            depth = 2.0

        # Clamp so the hint doesn't jump outside the map.
        depth = float(max(0.6, min(2.5, float(depth))))

        rx, ry, ryaw, _ = self.get_pose()
        obj_angle = float(ryaw) + float(angle_offset)
        wx = float(rx) + depth * math.cos(obj_angle)
        wy = float(ry) + depth * math.sin(obj_angle)

        gi, gj = self.world_to_grid(wx, wy)
        gi = max(0, min(self.GRID_SIZE - 1, int(gi)))
        gj = max(0, min(self.GRID_SIZE - 1, int(gj)))
        self.blue_hint_goal_cell = (gi, gj)
        try:
            self._blue_hint_until_time = float(self.robot.getTime()) + float(getattr(self, 'BLUE_HINT_TTL_S', 10.0))
        except Exception:
            self._blue_hint_until_time = 0.0

    def update_yellow_hint_goal(self, detection_info):
        """Update a rough grid goal in the direction of the yellow detection.

        This is used to bias frontier selection toward yellow so the robot keeps
        exploring/mapping instead of rotating in place searching.
        """
        if detection_info is None:
            return

        try:
            cx, cy = detection_info['centroid_px']
            img_w = detection_info['img_width']
            img_h = detection_info['img_height']
        except Exception:
            return

        cx_normalized = (cx - img_w / 2.0) / (img_w / 2.0)  # -1..1
        fov_h = self._camera_fov_h(img_w, img_h)
        angle_offset = -float(cx_normalized) * (float(fov_h) / 2.0)

        depth = self._get_lidar_range_at_angle(angle_offset, window_rays=3)
        if depth is None:
            depth = self._get_depth_at(cx, cy)
        if depth is None:
            depth = 2.0

        depth = float(max(0.6, min(2.5, float(depth))))

        rx, ry, ryaw, _ = self.get_pose()
        obj_angle = float(ryaw) + float(angle_offset)
        wx = float(rx) + depth * math.cos(obj_angle)
        wy = float(ry) + depth * math.sin(obj_angle)

        gi, gj = self.world_to_grid(wx, wy)
        gi = max(0, min(self.GRID_SIZE - 1, int(gi)))
        gj = max(0, min(self.GRID_SIZE - 1, int(gj)))
        self.yellow_hint_goal_cell = (gi, gj)
        try:
            self._yellow_hint_until_time = float(self.robot.getTime()) + float(getattr(self, 'YELLOW_HINT_TTL_S', 10.0))
        except Exception:
            self._yellow_hint_until_time = 0.0

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
            forward_distance_m = 2.6  # slightly more forward blocking
            side_width_m = 1.35  # slightly wider to each side
            
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
            front_distance_m = 1.10
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
            radius_cells = 9
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

    def mark_red_front_obstacles(self, detection_info):
        """Lock cells between the detected red wall and the robot as OCC.

        This uses force_occupy_cell() so lidar updates cannot free the cells.
        The start point is adjustable via RED_FRONT_BLOCK_FROM_WALL_M.
        """
        dbg = bool(getattr(self, 'DEBUG_RED_FRONT_BLOCK', False))
        if dbg:
            try:
                top_color = detection_info.get('color') if isinstance(detection_info, dict) else None
            except Exception:
                top_color = None
            print(f"RED FRONT BLOCK DBG: enter (top_color={top_color})")

        if not getattr(self, 'RED_FRONT_BLOCK_ENABLED', False):
            if dbg:
                print('RED FRONT BLOCK DBG: disabled (RED_FRONT_BLOCK_ENABLED=False)')
            return
        if detection_info is None:
            if dbg:
                print('RED FRONT BLOCK DBG: no detection_info')
            return

        # Only apply when red is currently the dominant detection.
        if bool(getattr(self, 'RED_FRONT_BLOCK_REQUIRE_DOMINANT', True)):
            try:
                if not (isinstance(detection_info, dict) and detection_info.get('color') == 'red'):
                    if dbg:
                        try:
                            print(f"RED FRONT BLOCK DBG: not dominant (top_color={detection_info.get('color')})")
                        except Exception:
                            print('RED FRONT BLOCK DBG: not dominant (top_color=<err>)')
                    return
            except Exception:
                if dbg:
                    print('RED FRONT BLOCK DBG: not dominant check errored')
                return

        red_det = self._get_detection_for_color(detection_info, 'red')
        if red_det is None:
            if dbg:
                print('RED FRONT BLOCK DBG: red_det not found in detection_info')
            return

        cov = float(red_det.get('coverage', 0.0) or 0.0)
        min_cov = float(getattr(self, 'RED_FRONT_BLOCK_MIN_COVERAGE', 0.02))
        if cov < min_cov:
            if dbg:
                print(f"RED FRONT BLOCK DBG: coverage too low (cov={cov:.3f} < min={min_cov:.3f})")
            return

        # Cooldown
        try:
            now_t = float(self.robot.getTime())
            until_t = float(getattr(self, '_red_front_block_until_time', 0.0))
            if now_t < until_t:
                if dbg:
                    print(f"RED FRONT BLOCK DBG: cooldown active (now={now_t:.2f} < until={until_t:.2f})")
                return
        except Exception:
            if dbg:
                print('RED FRONT BLOCK DBG: cooldown check errored (continuing)')
            pass

        # Use lidar-only depth for red; depth camera can hit background and cause
        # overly-large blocking that destroys frontier selection.
        try:
            cx, cy = red_det['centroid_px']
            img_w = red_det['img_width']
            img_h = red_det['img_height']
        except Exception:
            if dbg:
                print('RED FRONT BLOCK DBG: missing centroid/img dims in red_det')
            return

        cx_normalized = (float(cx) - float(img_w) / 2.0) / (float(img_w) / 2.0)
        fov_h = self._camera_fov_h(img_w, img_h)
        angle_offset = -float(cx_normalized) * (float(fov_h) / 2.0)

        depth = self._get_lidar_range_at_angle(angle_offset, window_rays=2)
        if depth is None:
            if dbg:
                print('RED FRONT BLOCK DBG: lidar depth at angle is None')
            return

        rx, ry, ryaw, _ = self.get_pose()
        wall_dist = float(depth)
        if wall_dist < 0.25:
            if dbg:
                print(f"RED FRONT BLOCK DBG: wall_dist too small (wall_dist={wall_dist:.2f}m)")
            return

        wx = float(rx) + wall_dist * math.cos(float(ryaw) + float(angle_offset))
        wy = float(ry) + wall_dist * math.sin(float(ryaw) + float(angle_offset))

        vx = wx - float(rx)
        vy = wy - float(ry)

        # Unit direction from robot toward the wall.
        ux = vx / wall_dist
        uy = vy / wall_dist

        from_wall = float(getattr(self, 'RED_FRONT_BLOCK_FROM_WALL_M', 0.30))
        stop_before = float(getattr(self, 'RED_FRONT_BLOCK_STOP_BEFORE_ROBOT_M', 0.12))
        width_m = float(getattr(self, 'RED_FRONT_BLOCK_WIDTH_M', 0.30))
        max_range = float(getattr(self, 'RED_FRONT_BLOCK_MAX_RANGE_M', 3.0))
        cooldown_s = float(getattr(self, 'RED_FRONT_BLOCK_COOLDOWN_S', 0.8))

        from_wall = max(0.0, from_wall)
        stop_before = max(0.0, stop_before)
        width_m = max(0.0, width_m)
        max_range = max(0.3, max_range)

        # We block from a point near the wall (pulled toward robot by from_wall)
        # back toward the robot, but stop_before keeps a small safety bubble.
        t_far = min(max_range, max(stop_before, wall_dist - from_wall))
        t_near = min(t_far, stop_before)
        if t_far <= t_near + 1e-6:
            if dbg:
                print(
                    "RED FRONT BLOCK DBG: degenerate corridor "
                    f"(wall_dist={wall_dist:.2f}, from_wall={from_wall:.2f}, stop_before={stop_before:.2f}, "
                    f"max_range={max_range:.2f} -> t_near={t_near:.2f}, t_far={t_far:.2f})"
                )
            return

        if dbg:
            print(
                "RED FRONT BLOCK DBG: geometry "
                f"(cov={cov:.3f}, wall_dist={wall_dist:.2f}, angle_offset={angle_offset:.2f}rad, "
                f"from_wall={from_wall:.2f}, stop_before={stop_before:.2f}, width={width_m:.2f}, "
                f"max_range={max_range:.2f}, t_near={t_near:.2f}, t_far={t_far:.2f})"
            )

        # Perpendicular unit vector for corridor width.
        px = -uy
        py = ux

        half_w = 0.5 * width_m
        step_along = max(self.MAP_RES, 0.5 * self.MAP_RES)
        step_side = max(self.MAP_RES, 1.0 * self.MAP_RES)

        newly_blocked = set()
        t = float(t_near)
        while t <= float(t_far) + 1e-9:
            cxw = float(rx) + t * ux
            cyw = float(ry) + t * uy

            s = -half_w
            while s <= half_w + 1e-9:
                sxw = cxw + s * px
                syw = cyw + s * py
                ci, cj = self.world_to_grid(sxw, syw)
                if self.inside_map(ci, cj):
                    if (ci, cj) not in getattr(self, 'forced_occupied_cells', set()):
                        self.force_occupy_cell(ci, cj)
                        newly_blocked.add((ci, cj))
                s += step_side

            t += step_along

        if newly_blocked:
            try:
                self._red_front_block_until_time = float(self.robot.getTime()) + max(0.0, float(cooldown_s))
            except Exception:
                pass

            # Invalidate hard-block cache immediately.
            self._hard_blocked_cache = None
            self._hard_blocked_cache_scan_id = None

            # If our current plan goes through the new red block, drop it.
            try:
                path = getattr(self, 'path', None)
                if isinstance(path, (list, tuple)) and any((p in newly_blocked) for p in path):
                    print('RED FRONT BLOCK: current path intersects blocked cells, clearing path')
                    self.path = None
                    self.path_goal = None
                    self.path_index = 0
                    self.need_new_goal = True
            except Exception:
                pass

            print(
                f"RED FRONT BLOCK: forced {len(newly_blocked)} cells OCC; "
                f"wall_dist={wall_dist:.2f}m from_wall={from_wall:.2f}m width={width_m:.2f}m"
            )
        elif dbg:
            print('RED FRONT BLOCK DBG: no cells blocked (newly_blocked empty)')


    def maybe_update_floating_walls(self):
        """Rate-limited wrapper for depth-camera floating-wall mapping."""
        if not getattr(self, 'FLOATING_WALL_ENABLED', False):
            return 0
        self._floating_wall_step_counter = int(getattr(self, '_floating_wall_step_counter', 0)) + 1
        every = max(1, int(getattr(self, 'FLOATING_WALL_POLL_EVERY_N_STEPS', 2)))
        if (self._floating_wall_step_counter % every) != 0:
            return 0
        return self.detect_and_mark_floating_walls()

    def detect_and_mark_floating_walls(self):
        """Mark lidar-invisible floating/low walls from the depth camera.

        The original project used a separate map object. This version writes into
        this controller's log-odds occupancy grid by calling force_occupy_cell().
        Points are kept only if their estimated height is above floor noise and
        below the robot clearance height. Higher overhead walls are ignored.
        """
        if self.camera_depth is None:
            return 0

        try:
            depth_raw = self.camera_depth.getRangeImage()
            if depth_raw is None or len(depth_raw) == 0:
                return 0
            dw = int(self.camera_depth.getWidth())
            dh = int(self.camera_depth.getHeight())
            if dw <= 0 or dh <= 0:
                return 0
            depth_m = np.asarray(depth_raw, dtype=np.float32).reshape((dh, dw))
        except Exception:
            return 0

        try:
            fov = float(self.camera_depth.getFov())
            if fov <= 0.0:
                return 0
            fx = dw / (2.0 * math.tan(fov / 2.0))
            fy = fx
            cx = dw / 2.0
            cy = dh / 2.0
        except Exception:
            return 0

        stride = max(1, int(getattr(self, 'DEPTH_PIXEL_STRIDE', 2)))
        min_dist = float(getattr(self, 'DEPTH_OBSTACLE_MIN_DIST', 0.15))
        max_dist = float(getattr(self, 'DEPTH_OBSTACLE_MAX_DIST', 0.80))
        cam_h = float(getattr(self, 'FLOATING_WALL_CAMERA_HEIGHT_M', 0.17))
        cam_pitch = float(getattr(self, 'FLOATING_WALL_CAMERA_PITCH_RAD', 0.0))
        min_h = float(getattr(self, 'FLOATING_WALL_MIN_HEIGHT', 0.05))
        max_h = float(getattr(self, 'ROBOT_CLEARANCE_HEIGHT', 0.20))
        fwd_off = float(getattr(self, 'FLOATING_WALL_FORWARD_OFFSET_M', 0.08))
        stop_before_robot = float(getattr(self, 'FLOATING_WALL_STOP_BEFORE_ROBOT_M', 0.10))

        v_idx, u_idx = np.indices(depth_m.shape, dtype=np.float32)
        valid = (
            np.isfinite(depth_m) &
            (depth_m > min_dist) &
            (depth_m < max_dist) &
            ((u_idx.astype(np.int32) % stride) == 0) &
            ((v_idx.astype(np.int32) % stride) == 0)
        )
        if not np.any(valid):
            return 0

        D = depth_m[valid]
        u = u_idx[valid]
        v = v_idx[valid]

        # Normalized pixel offsets. Robot frame convention: X forward, Y left, Z up.
        xn = (u - cx) / fx
        yn = (v - cy) / fy
        X_c0 = D
        Y_c0 = -D * xn
        Z_c0 = -D * yn

        # Optional pitch compensation. Positive pitch means camera looks downward.
        cos_p = math.cos(cam_pitch)
        sin_p = math.sin(cam_pitch)
        X_c = cos_p * X_c0 + sin_p * Z_c0
        Z_c = -sin_p * X_c0 + cos_p * Z_c0
        Y_c = Y_c0

        world_height = cam_h + Z_c
        in_body_band = (world_height > min_h) & (world_height < max_h)
        if not np.any(in_body_band):
            return 0

        X_r = X_c[in_body_band] + fwd_off
        Y_r = Y_c[in_body_band]

        # Reject points too close to the robot center to avoid self-marking.
        range_r = np.sqrt(X_r * X_r + Y_r * Y_r)
        keep = range_r > stop_before_robot
        if not np.any(keep):
            return 0
        X_r = X_r[keep]
        Y_r = Y_r[keep]

        rx, ry, ryaw, _ = self.get_pose()
        cos_h = math.cos(ryaw)
        sin_h = math.sin(ryaw)
        Xw = rx + cos_h * X_r - sin_h * Y_r
        Yw = ry + sin_h * X_r + cos_h * Y_r

        cells_to_lock = set()
        dilation = max(0, int(getattr(self, 'FLOATING_WALL_DILATION_CELLS', 1)))
        for wx, wy in zip(Xw, Yw):
            gi, gj = self.world_to_grid(float(wx), float(wy))
            if not self.inside_map(gi, gj):
                continue
            if dilation <= 0:
                cells_to_lock.add((gi, gj))
            else:
                r2 = dilation * dilation
                for dj in range(-dilation, dilation + 1):
                    for di in range(-dilation, dilation + 1):
                        if di * di + dj * dj > r2:
                            continue
                        ni, nj = gi + di, gj + dj
                        if self.inside_map(ni, nj):
                            cells_to_lock.add((ni, nj))

        if not cells_to_lock:
            return 0

        newly_locked = set()
        forced = getattr(self, 'forced_occupied_cells', set())
        for gi, gj in cells_to_lock:
            if (gi, gj) in forced:
                continue
            self.force_occupy_cell(gi, gj)
            newly_locked.add((gi, gj))

        if not newly_locked:
            return 0

        self.depth_wall_cells_marked = int(getattr(self, 'depth_wall_cells_marked', 0)) + len(newly_locked)
        self._hard_blocked_cache = None
        self._hard_blocked_cache_scan_id = None

        # If the current path crosses a newly locked floating-wall cell, force replanning.
        try:
            if self.path and any(cell in newly_locked for cell in self.path):
                print('[FloatingWall] current path intersects locked cells, clearing path')
                self.path = None
                self.path_goal = None
                self.path_index = 0
                self.need_new_goal = True
        except Exception:
            pass

        print(f'[FloatingWall] locked {len(newly_locked)} cells from depth camera')
        self.update_pygame_map(force=True)
        return len(newly_locked)

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

    def _frontier_unknown_neighbor_count(self, i, j, radius_cells=3):
        """Count UNKNOWN cells around a candidate frontier goal.

        Higher counts typically mean the frontier borders a larger not-yet-explored region.
        """
        r = int(max(1, radius_cells))
        cnt = 0
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if di == 0 and dj == 0:
                    continue
                ni = int(i) + int(di)
                nj = int(j) + int(dj)
                if not self.inside_map(ni, nj):
                    continue
                if self.display_state[nj][ni] == self.UNKNOWN:
                    cnt += 1
        return cnt
    
    def frontier_centroid(self,cluster):
        """Return a FREE goal cell representative for this cluster.

        Important: the numeric centroid can land on UNKNOWN/OCC due to rounding.
        We instead pick the frontier cell closest to the centroid, guaranteeing FREE.
        """
        cx = sum(c[0] for c in cluster) / len(cluster)
        cy = sum(c[1] for c in cluster) / len(cluster)
        unk_r = int(getattr(self, 'FRONTIER_UNKNOWN_RADIUS_CELLS', 3))
        # Prefer a representative in the part of the cluster that borders more UNKNOWN,
        # tie-breaking by closeness to the numeric centroid.
        return max(
            cluster,
            key=lambda c: (
                self._frontier_unknown_neighbor_count(c[0], c[1], radius_cells=unk_r),
                -((c[0] - cx) ** 2 + (c[1] - cy) ** 2),
            ),
        )

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

        frontier_set = set((int(i), int(j)) for (i, j) in frontiers)
        hard_blocked = self._compute_hard_blocked()

        visited = [[False for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        visited[sj][si] = True

        # 4-connected BFS matches our A* connectivity.
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        # Preference for frontiers that border more UNKNOWN (unexplored) cells.
        unk_r = int(getattr(self, 'FRONTIER_UNKNOWN_RADIUS_CELLS', 3))
        unk_bonus = float(getattr(self, 'FRONTIER_UNKNOWN_BONUS', 1.4))
        max_unk = (2 * unk_r + 1) ** 2 - 1
        max_bonus = unk_bonus * float(max_unk)

        best = None
        best_score = None

        # Track BFS distance explicitly.
        q = deque()
        q.append((si, sj, 0))

        while q:
            i, j, d = q.popleft()

            # Early stop: at larger BFS layers the distance term dominates, so
            # if even the best possible UNKNOWN bonus can't beat best_score, stop.
            if best_score is not None and (float(d) - max_bonus) > float(best_score):
                break

            if (i, j) in frontier_set and self.display_state[j][i] == self.FREE:
                # Don't consider a goal in hard-inflated space or already visited.
                if not hard_blocked[j][i] and not self.is_goal_visited((i, j)):
                    unk = self._frontier_unknown_neighbor_count(i, j, radius_cells=unk_r)
                    score = float(d) - unk_bonus * float(unk)
                    if best_score is None or score < best_score:
                        best_score = score
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
                q.append((ni, nj, d + 1))

        return best

    def find_farthest_reachable_frontier(self, frontiers, start_cell):
        """Return the farthest *reachable* frontier using BFS over FREE cells.

        This is intended as a fallback when the robot is stuck repeatedly failing
        to pick/plan to nearby frontiers.

        Primary criterion: maximize BFS distance from start.
        Tie-break: prefer more UNKNOWN neighbors (more exploration gain).
        """
        if not frontiers:
            return None

        si, sj = start_cell
        if not self.inside_map(si, sj):
            return None

        frontier_set = set((int(i), int(j)) for (i, j) in frontiers)
        hard_blocked = self._compute_hard_blocked()

        visited = [[False for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        visited[sj][si] = True

        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        unk_r = int(getattr(self, 'FRONTIER_UNKNOWN_RADIUS_CELLS', 3))

        best = None
        best_d = -1
        best_unk = -1

        q = deque()
        q.append((si, sj, 0))

        while q:
            i, j, d = q.popleft()

            if (i, j) in frontier_set and self.display_state[j][i] == self.FREE:
                if (not hard_blocked[j][i]) and (not self.is_goal_visited((i, j))):
                    unk = self._frontier_unknown_neighbor_count(i, j, radius_cells=unk_r)
                    if (d > best_d) or (d == best_d and unk > best_unk):
                        best = (i, j)
                        best_d = int(d)
                        best_unk = int(unk)

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
                q.append((ni, nj, d + 1))

        return best

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
        best_cost = None

        # Mild preference for frontiers that border more UNKNOWN.
        unk_r = int(getattr(self, 'FRONTIER_UNKNOWN_RADIUS_CELLS', 3))
        unk_bonus_d2 = float(getattr(self, 'FRONTIER_UNKNOWN_BONUS_D2', 15.0))

        while q:
            i, j = q.popleft()

            if (i, j) in frontier_set and self.display_state[j][i] == self.FREE and not hard_blocked[j][i]:
                d2 = (i - gi) ** 2 + (j - gj) ** 2
                unk = self._frontier_unknown_neighbor_count(i, j, radius_cells=unk_r)
                cost = float(d2) - unk_bonus_d2 * float(unk)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
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

        self.scan_id += 1
        self.last_pose.update({'x': x, 'y': y, 'yaw': yaw})

        # Ensure the robot's own cell is treated as free for planning.
        ri, rj = self.world_to_grid(x, y)
        self.update_cell(ri, rj, self.L_FREE)

        ranges = self.lidar.getRangeImage()
        fov = self.lidar.getFov()
        N = self.lidar.getHorizontalResolution()
        angle_step = fov / (N - 1)
        
        for k in range(N):
            d = ranges[k]
            if d <= self.lidar.getMinRange() or d >= self.lidar.getMaxRange(): continue
            
            angle = -( (-fov/2.0) + k * angle_step)
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

        # Gains - higher values for faster response
        k_lin = 5.0
        k_ang = 4.0

        # --- TURN-IN-PLACE IF MISALIGNED ---
        # Only turn in place for large angle errors; otherwise drive while turning
        if abs(angle_error) > 0.5:
            # Before turning in place at corners, move forward a little, then turn.
            if not hasattr(self, "_preturn_steps_left"):
                self._preturn_steps_left = 0
            if not hasattr(self, "_was_turning_in_place"):
                self._was_turning_in_place = False

            if not self._was_turning_in_place:
                self._was_turning_in_place = True
                preturn_time_s = 0.30
                steps = int((preturn_time_s * 1000.0) / max(1.0, float(self.timestep)))
                self._preturn_steps_left = max(1, steps)

            if self._preturn_steps_left > 0:
                self._preturn_steps_left -= 1
                v = 1.0
                w = 0.0
            else:
                v = 0.5  # keep some forward motion even while turning
                w = k_ang * angle_error
        else:
            if hasattr(self, "_was_turning_in_place"):
                self._was_turning_in_place = False
            if hasattr(self, "_preturn_steps_left"):
                self._preturn_steps_left = 0
            v = k_lin * dist
            w = k_ang * angle_error

        # If we're close to obstacles, reduce angular speed to avoid clipping corners.
        w_limit = self.MAX_W
        if min_front is not None and min_front < self.SAFETY_SLOW_DIST_M:
            w_limit = self.MAX_W_CLOSE
            # Also cap forward motion a bit when close to a wall.
            v = min(v, 3.0)

        w = max(min(w, w_limit), -w_limit)
        v = max(min(v, self.MAX_V), 0.0)

        vL = v - w
        vR = v + w

        for idx, motor in enumerate(self.motors):
            motor.setVelocity(vL if idx % 2 == 0 else vR)

        return False

    def spin_in_place(self, speed=2.0):
        """Rotate in place (call every control step while you want to spin)."""
        vL, vR = -speed, speed  # left backward, right forward
        for idx, motor in enumerate(self.motors):
            motor.setVelocity(vL if idx % 2 == 0 else vR)

    def turn_in_place_to_yaw(self, target_yaw, speed=2.0, tol_rad=0.18):
        """Turn in place until yaw is within tol_rad of target_yaw.

        Returns True when finished, otherwise False (and commands wheel speeds).
        """
        try:
            _, _, yaw, _ = self.get_pose()
        except Exception:
            return False

        # Normalize angle error to [-pi, pi]
        err = float(target_yaw) - float(yaw)
        err = math.atan2(math.sin(err), math.cos(err))
        if abs(err) <= float(tol_rad):
            self.stop()
            return True

        sgn = 1.0 if err > 0.0 else -1.0
        vL, vR = -float(speed) * sgn, float(speed) * sgn
        for idx, motor in enumerate(self.motors):
            motor.setVelocity(vL if idx % 2 == 0 else vR)
        return False

    def approach_color_centroid(self, detection_info, forward_speed=2.2, ang_gain=3.2):
        """Simple visual servoing: center the detected color and move forward.

        Used to approach the blue pillar before we have stable coordinates.
        Returns False (still approaching). If detection is missing, spins to search.
        """
        if detection_info is None:
            self.spin_in_place(speed=1.4)
            return False

        try:
            cx, cy = detection_info['centroid_px']
            img_w = detection_info['img_width']
        except Exception:
            self.spin_in_place(speed=1.4)
            return False

        # Convert pixel offset to an angle error (rad), similar to estimate_object_world_coords.
        cx_normalized = (cx - img_w / 2.0) / (img_w / 2.0)
        angle_err = -float(cx_normalized) * (float(getattr(self, 'CAMERA_FOV_H', 1.047)) / 2.0)

        # Safety: if obstacle is too close in front, don't drive forward.
        min_front = self._min_lidar_distance_in_front()
        v = float(forward_speed)
        if min_front is not None and min_front < float(getattr(self, 'SAFETY_STOP_DIST_M', 0.12)):
            v = 0.0

        w = float(ang_gain) * float(angle_err)
        w = max(min(w, float(getattr(self, 'MAX_W', 3.5))), -float(getattr(self, 'MAX_W', 3.5)))
        v = max(min(v, float(getattr(self, 'MAX_V', 6.0))), 0.0)

        vL = v - w
        vR = v + w
        for idx, motor in enumerate(self.motors):
            motor.setVelocity(vL if idx % 2 == 0 else vR)
        return False


    def trigger_red_avoidance(self, detection_info):
        """High-priority behavior: on strong red (>=70%), turn around + replan + optionally block front cells."""
        red_det = self._get_detection_for_color(detection_info, 'red')
        if red_det is None:
            return False

        coverage = float(red_det.get('coverage', 0.0) or 0.0)
        th = float(getattr(self, 'RED_AVOID_COVERAGE_THRESHOLD', 0.70))
        if coverage < th:
            return False

        # --- Mark ~100 cells in front as occupied ---
        try:
            x, y, yaw, _ = self.get_pose()  # Robot's pose in world coordinates
            
            # Block approximately 100 cells in front of the robot
            # Using a corridor approach: several rows of cells in direction of red wall
            # Start blocking at a safe distance (0.5m = 25 cells with 0.02m resolution)
            # to leave an escape path for the robot to turn and replan.
            cells_blocked = 0
            
            # Block cells in a corridor pattern (width x depth)
            corridor_width_cells = 40 # cells wide
            corridor_depth_cells = 20 # cells deep (5 x 20 = 100 cells)
            half_width = corridor_width_cells // 2
            
            # Start blocking 0.4m (20 cells) away from robot to leave escape path
            start_distance_m = 0.7
            start_depth_cells = int(start_distance_m / self.MAP_RES)
            
            # Perpendicular direction for width
            perp_x = -math.sin(yaw)
            perp_y = math.cos(yaw)
            
            for depth in range(start_depth_cells, start_depth_cells + corridor_depth_cells):
                for width_offset in range(-half_width, half_width + 1):
                    # Position along the corridor
                    wx = x + depth * self.MAP_RES * math.cos(yaw) + width_offset * self.MAP_RES * perp_x
                    wy = y + depth * self.MAP_RES * math.sin(yaw) + width_offset * self.MAP_RES * perp_y
                    ci, cj = self.world_to_grid(wx, wy)
                    if self.inside_map(ci, cj):
                        self.force_occupy_cell(ci, cj)
                        cells_blocked += 1
            
            print(f"RED AVOID: Blocked {cells_blocked} cells in front of robot (starting {start_distance_m}m away)")
            
            # Invalidate hard-block cache
            self._hard_blocked_cache = None
            self._hard_blocked_cache_scan_id = None
            
            # Clear visited goals so the robot can pick new frontiers immediately
            self.visited_goals.clear()
            self.frontier_failure_count = 0
            
        except Exception as e:
            print(f"Failed to mark cells in front as occupied: {e}")

        # Clear current plan/goal and force exploration to pick a different frontier.
        if getattr(self, 'mission_state', None) == 'explore' and self.current_goal is not None:
            self.mark_goal_visited(self.current_goal)

        if getattr(self, 'mission_state', None) not in (None, 'done'):
            self.mission_state = 'explore'

        self.current_goal = None
        self.need_new_goal = True
        self.path = None
        self.path_goal = None
        self.path_index = 0

        # Start a turn-around (about 180 degrees).
        try:
            self._red_forbidden_dir_world = (math.cos(float(yaw)), math.sin(float(yaw)))
            try:
                self._red_avoid_bias_until_time = float(self.robot.getTime()) + float(getattr(self, 'RED_AVOID_TTL_S', 8.0))
            except Exception:
                self._red_avoid_bias_until_time = 0.0

            self.red_avoid_target_yaw = float(yaw) + math.pi
            self.red_avoid_target_yaw = math.atan2(math.sin(self.red_avoid_target_yaw), math.cos(self.red_avoid_target_yaw))
            # Start with reverse movement before turning
            self.red_avoid_state = 'reverse'
            self._red_avoid_reverse_start_pos = (float(x), float(y))
        except Exception:
            self.red_avoid_state = None
            self.red_avoid_target_yaw = None
            self._red_avoid_reverse_start_pos = None

        # Cooldown to prevent immediate retrigger loops.
        try:
            self._red_cooldown_until_time = float(self.robot.getTime()) + float(getattr(self, 'RED_COOLDOWN_S', 6.0))
        except Exception:
            pass

        print(f"RED detected (coverage={coverage*100.0:.1f}% >= {th*100.0:.1f}%) -> turning 180° + replanning + blocking front cells")
        return True


    def process_red_avoidance(self):
        """If red avoidance is active, perform it and return True (handled).
        
        States: 'reverse' -> 'turn' -> None
        First moves backward to clear walls, then turns 180 degrees.
        """
        if getattr(self, 'red_avoid_state', None) is None:
            return False

        # State: reverse - move backward before turning
        if self.red_avoid_state == 'reverse':
            start_pos = getattr(self, '_red_avoid_reverse_start_pos', None)
            target_dist = float(getattr(self, 'RED_AVOID_REVERSE_DISTANCE_M', 0.15))
            
            if start_pos is None:
                # No start position, skip to turn
                self.red_avoid_state = 'turn'
                return True
            
            rx, ry, _, _ = self.get_pose()
            dist_moved = math.sqrt((rx - start_pos[0])**2 + (ry - start_pos[1])**2)
            
            if dist_moved >= target_dist:
                # Done reversing, now turn
                self.red_avoid_state = 'turn'
                self._red_avoid_reverse_start_pos = None
                self.stop()
                print(f"RED AVOID: Finished reversing {dist_moved:.2f}m, now turning...")
            else:
                # Continue reversing (drive backward)
                reverse_speed = -2.0  # Negative for backward
                for motor in self.motors:
                    motor.setVelocity(reverse_speed)
            return True

        # State: turn - rotate 180 degrees
        if self.red_avoid_state == 'turn' and self.red_avoid_target_yaw is not None:
            done = self.turn_in_place_to_yaw(self.red_avoid_target_yaw, speed=2.2, tol_rad=0.20)
            if done:
                self.red_avoid_state = None
                self.red_avoid_target_yaw = None
                self.stop()
                print("RED AVOID: Turn complete, replanning...")
            return True

        # Unknown state: reset.
        self.red_avoid_state = None
        self.red_avoid_target_yaw = None
        self._red_avoid_reverse_start_pos = None
        return False

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
            controller.maybe_update_floating_walls()
            controller.update_pygame_map()

            # Draw camera feed if a CameraDisplay device exists
            controller.update_camera_display()

            # Process camera and get detection info
            detection_info = controller.process_camera()

            # Red wall: lock cells between the wall and the robot as OCC (persistent).
            controller.mark_red_front_obstacles(detection_info)

            # Debug: visualize green detections on the map (overlay only).
            controller.debug_record_green_detection(detection_info, radius_m=0.10)

            # Green cooldown (prevents repeated green-trigger loops)
            now_t = controller.robot.getTime()
            green_cooldown_active = now_t < float(getattr(controller, "_green_cooldown_until_time", 0.0))

            # Red cooldown (prevents repeated red-trigger loops)
            red_cooldown_active = now_t < float(getattr(controller, "_red_cooldown_until_time", 0.0))
            
            # --- GREEN SCANNING STATE MACHINE ---
            # Check if green scan is in progress
            if controller.green_scan_state is not None:
                controller.process_green_scan(detection_info)
                controller.run_mapping()  # Keep mapping while scanning
                continue

            # --- RED AVOIDANCE (highest priority) ---
            # If already avoiding red, keep turning until done.
            if controller.process_red_avoidance():
                controller.run_mapping()
                continue

            # If red dominates the ROI (>=70%), turn around and replan.
            # Optionally, red-front forced occupancy may already be applied above.
            if (not red_cooldown_active) and detection_info:
                red_det = controller._get_detection_for_color(detection_info, 'red')
                if red_det is not None:
                    cov = float(red_det.get('coverage', 0.0) or 0.0)
                    th = float(getattr(controller, 'RED_AVOID_COVERAGE_THRESHOLD', 0.70))
                    if cov >= th and controller.trigger_red_avoidance(detection_info):
                        controller.run_mapping()
                        continue
            
            # Check if we should start a green scan (5% coverage triggers)
            # Ignore green while going to pillars so it doesn't interrupt pillar navigation.
            if (not green_cooldown_active) and controller.mission_state not in ('go_to_blue', 'go_to_yellow') and detection_info:
                green_det = controller._get_detection_for_color(detection_info, 'green')
                if green_det is None:
                    green_det = None
                coverage = green_det.get('coverage', 0.0) if green_det else 0.0
                # Check if this region hasn't been marked yet
                rx, ry, ryaw, _ = controller.get_pose()
                if green_det is not None:
                    cx, cy = green_det['centroid_px']
                    img_w = green_det['img_width']
                    cx_normalized = (cx - img_w / 2.0) / (img_w / 2.0)
                    angle_offset = -cx_normalized * (controller.CAMERA_FOV_H / 2.0)
                    obj_angle = ryaw + angle_offset
                    
                    # Estimate green location
                    depth = controller._get_depth_at(cx, cy) or 1.5
                    green_x = rx + depth * math.cos(obj_angle)
                    green_y = ry + depth * math.sin(obj_angle)
                    gi, gj = controller.world_to_grid(green_x, green_y)
                    region_key = (gi // 10, gj // 10)
                    
                    if coverage >= controller.GREEN_TRIGGER_COVERAGE and region_key not in controller.marked_green_regions:
                        # Mark this region immediately to prevent repeated retriggers
                        # if the scan aborts or the estimate jitters slightly.
                        controller.green_scan_pending_region_key = region_key
                        controller.marked_green_regions.add(region_key)

                        # If multiple colors are present in the same ROI (e.g., green + blue),
                        # record pillar detections too before we 'continue' into the green scan.
                        action = controller.handle_color_detection(detection_info)
                        if action == 'go_to_blue':
                            controller.mission_state = 'go_to_blue'

                        # Start scan using the green-specific detection payload.
                        controller.start_green_scan(green_det)
                        continue
            
            # Legacy green floor obstacle marking (for smaller detections)
            if (not green_cooldown_active) and controller.mission_state not in ('go_to_blue', 'go_to_yellow'):
                controller.mark_green_floor_obstacles(detection_info)

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
                # Update blue detection continuously so we lock coordinates once coverage is high enough.
                controller.handle_color_detection(detection_info)

                # Check if we've reached blue
                if controller.reached_color_object(controller.blue_coords, threshold_m=0.25):
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

                # While navigating to the blue object, treat green as poison and block
                # a larger area around it (forces replanning away from green).
                controller.mark_green_poison_zone(detection_info, radius_m=1.1)

                # If we haven't locked blue coordinates yet (coverage < 10%),
                # move toward the blue pillar using camera centroid steering.
                if controller.blue_coords is None:
                    blue_det = controller._get_detection_for_color(detection_info, 'blue') if detection_info else None
                    if blue_det is not None and float(blue_det.get('coverage', 0.0) or 0.0) >= float(getattr(controller, 'BLUE_SEEK_COVERAGE_THRESHOLD', 0.01)):
                        controller.approach_color_centroid(blue_det, forward_speed=1.2, ang_gain=3.0)
                    else:
                        # Lost sight: search by spinning.
                        controller.spin_in_place(speed=1.4)
                    continue
                
                # Navigate to blue
                blue_goal = controller.get_color_object_goal_free(controller.blue_coords)
                if not blue_goal:
                    blue_goal = controller.get_color_object_goal(controller.blue_coords)
                if not blue_goal:
                    # Keep trying to navigate to blue once coordinates exist.
                    controller.stop()
                    continue

                # If we already have a valid path to blue, just follow it - no replanning!
                if controller.path and controller.current_goal == blue_goal and controller.path_index < len(controller.path):
                    # Just move - no A* call
                    controller.move_to_goal(*controller.current_goal)
                    continue

                # Only compute path if we don't have one yet or goal changed
                if controller.current_goal != blue_goal or not controller.path:
                    rx, ry, _, _ = controller.get_pose()
                    ri, rj = controller.world_to_grid(rx, ry)
                    
                    blue_path = controller.astar((ri, rj), blue_goal)
                    if blue_path:
                        controller.current_goal = blue_goal
                        controller.need_new_goal = False
                        controller.path = blue_path
                        controller.path_goal = blue_goal
                        controller.path_index = 0
                        if controller.path and controller.path[0] == (ri, rj):
                            controller.path_index = 1
                        print(f"Setting goal to BLUE at grid {blue_goal}, path_len={len(blue_path)}")
                    else:
                        # No path yet - keep trying until a path exists.
                        controller.current_goal = None
                        controller.path = None
                        continue

                # Move if we have a path
                if controller.path and controller.current_goal:
                    controller.move_to_goal(*controller.current_goal)
                continue
            
            # --- STATE: EXPLORE ---
            if controller.mission_state == 'explore':
                # While exploring, continuously scan for colors
                action = controller.handle_color_detection(detection_info)
                
                # If blue found during exploration, go to it immediately
                if action == 'go_to_blue' and not controller.blue_reached:
                    controller.mission_state = 'go_to_blue'
                    controller.current_goal = None
                    controller.need_new_goal = True
                    controller.stop()
                    print("Blue detected during exploration - navigating to BLUE!")
                    continue

                # After blue is reached, if we see even a faint yellow, start approaching it
                # (locks exact coords at 10% coverage inside handle_color_detection).
                if action == 'go_to_yellow' and controller.blue_reached and (not controller.yellow_reached):
                    controller.mission_state = 'go_to_yellow'
                    controller.current_goal = None
                    controller.need_new_goal = True
                    controller.path = None
                    controller.stop()
                    print("Yellow detected (faint) after blue reached - approaching YELLOW!")
                    continue
                
                # CHECK: If both blue and yellow coordinates are found AND we have a path,
                # stop exploration immediately and follow that path.
                if controller.blue_found and controller.yellow_found:
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

                    # If we recently saw strong red and turned around, avoid planning
                    # into the red-facing half-plane for a short period.
                    try:
                        red_bias_active = float(controller.robot.getTime()) < float(getattr(controller, '_red_avoid_bias_until_time', 0.0))
                    except Exception:
                        red_bias_active = False

                    forbidden = getattr(controller, '_red_forbidden_dir_world', None)
                    if red_bias_active and isinstance(forbidden, (tuple, list)) and len(forbidden) == 2:
                        fdx, fdy = float(forbidden[0]), float(forbidden[1])
                        filtered = []
                        for (fi, fj) in frontiers:
                            fx, fy = controller.grid_to_world_center(int(fi), int(fj))
                            vx = float(fx) - float(rx)
                            vy = float(fy) - float(ry)
                            # dot > 0 means frontier lies in the forbidden half-plane
                            if (vx * fdx + vy * fdy) > 0.0:
                                continue
                            filtered.append((fi, fj))

                        # Only apply filtering if it leaves us with some candidates.
                        if filtered:
                            frontiers = filtered

                    # If we have a recent hint (low-coverage pillar seen), bias exploration
                    # toward that direction by choosing a reachable frontier closest to it.
                    blue_hint = getattr(controller, 'blue_hint_goal_cell', None)
                    yellow_hint = getattr(controller, 'yellow_hint_goal_cell', None)
                    try:
                        blue_hint_active = (blue_hint is not None) and (float(controller.robot.getTime()) < float(getattr(controller, '_blue_hint_until_time', 0.0)))
                    except Exception:
                        blue_hint_active = False
                    try:
                        yellow_hint_active = (yellow_hint is not None) and (float(controller.robot.getTime()) < float(getattr(controller, '_yellow_hint_until_time', 0.0)))
                    except Exception:
                        yellow_hint_active = False

                    hint_goal = None
                    # Preserve mission order: bias toward blue until it's reached, then yellow.
                    if blue_hint_active and (not controller.blue_found) and (not controller.blue_reached):
                        hint_goal = blue_hint
                    elif yellow_hint_active and (not controller.yellow_found) and (not controller.yellow_reached):
                        hint_goal = yellow_hint

                    # Always select the closest unexplored frontier
                    # find_nearest_reachable_frontier already prioritizes:
                    # 1. Closest reachable frontiers (by BFS distance)
                    # 2. Frontiers with more UNKNOWN neighbors (unexplored areas)
                    if hint_goal is not None:
                        chosen = controller.find_reachable_frontier_toward_goal(frontiers, (ri, rj), hint_goal)
                        if chosen is None:
                            chosen = controller.find_nearest_reachable_frontier(frontiers, (ri, rj))
                    else:
                        chosen = controller.find_nearest_reachable_frontier(frontiers, (ri, rj))
                    
                    # If no nearest frontier found, try farthest as fallback
                    if chosen is None:
                        print("No nearest frontier found, trying farthest reachable frontier...")
                        chosen = controller.find_farthest_reachable_frontier(frontiers, (ri, rj))
                    if chosen is None:
                        controller.frontier_failure_count += 1
                        print(f"No reachable unvisited frontier found (failure count: {controller.frontier_failure_count})")
                        
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
                # Update yellow detection continuously so we lock coordinates once coverage is high enough.
                controller.handle_color_detection(detection_info)

                # Check if we've reached yellow
                yellow_reach_coords = controller.yellow_pillar_coords if controller.yellow_pillar_coords else controller.yellow_coords
                if controller.reached_color_object(yellow_reach_coords, threshold_m=0.35):
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
                    yellow_det = controller._get_detection_for_color(detection_info, 'yellow') if detection_info else None
                    if yellow_det is not None and float(yellow_det.get('coverage', 0.0) or 0.0) >= float(getattr(controller, 'YELLOW_SEEK_COVERAGE_THRESHOLD', 0.002)):
                        controller.approach_color_centroid(yellow_det, forward_speed=1.2, ang_gain=3.0)
                    else:
                        controller.spin_in_place(speed=1.4)
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

                # Only plan once when we need a new path
                if controller.current_goal != yellow_goal or not controller.path:
                    direct_path = controller.astar((ri, rj), yellow_goal)
                    if direct_path:
                        controller.current_goal = yellow_goal
                        controller.need_new_goal = False
                        controller.path = direct_path
                        controller.path_goal = yellow_goal
                        controller.path_index = 0
                        if controller.path and controller.path[0] == (ri, rj):
                            controller.path_index = 1
                        print(f"Setting goal to YELLOW at grid {yellow_goal}, path_len={len(direct_path)}")
                    else:
                        # No path to this yellow goal - blacklist it and try a different nearby cell
                        print(f"No path to yellow goal {yellow_goal} - blacklisting and trying another nearby cell")
                        controller._yellow_goal_blacklist.add(yellow_goal)
                        controller.current_goal = None
                        controller.path = None
                        
                        # If we've blacklisted too many cells, fall back to exploration to map more
                        if len(controller._yellow_goal_blacklist) > 20:
                            print("Too many failed yellow goals - falling back to exploration")
                            controller._yellow_goal_blacklist.clear()
                            controller.mission_state = 'explore'
                            controller.need_new_goal = True
                        continue

                # Move if we have a path
                if controller.path and controller.current_goal:
                    controller.move_to_goal(*controller.current_goal)
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

