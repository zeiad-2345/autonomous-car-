import threading

class SharedState:
    """Single source of truth for all inter-thread state (Vision, Planning, Control)."""

    def __init__(self):
        self._lock = threading.Lock()
        
        # --- Perception (multi.py) dependencies ---
        self._latest_frame = None
        self._lane_result = None
        self._sign_result = None
        self._running = True

        # --- Control (skynet.py) dependencies ---
        self.target_speed = 0             # integer -50..50
        self.target_steer = 0             # integer -25..25
        self.imu_data     = {}
        self.encoder_data = {}
        self._pose = {"x_cm": 0.0, "y_cm": 0.0, "heading_rad": 0.0, "dist_cm": 0.0}

    # ── Sign detection ────────────────────────────────────────────────────
    def set_signs(self, result):
        with self._lock:
            self._sign_result = result

    def get_signs(self):
        with self._lock:
            return self._sign_result

    # ── Drive commands ────────────────────────────────────────────────────
    def set_command(self, speed, steer):
        with self._lock:
            self.target_speed = int(speed)
            self.target_steer = int(steer)

    def get_command(self):
        with self._lock:
            return self.target_speed, self.target_steer

    # ── Telemetry (IMU / encoder from Arduino) ────────────────────────────
    def set_telemetry(self, label, data):
        with self._lock:
            if label == "imu":
                self.imu_data = data
            elif label == "encoder":
                self.encoder_data = data

    # ── Frame sharing (Perception → Lane) ────────────────────────────────
    def set_latest_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame

    # ── Lane result (Lane → Planner) ─────────────────────────────────────
    def set_lane(self, result):
        with self._lock:
            self._lane_result = result

    def get_lane(self):
        with self._lock:
            return self._lane_result

    # ── Pose (Odometry → Localization) ───────────────────────────────────
    def set_pose(self, pose: dict):
        with self._lock:
            self._pose = pose

    def get_pose(self) -> dict:
        with self._lock:
            return dict(self._pose)

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def shutdown(self):
        with self._lock:
            self._running = False

    def is_running(self):
        with self._lock:
            return self._running
