#!/usr/bin/env python3
import threading
import time
import math
import json
import os

class LocalizationThread(threading.Thread):
    """
    Compares detected signs with the map_waypoints.json graph.
    If a detected sign matches a known map waypoint near the current 
    odometry pose, it triggers a 'snap' to correct drift.
    """
    def __init__(self, state, map_path):
        super().__init__(name="LocalizationThread", daemon=True)
        self.state = state
        self.map_path = map_path
        self.waypoints = []
        self.last_snap_time = 0
        self.SNAP_DISTANCE_THRESHOLD = 100.0  # cm

    def run(self):
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, "r") as f:
                    self.waypoints = json.load(f)
                print(f"[Loc] Loaded {len(self.waypoints)} waypoints from {self.map_path}")
            except Exception as e:
                print(f"[Loc] Error loading map: {e}")
        else:
            print(f"[Loc] No map found at {self.map_path}")
        
        while self.state.is_running():
            time.sleep(0.1)  # 10 Hz check
            
            det = self.state.get_detection()
            if not det or det.get("conf", 0) < 0.6:
                continue
            
            sign_type = det.get("sign")
            pose = self.state.get_pose()
            cx, cy = pose.get("x_cm", 0), pose.get("y_cm", 0)
            
            # Find nearest waypoint of the same type
            best_wp = None
            min_d = self.SNAP_DISTANCE_THRESHOLD
            
            for wp in self.waypoints:
                if wp.get("type") == sign_type:
                    d = math.hypot(cx - wp["x_cm"], cy - wp["y_cm"])
                    if d < min_d:
                        min_d = d
                        best_wp = wp
            
            # Snap if we found a match and haven't snapped recently (cooldown)
            if best_wp and (time.time() - self.last_snap_time > 5.0):
                print(f"[Loc] 🎯 SNAP! Matched {sign_type} at {best_wp['x_cm']}, {best_wp['y_cm']} (Dist: {min_d:.1f}cm)")
                self.state.set_snap_pose(
                    best_wp["x_cm"], 
                    best_wp["y_cm"], 
                    best_wp.get("heading_rad", pose["heading_rad"])
                )
                self.last_snap_time = time.time()

        print("[Loc] Stopped.")
