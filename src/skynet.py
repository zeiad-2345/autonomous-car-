#!/usr/bin/env python3
"""
Skynet — Integrated Autonomous Driving System
==============================================
Runs on the Raspberry Pi. Ties together all perception, planning, and actuation
into a single process with three cooperating threads.

Threads:
  PerceptionThread — Camera → YOLOv8 + sign_filters → annotated frame → TCP stream
  PlannerThread    — Sign label → driving rule lookup → speed / steer command
  SerialThread     — Speed / steer → Arduino; reads @imu / @encoder telemetry back

Usage (on the Pi):
    python3 src/skynet.py
    python3 src/skynet.py --no-stream          # skip video streaming to Mac
    python3 src/skynet.py --no-arduino         # skip serial (useful for bench testing)
    python3 src/skynet.py --conf 0.4           # change YOLO confidence threshold
    python3 src/skynet.py --laptop-ip 192.168.1.10  # override laptop IP

On the Mac — run in a second terminal to see the annotated feed:
    python3 services/rpi-wifi-fallback/frame_receiver_server.py --display
"""

import sys
import os
import time
import threading
import argparse
import socket
import signal

import cv2
import numpy as np

from shared_state import SharedState

# ── Core perception engine ─────────────────────────────────────────────────
try:
    from multi import run_live
except ImportError:
    print("[Skynet] ❌ multi.py not found. Skynet requires multi.py to run.")
    sys.exit(1)

# ── Optional imports (graceful fallbacks) ──────────────────────────────────

try:
    from serial_controller import SerialController
    SERIAL_AVAILABLE = True
except ImportError:
    try:
        # When run from inside the src/ folder directly
        from serial_controller import SerialController
        SERIAL_AVAILABLE = True
    except ImportError:
        SERIAL_AVAILABLE = False
        print("[Skynet] ⚠️  serial_controller not found. Arduino control disabled.")

# ── Planner rules table ────────────────────────────────────────────────────
# Each rule defines the response when a sign is detected.
# duration_s = how many seconds to hold this action (0 = until next sign/clear)

SIGN_RULES = {
    "stop":             {"speed": 0,   "steer": 0, "duration_s": 3},
    "highway_entrance": {"speed": 30,  "steer": 0, "duration_s": 0},
    "highway_exit":     {"speed": 15,  "steer": 0, "duration_s": 0},
    "crosswalk":        {"speed": 8,   "steer": 0, "duration_s": 2},
    "roundabout":       {"speed": 10,  "steer": 0, "duration_s": 0},
    "parking":          {"speed": 0,   "steer": 0, "duration_s": 5},
    "priority":         {"speed": 15,  "steer": 0, "duration_s": 0},
    "one_way":          {"speed": 15,  "steer": 0, "duration_s": 0},
    "no_entry":         {"speed": 0,   "steer": 0, "duration_s": 0},
}

DEFAULT_SPEED  = 15   # city cruise speed (cm/s) — BFMC max = 20
HIGHWAY_SPEED  = 30   # highway cruise speed (cm/s) — BFMC max = 40

# ══════════════════════════════════════════════════════════════════════════════
# Thread 1 — Planner (Reads data from multi.py state)
# ══════════════════════════════════════════════════════════════════════════════

class PlannerThread(threading.Thread):
    """
    Reads sign detections from SharedState, applies the SIGN_RULES table,
    and writes speed / steer commands back to SharedState.
    No ROS, no external libraries — pure Python logic.
    """

    def __init__(self, state: SharedState, args):
        super().__init__(name="PlannerThread", daemon=True)
        self.state = state
        self.args = args
        self._rule_active_until = 0
        self._current_speed = 0
        self._current_steer = 0
        # Lane-following PD gains (tune these on the track)
        self._kp_lane = 25.0   # proportional gain: error → steer degrees (MORE AGGRESSIVE)
        self._kd_lane = 8.0    # derivative gain: heading → steer damping
        self._last_lane_time = 0  # timestamp of last valid lane result
        self._no_lane_timeout = 2.0  # seconds without lane → brake
        self._is_highway = False  # track whether we're on highway
        self._recovery_until = 0  # timestamp for lane collision recovery
        self._active_sign = None  # tracks the sign currently dictating the rule
        self._sign_cooldowns = {} # dictionary of timestamps { "stop": 16912345.0 }

    def run(self):
        print("[Planner] Starting…")
        # Start at cruise speed ONLY if --cruise is enabled
        start_speed = DEFAULT_SPEED if self.args.cruise else 0
        self.state.set_command(start_speed, 0)
        self._current_speed = start_speed

        while self.state.is_running():
            now = time.time()

            # ── 0. Collision Recovery Mode ────────────────────────────────────
            if self._recovery_until > 0:
                if now < self._recovery_until:
                    # Keep reversing. We don't update sign/lane rules during recovery.
                    self.state.set_command(-15, self._current_steer)
                    time.sleep(0.05)
                    continue
                else:
                    self._recovery_until = 0
                    print("[Planner] 🔄 Recovery complete → resuming forward driving")
                    self._current_speed = HIGHWAY_SPEED if self._is_highway else DEFAULT_SPEED
                    self._current_steer = 0

            # ── 1. Sign detection (overrides everything) ──────────────────
            sign_data = self.state.get_signs()
            detection = None
            if sign_data and sign_data.get("detections"):
                # Extract the highest confidence sign from multi.py's format
                detection = max(sign_data["detections"], key=lambda d: d["conf"])
                
            if detection:
                sign = detection["sign"]
                rule = SIGN_RULES.get(sign)
                
                # Only apply rule if we aren't currently executing a timed rule,
                # AND this specific sign is not currently on a "cooldown"
                is_on_cooldown = self._sign_cooldowns.get(sign, 0) > now
                
                if rule and not is_on_cooldown and self._rule_active_until <= 0:
                    self._active_sign = sign
                    self._current_speed = rule["speed"]
                    if rule["steer"] != 0:
                        self._current_steer = rule["steer"]
                        
                    if rule["duration_s"] > 0:
                        self._rule_active_until = now + rule["duration_s"]
                        # Add a 3-second cooldown AFTER the rule finishes so the car can drive past the sign!
                        self._sign_cooldowns[sign] = now + rule["duration_s"] + 3.0
                        
                    print(f"[Planner] 🚦 {sign.upper():20s} → speed={self._current_speed:+3d}  steer={self._current_steer:+3d}")

            # If a timed rule (e.g., stop for 3s) has expired, resume cruise
            if self._rule_active_until > 0 and now > self._rule_active_until:
                self._rule_active_until = 0
                self._active_sign = None
                self._current_speed = HIGHWAY_SPEED if self._is_highway else DEFAULT_SPEED
                self._current_steer = 0
                print("[Planner] ⏱  Stop duration elapsed → resuming cruise")

            # ── 2. Lane following (continuous steering) ────────────────────
            lane = self.state.get_lane()
            if lane is not None:
                error   = lane.get("error", 0.0)    # -1.0 (left) to +1.0 (right)
                heading = lane.get("heading", 0.0)   # radians
                lane_type = lane.get("lane_type", "city")

                # Collision recovery trigger
                if abs(error) > 0.85 and self._current_speed > 0 and self._rule_active_until <= 0:
                    print(f"[Planner] ⚠️ LANE COLLISION (err={error:.2f}) → INITIATING REVERSE RECOVERY!")
                    self._recovery_until = now + 1.2  # Reverse for 1.2s
                    # If error > 0 (hitting right lane), steer right (+25) while reversing
                    # This pulls the rear to the right, sweeping the front to the LEFT (away from the line)
                    self._current_steer = 25 if error > 0 else -25
                    self._current_speed = -15  # Reverse at 15 cm/s
                    continue

                # Update speed based on lane type (city vs highway)
                if lane_type == "highway" and not self._is_highway:
                    self._is_highway = True
                    if self._current_speed > 0 and self._rule_active_until <= 0:
                        self._current_speed = HIGHWAY_SPEED
                        print(f"[Planner] 🛣️  Highway detected → speed={HIGHWAY_SPEED}")
                elif lane_type != "highway" and self._is_highway:
                    self._is_highway = False
                    if self._current_speed > 0 and self._rule_active_until <= 0:
                        self._current_speed = DEFAULT_SPEED
                        print(f"[Planner] 🏘️  City detected → speed={DEFAULT_SPEED}")

                # PD controller: steer = Kp * error + Kd * heading
                # error > 0 means lane centre is to the RIGHT → steer RIGHT (positive)
                # Clamp to Arduino steer range: -25 to +25
                steer_cmd = self._kp_lane * error + self._kd_lane * heading
                steer_cmd = max(-25, min(25, int(steer_cmd)))

                # Only apply lane steering if no sign rule is actively overriding steer
                if self._rule_active_until <= 0 or SIGN_RULES.get(self._active_sign, {}).get("steer", 0) == 0:
                    self._current_steer = steer_cmd

                self._last_lane_time = now

            # ── 3. Safety: brake if no lane data for too long ─────────────
            if self._current_speed > 0 and self._last_lane_time > 0:
                if now - self._last_lane_time > self._no_lane_timeout:
                    self._current_speed = 0
                    self._current_steer = 0
                    print("[Planner] ⚠️  No lane data for 2s → EMERGENCY STOP")

            self.state.set_command(self._current_speed, self._current_steer)
            time.sleep(0.05)  # 20 Hz planner loop

        # Safety: stop the car on shutdown
        self.state.set_command(0, 0)
        print("[Planner] Stopped.")



# ══════════════════════════════════════════════════════════════════════════════
# Thread 2 — Serial
# ══════════════════════════════════════════════════════════════════════════════

class SerialThread(threading.Thread):
    """
    Reads speed / steer from SharedState, sends to Arduino via SerialController.
    Telemetry (@imu, @encoder) flows back via the telemetry_callback.
    Runs at ~20 Hz to match Planner output rate.
    """

    def __init__(self, state: SharedState, args):
        super().__init__(name="SerialThread", daemon=True)
        self.state = state
        self.args = args
        self.ctrl = None

    def run(self):
        print("[Serial] Starting…")
        if not SERIAL_AVAILABLE or self.args.no_arduino:
            print("[Serial] Arduino control disabled (--no-arduino or module missing).")
            self.ctrl = None
        else:
            self.ctrl = SerialController(telemetry_callback=self._on_telemetry)
            connected = self.ctrl.start()
            if not connected:
                print("[Serial] ❌ Could not connect to Arduino. Running without hardware.")
                self.ctrl = None

        while self.state.is_running():
            speed, steer = self.state.get_command()
            
            if self.ctrl:
                self.ctrl.send_speed(speed)
                self.ctrl.send_steer(steer)

            mode = "📡 SENT" if self.ctrl else "🔌 DRY RUN"
            print(f"\r[Serial] {mode}: speed={speed:+3d} | steer={steer:+3d}        ", end="", flush=True)

            time.sleep(0.05)  # 20 Hz

        if self.ctrl:
            self.ctrl.stop()
        print("[Serial] Stopped.")

    def _on_telemetry(self, label, data):
        self.state.set_telemetry(label, data)
        # Pretty-print to SSH terminal
        if label == "imu":
            print(f"\r[IMU] roll={data.get('roll')} pitch={data.get('pitch')} yaw={data.get('yaw')}", end="")
        elif label == "encoder":
            print(f"\r[ENC] pos={data.get('position')}", end="")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Skynet — Integrated Autonomous Driving System for BFMC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 src/skynet.py                                   # Full run on Pi
  python3 src/skynet.py --no-arduino                      # Bench-test without Arduino
  python3 src/skynet.py --conf 0.35                       # Lower confidence, raw YOLO
        """
    )
    parser.add_argument("--model",       type=str,   default=None,
                        help="Path to YOLO .pt model file")
    parser.add_argument("--conf",        type=float, default=0.5,
                        help="YOLO confidence threshold (default 0.5)")
    parser.add_argument("--no-arduino",  action="store_true",
                        help="Run without Arduino (dry-run / bench testing)")
    parser.add_argument("--webcam",      action="store_true",
                        help="Force webcam instead of Pi Camera")
    parser.add_argument("--source",      type=str,   default=None,
                        help="Source video or image")
    parser.add_argument("--webcam-index",type=int,   default=0,
                        help="OpenCV webcam index if no Pi Camera is present (default 0)")
    parser.add_argument("--cruise",      action="store_true",
                        help="Start driving immediately at cruise speed (default False)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("   🚗  Skynet — Starting Integrated System")
    print(f"   Arduino:    {'DISABLED (dry run)' if args.no_arduino else 'ENABLED'}")
    print(f"   Using multi.py for all visual perception tasks")
    print("=" * 60)

    state = SharedState()

    def handle_signal(sig, frame):
        print(f"\n[Skynet] 🛑 Signal {sig} received — Emergency Shutdown…")
        state.shutdown()

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Core threads (always started) ────────────────────────────────────
    planner    = PlannerThread(state, args)
    serial     = SerialThread(state, args)

    # ── Start all threads ─────────────────────────────────────────────────
    planner.start()
    serial.start()

    all_threads = [planner, serial]

    # ── Hand over the main thread to multi.py ─────────────────────────────
    try:
        # Determine correct model path
        model_file = args.model or "src/perception/sign_recognition/bfmc_best_shirts.pt"
        
        # Starts camera, YOLO thread, Lane thread, and blocks with cv2.imshow
        run_live(
            model_path=model_file,
            conf=args.conf,
            webcam=args.webcam,
            source=args.source,
            add_traffic_box=True,
            state=state  # Inject Skynet's shared state!
        )
    except KeyboardInterrupt:
        print("\n\n[Skynet] ⛔ KeyboardInterrupt — shutting down…")
        state.shutdown()

    for t in all_threads:
        t.join(timeout=3)

    print("[Skynet] ✅ Shutdown complete.")
