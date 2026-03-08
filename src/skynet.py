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

# ── New perception modules (graceful fallback if files not found yet) ─────────
try:
    from src.perception.lane_detection.threadLaneDetection import LaneDetectionThread
    LANE_AVAILABLE = True
except ImportError:
    try:
        from perception.lane_detection.threadLaneDetection import LaneDetectionThread
        LANE_AVAILABLE = True
    except ImportError:
        LANE_AVAILABLE = False
        print("[Skynet] ⚠️  LaneDetectionThread not found. Lane keeping disabled.")

try:
    from src.perception.odometry.threadOdometry import OdometryThread
    ODOM_AVAILABLE = True
except ImportError:
    try:
        from perception.odometry.threadOdometry import OdometryThread
        ODOM_AVAILABLE = True
    except ImportError:
        ODOM_AVAILABLE = False
        print("[Skynet] ⚠️  OdometryThread not found. Odometry disabled.")

# ── Optional imports (graceful fallbacks) ──────────────────────────────────

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[Skynet] ⚠️  ultralytics not installed. YOLO disabled.")

try:
    from src.serial_controller import SerialController
    SERIAL_AVAILABLE = True
except ImportError:
    try:
        # When run from inside the src/ folder directly
        from serial_controller import SerialController
        SERIAL_AVAILABLE = True
    except ImportError:
        SERIAL_AVAILABLE = False
        print("[Skynet] ⚠️  serial_controller not found. Arduino control disabled.")

try:
    from src.perception.sign_recognition.sign_filters import validate_detection
    FILTERS_AVAILABLE = True
except ImportError:
    try:
        from perception.sign_recognition.sign_filters import validate_detection
        FILTERS_AVAILABLE = True
    except ImportError:
        FILTERS_AVAILABLE = False
        print("[Skynet] ⚠️  sign_filters not found. Filters disabled.")

try:
    import picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

# ── Sign label map ─────────────────────────────────────────────────────────

LABEL_MAP = {
    "stop sign": "stop", "stop": "stop",
    "parking": "parking", "parking_sign": "parking", "p": "parking",
    "priority": "priority", "priority_road": "priority", "priority road": "priority", "give_way": "priority",
    "crosswalk": "crosswalk", "pedestrian": "crosswalk", "pedestrian_crossing": "crosswalk",
    "highway": "highway_entrance", "highway_entrance": "highway_entrance", "motorway": "highway_entrance", "motorway_begin": "highway_entrance",
    "highway_exit": "highway_exit", "motorway_end": "highway_exit", "end_motorway": "highway_exit",
    "roundabout": "roundabout", "roundabout_sign": "roundabout",
    "one_way": "one_way", "one-way": "one_way", "oneway": "one_way",
    "no_entry": "no_entry", "no-entry": "no_entry", "no entry": "no_entry", "noentry": "no_entry",
}

SIGN_COLORS = {
    "stop": (0, 0, 255), "parking": (255, 150, 0), "priority": (0, 215, 255),
    "crosswalk": (255, 200, 0), "highway_entrance": (0, 180, 0), "highway_exit": (0, 130, 0),
    "roundabout": (255, 100, 0), "one_way": (255, 50, 50), "no_entry": (50, 50, 255),
}

# ── Planner rules table ────────────────────────────────────────────────────
# Each rule defines the response when a sign is detected.
# duration_s = how many seconds to hold this action (0 = until next sign/clear)

SIGN_RULES = {
    "stop":             {"speed": 0,   "steer": 0, "duration_s": 3},
    "highway_entrance": {"speed": 30,  "steer": 0, "duration_s": 0},
    "highway_exit":     {"speed": 12,  "steer": 0, "duration_s": 0},
    "crosswalk":        {"speed": 10,  "steer": 0, "duration_s": 0},
    "roundabout":       {"speed": 10,  "steer": 0, "duration_s": 0},
    "parking":          {"speed": 0,   "steer": 0, "duration_s": 0},
    "priority":         {"speed": 20,  "steer": 0, "duration_s": 0},
    "one_way":          {"speed": 20,  "steer": 0, "duration_s": 0},
    "no_entry":         {"speed": 0,   "steer": 0, "duration_s": 0},
}

DEFAULT_SPEED  = 15   # Cruise speed when no sign is actively in effect
STREAM_PORT    = 5012 # Must match frame_receiver_server.py
INFERENCE_HZ   = 10   # YOLO rate: 10 Hz → 4cm per frame at 40cm/s highway speed (safe)

# ══════════════════════════════════════════════════════════════════════════════
# Shared state — thread-safe via lock
# ══════════════════════════════════════════════════════════════════════════════

class SharedState:
    """Single source of truth for all inter-thread state."""

    def __init__(self):
        self._lock = threading.Lock()
        # Perception → Planner (sign detection)
        self.last_detection = None        # {"sign": str, "conf": float, "bbox": list}
        # Perception → Lane (latest raw frame for LaneDetectionThread)
        self._latest_frame  = None
        # Planner → Serial
        self.target_speed = 0             # integer -50..50
        self.target_steer = 0             # integer -25..25
        # Serial → Odometry / Dashboard
        self.imu_data     = {}
        self.encoder_data = {}
        # Lane → Planner
        self._lane_result = None          # {error, heading, lane_type, annotated}
        # Odometry → Localization
        self._pose = {"x_cm": 0.0, "y_cm": 0.0, "heading_rad": 0.0, "dist_cm": 0.0}
        self.running = True

    # ── Sign detection ────────────────────────────────────────────────────
    def set_detection(self, detection):
        with self._lock:
            self.last_detection = detection

    def get_detection(self):
        with self._lock:
            d = self.last_detection
            self.last_detection = None    # consume once
            return d

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
            self.running = False

    def is_running(self):
        with self._lock:
            return self.running


# ══════════════════════════════════════════════════════════════════════════════
# Thread 1 — Perception
# ══════════════════════════════════════════════════════════════════════════════

class PerceptionThread(threading.Thread):
    """
    Reads camera frames, runs YOLOv8, applies sign_filters, draws bounding boxes,
    and streams the annotated frame to the Mac via TCP.
    """

    def __init__(self, state: SharedState, args):
        super().__init__(name="PerceptionThread", daemon=True)
        self.state = state
        self.args = args
        self.model = None
        self.cap = None
        self.stream_socket = None
        self._interval = 1.0 / INFERENCE_HZ

    def run(self):
        print("[Perception] Starting…")
        self._init_camera()
        self._load_model()
        if args.stream:
            self._init_stream()

        last_time = 0
        while self.state.is_running():
            now = time.time()
            if now - last_time < self._interval:
                time.sleep(0.01)
                continue
            last_time = now

            frame = self._get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # Share raw frame for LaneDetectionThread
            self.state.set_latest_frame(frame)

            if self.model:
                detections = self._infer(frame)
                self._draw(frame, detections)
                # Publish best detection to planner
                if detections:
                    best = max(detections, key=lambda d: d["conf"])
                    self.state.set_detection(best)
            
            if self.stream_socket:
                self._stream_frame(frame)

        self._cleanup()

    # ── Camera ────────────────────────────────────────────────────────────

    def _init_camera(self):
        if PICAM_AVAILABLE:
            try:
                self.cam = picamera2.Picamera2()
                config = self.cam.create_preview_configuration(
                    main={"format": "RGB888", "size": (640, 480)},
                )
                self.cam.configure(config)
                self.cam.start()
                self.cap = None  # signal: use picamera2
                print("[Perception] ✅ Pi Camera initialized")
                return
            except Exception as e:
                print(f"[Perception] Pi Camera failed: {e}. Falling back to OpenCV.")

        idx_candidates = [idx, 1, 0, 2] # Try user index, then fallback
        self.cam = None
        self.cap = None
        
        for i in idx_candidates:
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                print(f"[Perception] ✅ OpenCV webcam {i} initialized")
                return
            self.cap.release()
            
        print("[Perception] ⚠️  No camera found. Perception disabled.")
        self.cap = None

    def _get_frame(self):
        if self.cam is not None and PICAM_AVAILABLE:
            try:
                return self.cam.capture_array()
            except Exception:
                return None
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def _cleanup(self):
        if PICAM_AVAILABLE and self.cam:
            self.cam.stop()
        if self.cap:
            self.cap.release()
        if self.stream_socket:
            self.stream_socket.close()
        print("[Perception] Stopped.")

    # ── YOLO ──────────────────────────────────────────────────────────────

    def _load_model(self):
        if not YOLO_AVAILABLE:
            return
        model_candidates = [
            self.args.model,
            "src/perception/sign_recognition/bfmc_best_shirts.pt",
            "src/perception/sign_recognition/bfmc_best.pt",
            "yolov8n.pt",
        ]
        for path in model_candidates:
            if path and os.path.exists(path):
                try:
                    self.model = YOLO(path)
                    print(f"[Perception] ✅ YOLO model loaded: {path}")
                    return
                except Exception as e:
                    print(f"[Perception] Failed to load {path}: {e}")
        print("[Perception] ⚠️  No YOLO model found.")

    def _infer(self, frame):
        detections = []
        try:
            results = self.model(frame, conf=self.args.conf, verbose=False)
        except Exception as e:
            print(f"[Perception] Inference error: {e}")
            return detections

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                raw_label = self.model.names[cls_id].lower().strip()
                conf = float(box.conf[0])
                bfmc_sign = LABEL_MAP.get(raw_label)
                if bfmc_sign is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if FILTERS_AVAILABLE and not self.args.no_filters:
                    if not validate_detection(frame, bfmc_sign, (x1, y1, x2, y2)):
                        continue

                detections.append({
                    "sign": bfmc_sign,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                })
        return detections

    def _draw(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = d["sign"]
            conf = d["conf"]
            color = SIGN_COLORS.get(label, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label.upper()} {conf:.0%}"
            cv2.putText(frame, text, (x1, max(y1-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── TCP Stream ─────────────────────────────────────────────────────────

    def _init_stream(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((self.args.laptop_ip, STREAM_PORT))
            sock.settimeout(0.2)
            self.stream_socket = sock
            print(f"[Perception] ✅ Video stream connected to {self.args.laptop_ip}:{STREAM_PORT}")
        except Exception as e:
            print(f"[Perception] Stream not connected ({e}). Video streaming disabled.")
            self.stream_socket = None

    def _stream_frame(self, frame):
        try:
            req = self.stream_socket.recv(1)
            if req != b'F':
                return
            resized = cv2.resize(frame, (640, 480))
            _, jpg = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
            data = jpg.tobytes()
            self.stream_socket.sendall(len(data).to_bytes(8, 'big'))
            self.stream_socket.sendall(data)
            self.stream_socket.recv(1024)   # consume "FRAME_PROCESSED"
        except socket.timeout:
            pass   # Server not ready — skip this frame
        except Exception as e:
            print(f"[Perception] Stream broken: {e}")
            self.stream_socket.close()
            self.stream_socket = None


# ══════════════════════════════════════════════════════════════════════════════
# Thread 2 — Planner
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

    def run(self):
        print("[Planner] Starting…")
        # Start at cruise speed ONLY if --cruise is enabled
        start_speed = DEFAULT_SPEED if self.args.cruise else 0
        self.state.set_command(start_speed, 0)
        self._current_speed = start_speed

        while self.state.is_running():
            detection = self.state.get_detection()
            now = time.time()

            if detection:
                sign = detection["sign"]
                rule = SIGN_RULES.get(sign)
                if rule:
                    self._current_speed = rule["speed"]
                    self._current_steer = rule["steer"]
                    if rule["duration_s"] > 0:
                        self._rule_active_until = now + rule["duration_s"]
                    else:
                        self._rule_active_until = 0
                    print(f"[Planner] 🚦 {sign.upper():20s} → speed={self._current_speed:+3d}  steer={self._current_steer:+3d}")

            # If a timed rule (e.g., stop for 3s) has expired, resume cruise
            if self._rule_active_until > 0 and now > self._rule_active_until:
                self._rule_active_until = 0
                self._current_speed = DEFAULT_SPEED
                self._current_steer = 0
                print("[Planner] ⏱  Stop duration elapsed → resuming cruise")

            self.state.set_command(self._current_speed, self._current_steer)
            time.sleep(0.05)  # 20 Hz planner loop

        # Safety: stop the car on shutdown
        self.state.set_command(0, 0)
        print("[Planner] Stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# Thread 3 — Serial
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
            while self.state.is_running():
                spd, steer = self.state.get_command()
                if spd != 0 or steer != 0:
                    print(f"[Serial] (DRY RUN) speed={spd:+3d}  steer={steer:+3d}")
                time.sleep(0.1)
            return

        self.ctrl = SerialController(telemetry_callback=self._on_telemetry)
        connected = self.ctrl.start()
        if not connected:
            print("[Serial] ❌ Could not connect to Arduino. Running without hardware.")
            while self.state.is_running():
                time.sleep(0.1)
            return

        last_speed, last_steer = None, None
        while self.state.is_running():
            speed, steer = self.state.get_command()
            # Only write when value actually changes (avoid serial spam)
            if speed != last_speed:
                self.ctrl.send_speed(speed)
                last_speed = speed
            if steer != last_steer:
                self.ctrl.send_steer(steer)
                last_steer = steer
            time.sleep(0.05)  # 20 Hz

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
  python3 src/skynet.py --no-stream                       # No video to Mac
  python3 src/skynet.py --no-arduino                      # Bench-test without Arduino
  python3 src/skynet.py --conf 0.35 --no-filters          # Lower confidence, raw YOLO
  python3 src/skynet.py --laptop-ip 192.168.50.2          # Custom Mac IP
        """
    )
    parser.add_argument("--model",       type=str,   default=None,
                        help="Path to YOLO .pt model file")
    parser.add_argument("--conf",        type=float, default=0.5,
                        help="YOLO confidence threshold (default 0.5)")
    parser.add_argument("--no-filters",  action="store_true",
                        help="Disable post-detection shape/color/size filters")
    parser.add_argument("--laptop-ip",   type=str,   default="10.82.10.45",
                        help="Laptop IP for video streaming (default 10.82.10.45)")
    parser.add_argument("--no-stream",   action="store_true",
                        help="Disable TCP video streaming to laptop")
    parser.add_argument("--no-arduino",  action="store_true",
                        help="Run without Arduino (dry-run / bench testing)")
    parser.add_argument("--webcam-index",type=int,   default=0,
                        help="OpenCV webcam index if no Pi Camera is present (default 0)")
    parser.add_argument("--cruise",      action="store_true",
                        help="Start driving immediately at cruise speed (default False)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Add stream flag to args for PerceptionThread
    args.stream = not args.no_stream

    print("=" * 60)
    print("   🚗  Skynet — Starting Integrated System")
    print(f"   Streaming:  {'ON → ' + args.laptop_ip if args.stream else 'OFF'}")
    print(f"   Arduino:    {'DISABLED (dry run)' if args.no_arduino else 'ENABLED'}")
    print(f"   Filters:    {'DISABLED' if args.no_filters else 'ENABLED'}")
    print(f"   Lane Det.:  {'ENABLED' if LANE_AVAILABLE else 'DISABLED (module missing)'}")
    print(f"   Odometry:   {'ENABLED' if ODOM_AVAILABLE else 'DISABLED (module missing)'}")
    print("=" * 60)

    state = SharedState()

    def handle_signal(sig, frame):
        print(f"\n[Skynet] 🛑 Signal {sig} received — Emergency Shutdown…")
        state.shutdown()

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Core threads (always started) ────────────────────────────────────
    perception = PerceptionThread(state, args)
    planner    = PlannerThread(state, args)
    serial     = SerialThread(state, args)

    # ── Optional perception threads ───────────────────────────────────────
    lane = LaneDetectionThread(state, args) if LANE_AVAILABLE else None
    odom = OdometryThread(state)            if ODOM_AVAILABLE else None

    # ── Start all threads ─────────────────────────────────────────────────
    perception.start()
    planner.start()
    serial.start()
    if lane: lane.start()
    if odom: odom.start()

    all_threads = [t for t in [perception, planner, serial, lane, odom] if t]

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[Skynet] ⛔ KeyboardInterrupt — shutting down…")
        state.shutdown()

    for t in all_threads:
        t.join(timeout=3)

    print("[Skynet] ✅ Shutdown complete.")

