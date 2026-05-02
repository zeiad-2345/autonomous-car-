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

Usage (on the Pi — everything runs locally):
    python3 src/skynet.py
    python3 src/skynet.py --no-arduino         # skip serial (useful for bench testing)
    python3 src/skynet.py --conf 0.4           # change YOLO confidence threshold

Usage (remote mode — perception on laptop, Pi streams frames):
    # 1. On laptop:  python3 src/skynet.py --remote --port 5555
    # 2. On Pi:      python3 src/pi_bridge.py --laptop-ip <LAPTOP_IP> --port 5555
"""

import sys
import os
import time
import threading
import argparse
import socket
import signal

import struct

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

try:
    from control.Stanley_Control import Control_Code
    STANLEY_AVAILABLE = True
    print("[Skynet] ✅ Stanley controller loaded.")
except ImportError:
    STANLEY_AVAILABLE = False
    print("[Skynet] ⚠️  Stanley controller not found. Falling back to simple PD.")

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
        # Lane-following PID gains (tune these on the track)
        self._kp_lane = 45.0   # proportional gain: error → steer degrees
        self._ki_lane = 5.0    # integral gain: accumulated error → correct drift
        self._kd_lane = 15.0   # derivative gain: heading → steer damping
        self._integral_error = 0.0  # accumulated error for I term
        self._integral_max = 30.0   # anti-windup clamp (max integral contribution in degrees)
        self._last_lane_time = 0  # timestamp of last valid lane result
        self._no_lane_timeout = 2.0  # seconds without lane → brake
        self._is_highway = False  # track whether we're on highway
        self._recovery_until = 0  # timestamp for lane collision recovery
        self._active_sign = None  # tracks the sign currently dictating the rule
        self._sign_cooldowns = {} # dictionary of timestamps { "stop": 16912345.0 }
        self._steer_hold_until = 0  # minimum time to hold a steer command
        self._steer_hold_sec = 0.7  # hold steering briefly so servo has time to move

        self._high_steer_start = 0.0 # Timer for allowing quick steering spikes on straight lines

        # Lane delay buffer: react to values from 2s ago, not the latest
        self._lane_delay_sec = 3
        self._lane_buffer = []  # list of (timestamp, error, heading, lane_type)

    def run(self):
        print("[Planner] Starting…")
        # Start at cruise speed ONLY if --cruise is enabled
        cruise_speed = self.args.speed if self.args.speed is not None else DEFAULT_SPEED
        start_speed = cruise_speed if self.args.cruise else 0
        self.state.set_command(start_speed, 0)
        self._current_speed = start_speed

        while self.state.is_running():
            now = time.time()

            # ── 0. Collision Recovery Mode ────────────────────────────────────
            if not getattr(self.args, 'no_lane_safety', False) and self._recovery_until > 0:
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
                error   = lane.get("error", 0.0)
                heading = lane.get("heading", 0.0)
                lane_type = lane.get("lane_type", "city")

                # Push new reading into delay buffer
                self._lane_buffer.append((now, error, heading, lane_type))

                # Prune entries older than 2× delay (keep memory bounded)
                cutoff = now - self._lane_delay_sec * 2
                self._lane_buffer = [e for e in self._lane_buffer if e[0] >= cutoff]

                # Use the entry closest to 2 seconds ago
                target_time = now - self._lane_delay_sec
                delayed = min(self._lane_buffer, key=lambda e: abs(e[0] - target_time))
                d_error, d_heading, d_lane_type = delayed[1], delayed[2], delayed[3]

                # Update speed based on lane type (city vs highway)
                if d_lane_type == "highway" and not self._is_highway:
                    self._is_highway = True
                    if self._current_speed > 0 and self._rule_active_until <= 0:
                        self._current_speed = HIGHWAY_SPEED
                        print(f"[Planner] 🛣️  Highway detected → speed={HIGHWAY_SPEED}")
                elif d_lane_type != "highway" and self._is_highway:
                    self._is_highway = False
                    if self._current_speed > 0 and self._rule_active_until <= 0:
                        self._current_speed = DEFAULT_SPEED
                        print(f"[Planner] 🏘️  City detected → speed={DEFAULT_SPEED}")

                # Read the confirmed curve state directly from multi.py's timer logic
                is_in_curve_confirmed = self.state.get_curve_mode()

                # Integral Memory: fixes a "shifted face" by accumulating long-term error
                if not is_in_curve_confirmed:
                    self._integral_error += error
                    self._integral_error = max(-5.0, min(5.0, self._integral_error))
                else:
                    self._integral_error = 0.0 # Reset memory in curves

                # PID controller: steer = Kp * error + Ki * integral + Kd * heading
                # error > 0 means lane centre is to the RIGHT → steer RIGHT (positive)
               
                steer_raw = (self._kp_lane * error * 3.5
                             + self._ki_lane * self._integral_error
                             + self._kd_lane * heading)
                

                steer_cmd = int(steer_raw)
                
                if not is_in_curve_confirmed:
                    if self._high_steer_start == 0.0:
                        self._high_steer_start = now
                        
                    if now - self._high_steer_start > .5:
                 
                        steer_cmd = max(-120, min(120, steer_cmd))
                else:
                    self._high_steer_start = 0.0
                    # In curves, allow full steering range
                    steer_cmd = max(-120, min(120, steer_cmd))

                # Only apply lane steering if no sign rule is actively overriding steer
                if self._rule_active_until <= 0 or SIGN_RULES.get(self._active_sign, {}).get("steer", 0) == 0:
                    # Hold steering for a minimum duration so it has physical effect
                    if now >= self._steer_hold_until or abs(steer_cmd) > abs(self._current_steer):
                        self._current_steer = steer_cmd
                        if abs(steer_cmd) >= 15:
                            self._steer_hold_until = now + self._steer_hold_sec

                self._last_lane_time = now

            # ── 3. Safety: brake if no lane data for too long ─────────────
            if not getattr(self.args, 'no_lane_safety', False) and self._current_speed > 0 and self._last_lane_time > 0:
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
        print("[Serial] Starting…", flush=True)
        if not SERIAL_AVAILABLE or self.args.no_arduino:
            print("[Serial] Arduino control disabled (--no-arduino or module missing).", flush=True)
            self.ctrl = None
        else:
            self.ctrl = SerialController(telemetry_callback=self._on_telemetry)
            connected = self.ctrl.start()
            if not connected:
                print("[Serial] ❌ Could not connect to Arduino. Running without hardware.", flush=True)
                self.ctrl = None

        while self.state.is_running():
            speed, steer = self.state.get_command()
            
            if self.ctrl:
                self.ctrl.send_speed(speed)
                self.ctrl.send_steer(steer)

            pwm = int(abs(speed) / 50 * 255)
            dc_cmd = f"DC:FORWARD:{pwm}" if speed > 0 else (f"DC:BACKWARD:{pwm}" if speed < 0 else "DC:STOP")
            servo_cmd = f"SERVO:{max(-120, min(120, steer))}"
            print(f"{dc_cmd}\\n{servo_cmd}\\n", flush=True)

            time.sleep(0.05)  # 20 Hz

        if self.ctrl:
            self.ctrl.stop()
        print("[Serial] Stopped.", flush=True)

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

# ══════════════════════════════════════════════════════════════════════════════
# Remote Mode — Receive frames from Pi over TCP, run perception, send commands
# ══════════════════════════════════════════════════════════════════════════════

def _recv_exactly(sock, n):
    """Receive exactly n bytes from a socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def _recv_frame(sock):
    """Receive [4-byte len][jpeg] → decoded OpenCV frame."""
    length = struct.unpack(">I", _recv_exactly(sock, 4))[0]
    if length > 10_000_000:
        raise ValueError(f"Frame too large: {length}")
    jpeg = _recv_exactly(sock, length)
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("JPEG decode failed")
    return frame


def _send_command(sock, speed, steer):
    """Send [4-byte len][utf-8 'speed,steer'] to Pi."""
    data = f"{speed},{steer}".encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def run_remote(state: SharedState, args):
    """
    Remote mode: TCP frame receiver + command sender.
    Runs as a background thread. Receives JPEG frames from Pi over TCP,
    feeds them into SharedState. Reads speed/steer from SharedState and
    sends them back to Pi. Everything else (perception, planner, display)
    is handled by run_live() on the main thread — identical to local mode.
    """

    # TCP server — wait for Pi
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"[Remote] Listening on 0.0.0.0:{args.port} — run pi_bridge.py on the Pi now")
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[Remote] ✅ Pi connected from {addr}")

    try:
        while state.is_running():
            # 1. Receive frame from Pi → SharedState
            frame = _recv_frame(conn)
            state.set_latest_frame(frame.copy())

            # 2. Read planner's command from SharedState → send to Pi
            speed, steer = state.get_command()
            _send_command(conn, speed, steer)

    except (ConnectionError, ValueError) as e:
        print(f"\n[Remote] Connection lost: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        state.shutdown()
        conn.close()
        srv.close()


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
    parser.add_argument("--speed",       type=int,   default=None,
                        help="Starting cruise speed in cm/s (default 15, max 50)")
    parser.add_argument("--no-lane-safety", action="store_true",
                        help="Disable lane collision recovery and no-lane emergency stop (for testing)")
    parser.add_argument("--remote",      action="store_true",
                        help="Remote mode: receive frames from Pi over TCP instead of local camera")
    parser.add_argument("--port",        type=int,   default=5555,
                        help="TCP port for remote mode (default 5555)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("   🚗  Skynet — Starting Integrated System")
    print(f"   Mode:      {'REMOTE (laptop ← Pi frames)' if args.remote else 'LOCAL (on-device camera)'}")
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

    if args.remote:
        # ── REMOTE MODE: TCP bridge feeds frames into SharedState,
        #    run_live(remote=True) handles perception + display ────────────
        print(f"[Skynet] 📡 Remote mode — waiting for Pi on port {args.port}…")
        bridge = threading.Thread(
            target=run_remote, args=(state, args),
            name="RemoteBridge", daemon=True,
        )
        bridge.start()
        all_threads.append(bridge)

        # run_live reads frames from SharedState (fed by TCP bridge)
        try:
            model_file = args.model or "src/perception/sign_recognition/bfmc_best_shirts.pt"
            run_live(
                model_path=model_file,
                conf=args.conf,
                webcam=args.webcam,
                source=args.source,
                add_traffic_box=True,
                state=state,
                remote=True,
            )
        except KeyboardInterrupt:
            print("\n\n[Skynet] ⛔ KeyboardInterrupt — shutting down…")
            state.shutdown()
    else:
        # ── LOCAL MODE: hand over the main thread to multi.py ─────────────
        try:
            model_file = args.model or "src/perception/sign_recognition/bfmc_best_shirts.pt"
            run_live(
                model_path=model_file,
                conf=args.conf,
                webcam=args.webcam,
                source=args.source,
                add_traffic_box=True,
                state=state,
            )
        except KeyboardInterrupt:
            print("\n\n[Skynet] ⛔ KeyboardInterrupt — shutting down…")
        except Exception as e:
            import traceback
            print(f"\n\n[Skynet] ❌ FATAL CRASH: {e}")
            traceback.print_exc()
        finally:
            state.shutdown()

    for t in all_threads:
        t.join(timeout=3)

    print("[Skynet] ✅ Shutdown complete.")