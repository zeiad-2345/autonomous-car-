#!/usr/bin/env python3
"""
Skynet PI — Raspberry Pi Side (Camera + Arduino)
=================================================
Runs on the Raspberry Pi. Captures camera frames, streams them to the laptop
over TCP, receives speed/steer commands back, and forwards them to Arduino.

Architecture:
  Pi Camera → TCP → [Laptop does YOLO + Lane + Planner] → TCP → Arduino

Usage (on the Pi):
    python3 src/skynet_pi.py --laptop-ip 192.168.1.10
    python3 src/skynet_pi.py --laptop-ip 192.168.1.10 --no-arduino   # dry-run
    python3 src/skynet_pi.py --laptop-ip 192.168.1.10 --port 5555

On the Laptop — run skynet_laptop.py to receive frames and send commands.
"""

import sys
import os
import time
import struct
import threading
import argparse
import socket
import signal

import cv2
import numpy as np

# ── Serial controller import ──────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from serial_controller import SerialController
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[Pi] ⚠️  serial_controller not found. Arduino control disabled.")

# ── Pi Camera import ──────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False
    print("[Pi] ⚠️  picamera2 not found. Will use OpenCV webcam fallback.")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80      # 0-100, higher = better quality, bigger frames
SEND_HZ      = 20      # target frame rate


# ══════════════════════════════════════════════════════════════════════════════
# Camera Helpers
# ══════════════════════════════════════════════════════════════════════════════

def open_picamera(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Open Raspberry Pi Camera2 and return a capture function."""
    cam = Picamera2()
    cam.preview_configuration.main.size = (width, height)
    cam.preview_configuration.main.format = "RGB888"
    cam.configure("preview")
    cam.start()
    time.sleep(1)
    print(f"[Pi] Pi Camera opened: {width}x{height}")
    return cam


def open_webcam(index=0, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Fallback: open an OpenCV webcam."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"[Pi] ❌ Cannot open webcam index {index}")
        sys.exit(1)
    print(f"[Pi] Webcam opened: index={index}")
    return cap


# ══════════════════════════════════════════════════════════════════════════════
# TCP Helpers (length-prefixed binary protocol)
# ══════════════════════════════════════════════════════════════════════════════

def send_frame(sock, jpeg_bytes):
    """Send a JPEG frame: [4-byte length][jpeg bytes]"""
    length = len(jpeg_bytes)
    sock.sendall(struct.pack(">I", length))
    sock.sendall(jpeg_bytes)


def recv_exactly(sock, n):
    """Receive exactly n bytes from socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def recv_command(sock):
    """Receive a command: [4-byte length][utf-8 string]"""
    length_bytes = recv_exactly(sock, 4)
    length = struct.unpack(">I", length_bytes)[0]
    if length > 4096:  # sanity check
        raise ValueError(f"Command too large: {length} bytes")
    data = recv_exactly(sock, length)
    return data.decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Skynet Pi — Camera + Arduino bridge")
    parser.add_argument("--laptop-ip", type=str, required=True,
                        help="IP address of the laptop running skynet_laptop.py")
    parser.add_argument("--port", type=int, default=5555,
                        help="TCP port (default 5555)")
    parser.add_argument("--no-arduino", action="store_true",
                        help="Run without Arduino (dry-run)")
    parser.add_argument("--webcam", action="store_true",
                        help="Force OpenCV webcam instead of Pi Camera")
    parser.add_argument("--webcam-index", type=int, default=0,
                        help="OpenCV webcam index (default 0)")
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY,
                        help=f"JPEG quality 0-100 (default {JPEG_QUALITY})")
    return parser.parse_args()


def main():
    args = parse_args()
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        print(f"\n[Pi] 🛑 Signal {sig} — shutting down…")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Open camera ───────────────────────────────────────────────────────
    picam = None
    webcam = None
    if not args.webcam and PICAM_AVAILABLE:
        picam = open_picamera()
    else:
        webcam = open_webcam(args.webcam_index)

    def grab_frame():
        if picam is not None:
            frame = picam.capture_array()
            # picam2 returns RGB, OpenCV expects BGR for imencode
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ok, frame = webcam.read()
            if not ok:
                return None
        # Resize to target resolution
        h, w = frame.shape[:2]
        if w != FRAME_WIDTH or h != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        return frame

    # ── Open Arduino serial ───────────────────────────────────────────────
    serial_ctrl = None
    if not args.no_arduino and SERIAL_AVAILABLE:
        serial_ctrl = SerialController()
        if not serial_ctrl.start():
            print("[Pi] ⚠️  Arduino not connected. Running in dry-run mode.")
            serial_ctrl = None

    # ── Connect to laptop ─────────────────────────────────────────────────
    print(f"[Pi] Connecting to laptop at {args.laptop_ip}:{args.port} …")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect((args.laptop_ip, args.port))
    except (socket.timeout, ConnectionRefusedError) as e:
        print(f"[Pi] ❌ Cannot connect to laptop: {e}")
        print("[Pi] Make sure skynet_laptop.py is running on the laptop first!")
        sys.exit(1)
    sock.settimeout(None)
    print("[Pi] ✅ Connected to laptop!")

    # ── Main loop: send frames, receive commands ──────────────────────────
    frame_count = 0
    t0 = time.time()
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    print("=" * 60)
    print(f"   🚗  Skynet Pi — Streaming to {args.laptop_ip}:{args.port}")
    print(f"   Arduino: {'ENABLED' if serial_ctrl else 'DISABLED (dry-run)'}")
    print(f"   Camera:  {'Pi Camera' if picam else 'Webcam'}")
    print("=" * 60)

    try:
        while running:
            loop_start = time.time()

            # 1. Capture frame
            frame = grab_frame()
            if frame is None:
                continue

            # 2. Encode to JPEG
            ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            # 3. Send frame to laptop
            try:
                send_frame(sock, jpeg.tobytes())
            except (BrokenPipeError, ConnectionError):
                print("[Pi] ❌ Laptop disconnected.")
                break

            # 4. Receive speed/steer command from laptop
            try:
                cmd_str = recv_command(sock)
                # Expected format: "speed,steer"
                parts = cmd_str.strip().split(",")
                speed = int(float(parts[0]))
                steer = int(float(parts[1]))
            except (ConnectionError, ValueError) as e:
                print(f"[Pi] ❌ Command error: {e}")
                break

            # 5. Send to Arduino
            if serial_ctrl:
                serial_ctrl.send_speed(speed)
                serial_ctrl.send_steer(steer)

            mode = "📡" if serial_ctrl else "🔌 DRY"
            frame_count += 1
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\r[Pi] {mode} speed={speed:+3d} steer={steer:+3d} | frame#{frame_count} | {fps:.1f} FPS   ", end="", flush=True)

            # Throttle to target FPS
            dt = time.time() - loop_start
            sleep_time = (1.0 / SEND_HZ) - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"\n[Pi] ❌ Error: {e}")
    finally:
        # Safety: stop the car
        if serial_ctrl:
            serial_ctrl.send_speed(0)
            serial_ctrl.send_steer(0)
            serial_ctrl.stop()

        sock.close()
        if webcam is not None:
            webcam.release()
        print("\n[Pi] Shutdown complete.")


if __name__ == "__main__":
    main()
