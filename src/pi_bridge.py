#!/usr/bin/env python3
"""
Pi Bridge — Runs on the Raspberry Pi
=====================================
Captures camera frames, streams them to the laptop (where skynet.py runs
in --remote mode), receives speed/steer commands back, and sends them to Arduino.

Usage:
    python3 src/pi_bridge.py --laptop-ip 192.168.1.10
    python3 src/pi_bridge.py --laptop-ip 192.168.1.10 --no-arduino
    python3 src/pi_bridge.py --laptop-ip 192.168.1.10 --port 5555

On the laptop (run FIRST):
    python3 src/skynet.py --remote --port 5555 --cruise
"""

import sys
import os
import time
import struct
import argparse
import socket
import signal

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from serial_controller import SerialController
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[Pi] serial_controller not found. Arduino disabled.")

try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────
FRAME_W  = 320
FRAME_H  = 240
JPEG_Q   = 50
TARGET_HZ = 30


def parse_args():
    p = argparse.ArgumentParser(description="Pi Bridge — stream frames to laptop")
    p.add_argument("--laptop-ip", type=str, required=True)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--no-arduino", action="store_true")
    p.add_argument("--webcam", action="store_true")
    p.add_argument("--webcam-index", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=JPEG_Q)
    return p.parse_args()


def recv_exactly(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def main():
    args = parse_args()
    running = True

    def on_signal(sig, _):
        nonlocal running
        print(f"\n[Pi] Signal {sig} — stopping")
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # ── Camera ────────────────────────────────────────────────────────────
    picam = None
    webcam_cap = None
    if not args.webcam and PICAM_AVAILABLE:
        picam = Picamera2()
        picam.preview_configuration.main.size = (FRAME_W, FRAME_H)
        picam.preview_configuration.main.format = "RGB888"
        picam.configure("preview")
        picam.start()
        time.sleep(1)
        print(f"[Pi] Pi Camera ready ({FRAME_W}x{FRAME_H})")
    else:
        webcam_cap = cv2.VideoCapture(args.webcam_index)
        webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not webcam_cap.isOpened():
            print("[Pi] Cannot open webcam"); sys.exit(1)
        print(f"[Pi] Webcam ready (index {args.webcam_index})")

    def grab():
        if picam:
            f = picam.capture_array()
            return cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        ok, f = webcam_cap.read()
        return f if ok else None

    # ── Arduino ───────────────────────────────────────────────────────────
    serial_ctrl = None
    if not args.no_arduino and SERIAL_AVAILABLE:
        serial_ctrl = SerialController()
        if not serial_ctrl.start():
            print("[Pi] Arduino not found, dry-run mode")
            serial_ctrl = None

    # ── Connect to laptop ─────────────────────────────────────────────────
    print(f"[Pi] Connecting to {args.laptop_ip}:{args.port} …")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(10)
    try:
        sock.connect((args.laptop_ip, args.port))
    except (socket.timeout, ConnectionRefusedError) as e:
        print(f"[Pi] Cannot connect: {e}")
        print("[Pi] Start skynet.py --remote on laptop first!")
        sys.exit(1)
    sock.settimeout(None)
    print("[Pi] Connected!")

    enc_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    n = 0
    t0 = time.time()

    print("=" * 50)
    print(f"  Streaming to {args.laptop_ip}:{args.port}")
    print(f"  Arduino: {'ON' if serial_ctrl else 'OFF'}")
    print("=" * 50)

    try:
        while running:
            loop_t = time.time()
            frame = grab()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            if w != FRAME_W or h != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            ok, jpeg = cv2.imencode(".jpg", frame, enc_params)
            if not ok:
                continue

            # Send frame
            data = jpeg.tobytes()
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)

            # Receive command
            length = struct.unpack(">I", recv_exactly(sock, 4))[0]
            if length > 4096:
                break
            cmd = recv_exactly(sock, length).decode("utf-8")
            parts = cmd.split(",")
            speed = int(float(parts[0]))
            steer = int(float(parts[1]))

            if serial_ctrl:
                serial_ctrl.send_speed(speed)
                serial_ctrl.send_steer(steer)

            n += 1
            fps = n / (time.time() - t0) if time.time() > t0 else 0
            tag = "TX" if serial_ctrl else "DRY"
            print(f"\r[Pi] {tag} spd={speed:+3d} str={steer:+3d} | #{n} {fps:.1f}fps  ", end="", flush=True)

            dt = time.time() - loop_t
            wait = (1.0 / TARGET_HZ) - dt
            if wait > 0:
                time.sleep(wait)

    except (ConnectionError, BrokenPipeError) as e:
        print(f"\n[Pi] Lost connection: {e}")
    except Exception as e:
        print(f"\n[Pi] Error: {e}")
    finally:
        if serial_ctrl:
            serial_ctrl.send_speed(0)
            serial_ctrl.send_steer(0)
            serial_ctrl.stop()
        sock.close()
        if webcam_cap:
            webcam_cap.release()
        print("\n[Pi] Done.")


if __name__ == "__main__":
    main()