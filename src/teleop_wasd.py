#!/usr/bin/env python3
"""
Teleop WASD
Streamlines camera feed and takes over the SSH input for manual Arduino control.
Runs in the foreground as part of `raven start manual`.
"""
import sys
import tty
import termios
import threading
import time
import socket
import argparse
import signal
import cv2

try:
    from serial_controller import SerialController
except ImportError:
    try:
        from src.serial_controller import SerialController
    except ImportError:
        print("[Teleop] SerialController not found. Cannot control Arduino.")
        sys.exit(1)

try:
    import picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

STREAM_PORT = 5012

class ManualTeleop:
    def __init__(self, laptop_ip=None, no_stream=False, webcam_index=0):
        self.laptop_ip = laptop_ip
        self.no_stream = no_stream
        self.webcam_index = webcam_index

        self.speed = 0
        self.steer = 0
        self.running = True

        self.ctrl = SerialController(telemetry_callback=self._on_telemetry)
        if not self.ctrl.start():
            print("[Teleop] ❌ Failed to connect to Arduino.")
            
        self.stream_socket = None
        self.cap = None
        self.cam = None

    def _on_telemetry(self, label, data):
        # We can just silently swallow or print minimal telemetry
        pass

    def _init_camera(self):
        if PICAM_AVAILABLE:
            try:
                self.cam = picamera2.Picamera2()
                config = self.cam.create_preview_configuration(
                    main={"format": "RGB888", "size": (640, 480)},
                )
                self.cam.configure(config)
                self.cam.start()
                return
            except Exception as e:
                print(f"[Teleop] Pi Camera failed: {e}. Falling back to OpenCV.")

        self.cap = cv2.VideoCapture(self.webcam_index)

    def _init_stream(self):
        if self.no_stream or not self.laptop_ip:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((self.laptop_ip, STREAM_PORT))
            sock.settimeout(0.2)
            self.stream_socket = sock
            print(f"[Teleop] ✅ Video stream connected to {self.laptop_ip}:{STREAM_PORT}")
        except Exception as e:
            print(f"[Teleop] ⚠️ Stream not connected ({e}).")

    def camera_loop(self):
        self._init_camera()
        self._init_stream()
        
        while self.running:
            frame = None
            if self.cam is not None and PICAM_AVAILABLE:
                try:
                    frame = self.cam.capture_array()
                except Exception:
                    pass
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: frame = None
                
            if frame is not None and self.stream_socket:
                try:
                    req = self.stream_socket.recv(1)
                    if req == b'F':
                        resized = cv2.resize(frame, (640, 480))
                        # Draw a small HUD
                        cv2.putText(resized, f"MANUAL | Spd: {self.speed} Str: {self.steer}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        _, jpg = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        data = jpg.tobytes()
                        self.stream_socket.sendall(len(data).to_bytes(8, 'big'))
                        self.stream_socket.sendall(data)
                        self.stream_socket.recv(1024)
                except socket.timeout:
                    pass
                except Exception:
                    self.stream_socket.close()
                    self.stream_socket = None
            time.sleep(0.05)
            
        if self.cam is not None and PICAM_AVAILABLE:
            self.cam.stop()
        if self.cap:
            self.cap.release()
        if self.stream_socket:
            self.stream_socket.close()

    def get_char(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def run(self):
        cam_thread = threading.Thread(target=self.camera_loop, daemon=True)
        cam_thread.start()
        
        print("\r\n====================================")
        print("        🎮 MANUAL TELEOP 🎮         ")
        print("          W: Accelerate             ")
        print("  A: Steer Left     D: Steer Right  ")
        print("          S: Reverse                ")
        print("                                    ")
        print("      Space/X: Brake  | Q: Quit     ")
        print("====================================\r\n")
        
        while self.running:
            ch = self.get_char().lower()
            if ch == 'q' or ch == '\x03': # ctrl+c
                self.running = False
                break
            elif ch == 'w':
                self.speed = min(50, self.speed + 5)
            elif ch == 's':
                self.speed = max(-50, self.speed - 5)
            elif ch == 'a':
                self.steer = max(-25, self.steer - 5)
            elif ch == 'd':
                self.steer = min(25, self.steer + 5)
            elif ch == ' ' or ch == 'x':
                self.speed = 0
                self.steer = 0
                
            self.ctrl.send_speed(self.speed)
            self.ctrl.send_steer(self.steer)
            print(f"\r[Teleop] Spd: {self.speed:3d} | Str: {self.steer:3d}     ", end="")
            
        self.ctrl.send_speed(0)
        self.ctrl.send_steer(0)
        self.ctrl.stop()
        print("\r\n[Teleop] Exiting manual mode.\r\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--laptop-ip", type=str, default=None)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--webcam-index", type=int, default=0)
    args = parser.parse_args()
    
    teleop = ManualTeleop(laptop_ip=args.laptop_ip, no_stream=args.no_stream, webcam_index=args.webcam_index)
    
    def handle_sig(sig, frame):
        teleop.running = False
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)
    
    teleop.run()
