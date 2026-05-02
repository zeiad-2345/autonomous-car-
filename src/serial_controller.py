#!/usr/bin/env python3
"""
RAVEN — Serial Controller
=========================
Handles serial communication with the Arduino RP2040.

Standalone usage (interactive keyboard control):
    python3 src/serial_controller.py

As an importable module for skynet.py:
    from src.serial_controller import SerialController
    ctrl = SerialController()
    ctrl.start()          # starts background reader thread
    ctrl.send_speed(20)
    ctrl.send_steer(-10)
    ctrl.stop()
"""
import serial
import time
import threading
import socket
#####################################
# === CONFIG ===
LOCAL_HOST = '0.0.0.0'
LOCAL_PORT = 1234
#####################################

#####################################
# === Connect to local ===
def connect_to_local():
    local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to local server at {LOCAL_HOST}:{LOCAL_PORT}...")
    local_socket.bind((LOCAL_HOST, LOCAL_PORT))
    local_socket.listen(1)
    print("Connected to local server!")
    return local_socket
#####################################

SERIAL_PORT = "/dev/ttyACM0"
# If testing on Mac, this might be "/dev/cu.usbmodem14201" or similar
# SERIAL_PORT = "/dev/cu.usbmodem14201"
BAUD_RATE = 115200

def read_telemetry(arduino):
    """
    Background thread to continuously read and parse telemetry
    from the Arduino without blocking the main thread.
    """
    while True:
        try:
            if arduino.in_waiting > 0:
                line = arduino.readline().decode(errors="ignore").strip()
                if line:
                    if line.startswith("@imu:"):
                        # Parse IMU data: @imu:roll;pitch;yaw;ax;ay;az;;
                        data = line[5:-2].split(";")
                        if len(data) >= 6:
                            roll, pitch, yaw, ax, ay, az = data[:6]
                            print(f"\r[IMU] R:{roll} P:{pitch} Y:{yaw} | A: {ax},{ay},{az}", end=" "*20)
                    elif line.startswith("@encoder:"):
                        # Parse encoder data: @encoder:POS;;
                        pos = line[9:-2]
                        print(f"\r[ENC] Position: {pos}", end=" "*40)
                    else:
                        print(f"\r[ARDUINO] {line}")
        except Exception as e:
            print(f"\r[ERROR in Reader] {e}")
            break


class SerialController:
    """
    Manages serial communication with the Arduino RP2040.
    Thread-safe: a background daemon thread continuously reads telemetry
    so that the main thread can send commands without blocking.

    Usage:
        ctrl = SerialController()
        ctrl.start()             # open port and start reader thread
        ctrl.send_speed(20)      # send #speed:20;;
        ctrl.send_steer(-10)     # send #steer:-10;;
        ctrl.stop()              # close port cleanly
    """

    def __init__(self, port=SERIAL_PORT, baud=BAUD_RATE, telemetry_callback=None):
        """
        Args:
            port: Serial device path (default /dev/ttyACM0).
            baud: Baud rate (default 115200).
            telemetry_callback: Optional function(label: str, data: dict) called
                                  whenever a telemetry line (@imu, @encoder) is received.
        """
        self.port = port
        self.baud = baud
        self.telemetry_callback = telemetry_callback
        self._arduino = None
        self._lock = threading.Lock()       # protects serial writes
        self._running = False
        self._reader_thread = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Open the serial port and start the background telemetry reader."""
        try:
            self._arduino = serial.Serial(self.port, self.baud, timeout=1)
        except Exception as e:
            print(f"[Serial] ❌ Failed to open {self.port}: {e}")
            print("[Serial] Note: Change SERIAL_PORT in serial_controller.py if needed.")
            self._arduino = None
            return False

        time.sleep(2)  # Allow Arduino to reset after port open
        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        print(f"[Serial] ✅ Connected to Arduino on {self.port}")
        return True

    def stop(self):
        """Stop the reader thread and close the port."""
        self._running = False
        # Send a stop command before closing as a safety measure
        self.send_speed(0)
        self.send_steer(0)
        if self._arduino and self._arduino.is_open:
            self._arduino.close()
        print("[Serial] Port closed.")

    def send_speed(self, value: int):
        """
        Send a motor command. value: -50 to 50.
        Positive  → DC:FORWARD:{pwm}\n
        Negative  → DC:BACKWARD:{pwm}\n
        Zero      → DC:STOP\n
        PWM is scaled from the 0–50 input range to 0–255.
        """
        if not self.is_connected():
            return
        pwm = int(abs(value) / 50 * 255)
        if value > 0:
            command = f"DC:FORWARD:{pwm}\n"
        elif value < 0:
            command = f"DC:BACKWARD:{pwm}\n"
        else:
            command = "DC:STOP\n"
        with self._lock:
            try:
                self._arduino.write(command.encode())
                print(f"[Serial] TX {command.strip()}")
            except Exception as e:
                print(f"[Serial] Write error: {e}")

    def send_steer(self, value: int):
        """
        Send a servo command. value: -120 to 120.
        """
        if not self.is_connected():
            return
        angle = max(-120, min(120, value))
        command = f"SERVO:{angle}\n"
        with self._lock:
            try:
                self._arduino.write(command.encode())
                print(f"[Serial] TX {command.strip()}")
            except Exception as e:
                print(f"[Serial] Write error: {e}")

    def is_connected(self) -> bool:
        """Returns True if the serial port is open."""
        return self._arduino is not None and self._arduino.is_open

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _send(self, key: str, value):
        """Internal: (legacy helper, not used for DC/SERVO protocol)"""
        pass

    def _read_loop(self):
        """Background thread: continuously reads and parses Arduino telemetry."""
        while self._running:
            try:
                if self._arduino and self._arduino.in_waiting > 0:
                    line = self._arduino.readline().decode(errors="ignore").strip()
                    if not line:
                        continue

                    if line.startswith("@imu:"):
                        # Format: @imu:roll;pitch;yaw;ax;ay;az;;
                        data = line[5:].rstrip(";").split(";")
                        if len(data) >= 6:
                            roll, pitch, yaw, ax, ay, az = data[:6]
                            telemetry = {"roll": roll, "pitch": pitch, "yaw": yaw,
                                         "ax": ax, "ay": ay, "az": az}
                            if self.telemetry_callback:
                                self.telemetry_callback("imu", telemetry)
                            else:
                                print(f"\r[IMU] R:{roll} P:{pitch} Y:{yaw} | A:{ax},{ay},{az}", end="")

                    elif line.startswith("@encoder:"):
                        # Format: @encoder:POS;;
                        pos = line[9:].rstrip(";")
                        telemetry = {"position": pos}
                        if self.telemetry_callback:
                            self.telemetry_callback("encoder", telemetry)
                        else:
                            print(f"\r[ENC] Position: {pos}", end="")
                    else:
                        print(f"\r[Arduino] {line}")
            except Exception as e:
                print(f"\r[Serial] Read error: {e}")
                break


# ─── Standalone Interactive Mode ─────────────────────────────────────────────

def main():
    """Run the serial controller in interactive keyboard mode."""
    print("Opening serial connection...")
    ctrl = SerialController()
    if not ctrl.start():
        return
    print("Connected. Enter 'speed angle' pairs (e.g., '20 -10') or 'exit'.\n")
    ctrl.send_speed(int(0))
    ctrl.send_steer(int(0))
    
    pc_socket = connect_to_local()
    conn,addr = pc_socket.accept()
    while True:
        try:
            time.sleep(0.1)

            try:
                
                # 4. Wait for control from PC
                #print("[TCP] Waiting for PC control...")
                control_data = ""
                while '\n' not in control_data:
                    chunk = conn.recv(1024).decode('utf-8')
                    if not chunk:
                        raise ConnectionError("PC server disconnected.")
                    control_data += chunk

                control_line, _ = control_data.split('\n', 1)
                control_line = control_line.strip()
                print(f"[TCP] Received control: {control_line}")
                
                
                
                speed_str, angle_str = control_line.split(',')
                speed = float(speed_str)
                angle = float(angle_str)

                if not -50 <= speed <= 50:
                    print("Speed must be between -50 and 50")
                    continue
                if not -120 <= angle <= 120:
                    print("Angle must be between -120 and 120")
                    continue

                ctrl.send_speed(int(speed))
                ctrl.send_steer(int(angle))
                print(f"Sent: speed={speed}, steer={angle}")

            except ValueError as e:
                print(e)
                print("Invalid input format. Example: 20 -10")

        except KeyboardInterrupt:
             ctrl.send_speed(int(0))
             ctrl.send_steer(int(0))          
             break
        except:
             ctrl.send_speed(int(0))
             ctrl.send_steer(int(0))
             ctrl.stop()
    print("\nConnection closed.")


if __name__ == "__main__":
    main()
 