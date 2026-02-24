#!/usr/bin/env python3
import serial
import time
import threading

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

def send_command(arduino, key, value):
    command = f"#{key}:{value};;"
    arduino.write(command.encode())
    print(f"\nSent: {command}")

def main():
    print("Opening serial connection...")
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"Failed to open port {SERIAL_PORT}: {e}")
        print("Note: If testing on Mac, change SERIAL_PORT to your /dev/cu.usbmodem...")
        return

    time.sleep(2)  # Allow Arduino to reset
    print("Connected.\n")

    # Start the background reader thread
    reader_thread = threading.Thread(target=read_telemetry, args=(arduino,), daemon=True)
    reader_thread.start()

    while True:
        try:
            # Short sleep to not clash immediately with prints
            time.sleep(0.5)
            print("\n") 
            user_input = input(
                "Enter speed (-50 to 50) and angle (-25 to 25) separated by space (or 'exit'): "
            ).strip()

            if user_input.lower() == "exit":
                break

            try:
                speed_str, angle_str = user_input.split()
                speed = int(speed_str)
                angle = int(angle_str)

                # Validate ranges according to firmware
                if not -50 <= speed <= 50:
                    print("Speed must be between -50 and 50")
                    continue

                if not -25 <= angle <= 25:
                    print("Angle must be between -25 and 25")
                    continue

                # Send commands separately (as firmware expects)
                send_command(arduino, "speed", speed)
                send_command(arduino, "steer", angle)

            except ValueError:
                print("Invalid input format. Example: 20 -10")
        
        except KeyboardInterrupt:
            break

    arduino.close()
    print("\nConnection closed.")

if __name__ == "__main__":
    main()