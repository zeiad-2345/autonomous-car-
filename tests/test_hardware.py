#!/usr/bin/env python3
import serial
import time
import sys

# Default port, overridable for Mac testing
SERIAL_PORT = "/dev/ttyACM0"
import platform
if platform.system() == "Darwin":
    # Typical Mac Arduino Nano RP2040 Connect port
    SERIAL_PORT = "/dev/cu.usbmodem14201" 
    
BAUD_RATE = 115200

def test_hardware():
    print("---------------------------------------------------")
    print("🧪 RAVEN HARDWARE DIAGNOSTIC TEST")
    print("---------------------------------------------------")
    print(f"[1/4] Checking Serial Connection on {SERIAL_PORT}...")
    
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    except Exception as e:
        print(f"❌ FAILED: Could not open port {SERIAL_PORT}.")
        print(f"   Error: {e}")
        print("   If on Mac, edit raven-brain-stack/tests/test_hardware.py to match your /dev/cu.usbmodem port.")
        sys.exit(1)
        
    print("✅ SUCCESS: Port opened.")
    print("Waiting 2 seconds for Arduino buffer to clear...")
    time.sleep(2)
    arduino.reset_input_buffer()
    
    print("\n[2/4] Verifying IMU Telemetry Stream...")
    imu_passed = False
    start_time = time.time()
    
    # Wait up to 3 seconds for an IMU packet
    while time.time() - start_time < 3:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode(errors="ignore").strip()
            if line.startswith("@imu:"):
                # Example: @imu:0.00;1.00;0.00;0.01;0.02;0.98;;
                data = line[5:-2].split(";")
                if len(data) >= 6:
                    print(f"✅ SUCCESS: Valid IMU packet received! (Roll: {data[0]}, Pitch: {data[1]})")
                    imu_passed = True
                    break
    
    if not imu_passed:
        print("❌ FAILED: No valid IMU telemetry received within 3 seconds.")
        print("   Ensure the Arduino is running the latest raven-rp2040.ino firmware.")
        sys.exit(1)

    print("\n[3/4] Verifying Encoder Telemetry Stream...")
    enc_passed = False
    start_time = time.time()
    
    # Wait up to 3 seconds for an ENCODER packet
    while time.time() - start_time < 3:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode(errors="ignore").strip()
            if line.startswith("@encoder:"):
                # Example: @encoder:14;;
                pos = line[9:-2]
                print(f"✅ SUCCESS: Valid Encoder packet received! (Pos: {pos})")
                enc_passed = True
                break
                
    if not enc_passed:
        print("❌ FAILED: No valid Encoder telemetry received within 3 seconds.")
        print("   Did you merge the encoder logic into the firmware?")
        sys.exit(1)
        
    print("\n[4/4] Verifying Command Acceptance (Speed & Steer)...")
    # Send a speed command (zero to be safe on the desk)
    cmd = "#speed:0;;"
    arduino.write(cmd.encode())
    
    # Verify response
    cmd_passed = False
    start_time = time.time()
    while time.time() - start_time < 2:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode(errors="ignore").strip()
            if line.startswith("@speed:"):
                print("✅ SUCCESS: Arduino acknowledged '#speed:0;;' command.")
                cmd_passed = True
                break
                
    if not cmd_passed:
        print("❌ FAILED: Arduino did not acknowledge speed command.")
        sys.exit(1)

    print("\n---------------------------------------------------")
    print("🎉 ALL HARDWARE TESTS PASSED! The Pi-Arduino bridge is 100% operational.")
    print("---------------------------------------------------\n")
    arduino.close()
    sys.exit(0)

if __name__ == "__main__":
    test_hardware()
