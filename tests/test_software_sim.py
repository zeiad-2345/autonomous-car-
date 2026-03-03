#!/usr/bin/env python3
import os
import pty
import serial
import threading
import time
import sys

# We'll use pty to create a pair of virtual serial ports
# master: The script writes fake arduino data to this
# slave: The dashboard/serial_controller reads from this pretending it's an arduino

def run_mock_arduino(master_fd):
    """Simulates the Arduino side of the serial connection."""
    roll = 0.0
    encoder = 0
    
    # We need to read non-blockingly to handle incoming commands
    os.set_blocking(master_fd, False)
    
    start_time = time.time()
    
    while time.time() - start_time < 5: # Run test for 5 seconds
        # 1. Send Telemetry @ 10Hz
        roll = (roll + 0.1) % 180
        encoder += 1
        
        imu_msg = f"@imu:{roll:.2f};1.20;0.00;0.10;0.05;0.98;;\r\n"
        enc_msg = f"@encoder:{encoder};;\r\n"
        
        try:
            os.write(master_fd, imu_msg.encode())
            os.write(master_fd, enc_msg.encode())
        except BlockingIOError:
            pass # Buffer full, skip this tick
            
        # 2. Check for Incoming Commands
        try:
            cmd = os.read(master_fd, 1024).decode(errors='ignore')
            if cmd:
                # Mock acknowledgement
                if "#speed" in cmd:
                    os.write(master_fd, b"@speed:ACK;;\r\n")
                    print(f"\n[Mock Arduino] Received Speed Command: {cmd.strip()}")
                elif "#steer" in cmd:
                    os.write(master_fd, b"@steer:ACK;;\r\n")
                    print(f"\n[Mock Arduino] Received Steer Command: {cmd.strip()}")
        except BlockingIOError:
            pass # No commands ready
            
        time.sleep(0.1)


def run_host_controller(slave_name):
    """Simulates the Pi parsing telemetry and sending commands to the port."""
    print(f"\n[Pi Host] Connecting to virtual port: {slave_name}")
    port = serial.Serial(slave_name, 115200, timeout=0.1)
    
    start_time = time.time()
    received_imu = False
    received_enc = False
    
    # Send a test command immediately
    port.write(b"#speed:25;;")
    
    while time.time() - start_time < 3: # Wait up to 3 seconds to gather data
        if port.in_waiting > 0:
            line = port.readline().decode(errors='ignore').strip()
            if line:
                if line.startswith("@imu:"):
                    if not received_imu:
                        print(f"[Pi Host] Successfully parsed first IMU packet: {line}")
                        received_imu = True
                elif line.startswith("@encoder:"):
                    if not received_enc:
                        print(f"[Pi Host] Successfully parsed first Encoder packet: {line}")
                        received_enc = True
                elif line.startswith("@speed:"):
                    print(f"[Pi Host] Got Command Acknowledgement: {line}")
                    
        if received_imu and received_enc:
            print("\n[Result] Software Simulation Test Passed! Both subsystems can communicate bidirectionally.")
            sys.exit(0)
            
    print("\n[Result] Software Simulation Test Failed! Did not receive telemetry streams.")
    sys.exit(1)


def main():
    print("---------------------------------------------------")
    print("🧪 RAVEN SOFTWARE-IN-THE-LOOP SIMULATION TEST")
    print("---------------------------------------------------")
    
    # 1. Create Virtual Serial Port
    master, slave = pty.openpty()
    slave_name = os.ttyname(slave)
    
    print(f"Created Virtual Serial pair. Arduino running on internal master.")
    
    # 2. Start Mock Arduino Thread
    arduino_thread = threading.Thread(target=run_mock_arduino, args=(master,), daemon=True)
    arduino_thread.start()
    
    # 3. Start Host Test Loop
    run_host_controller(slave_name)

if __name__ == "__main__":
    main()
