#!/usr/bin/env python3
import socket
import time
import numpy as np
import cv2
from picamera2 import Picamera2

from control.Stanley_Control import *

#####################################
# === CONFIG ===
PC_HOST = '192.168.1.164'  # Change as needed
PC_PORT = 4000

LOCAL_HOST = '0.0.0.0'
LOCAL_PORT = 1234
#####################################

#####################################
# === Image Frame Config
FRAME_WIDTH = 640*2
FRAME_HEIGHT = 480*2

# === Camera Initialization and Configuration ===
cam = Picamera2()
cam.preview_configuration.main.size = (FRAME_WIDTH,FRAME_HEIGHT)
cam.preview_configuration.main.format = "RGB888"
cam.configure("preview")
cam.start()
time.sleep(1)  # give time to warm up
#####################################

#####################################
# === Connect to PC ===
def connect_to_pc():
    pc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to PC server at {PC_HOST}:{PC_PORT}...")
    pc_socket.connect((PC_HOST, PC_PORT))
    print("Connected to PC server!")
    return pc_socket
#####################################

#####################################
# === Connect to local ===
def connect_to_local():
    local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to local server at {LOCAL_HOST}:{LOCAL_PORT}...")
    local_socket.connect((LOCAL_HOST, LOCAL_PORT))
    print("Connected to local server!")
    return local_socket
#####################################


#####################################
def main(args=None):
    pc_socket = connect_to_pc()
    
    local_socket = connect_to_local()

    while True:
        try:
            # 3. Forward to PC the frame
            # Wait for server request
            request = pc_socket.recv(1)
            if not request or request != b'F':
            	print("[CLIENT] Invalid or no frame request.")
            	break
            
            #Capture Pi Frame
            frame = cam.capture_array()
            
            # === Resize frame ===
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # Encode as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
            	print("[CLIENT] JPEG encoding failed.")
            	break
            
            image_bytes = jpeg.tobytes()
            image_len = len(image_bytes)
            
            # === Send frame ===
            pc_socket.sendall(image_len.to_bytes(8, byteorder='big'))
            pc_socket.sendall(image_bytes)

            # 4. Wait for control from PC
            print("[TCP] Waiting for PC control...")
            control_data = ""
            while '\n' not in control_data:
                chunk = pc_socket.recv(1024).decode('utf-8')
                if not chunk:
                    raise ConnectionError("PC server disconnected.")
                control_data += chunk

            control_line, _ = control_data.split('\n', 1)
            control_line = control_line.strip()
            print(f"[TCP] Received control: {control_line}")

            # 5. Decode control
            n_str, th_str = control_line.split(',')
            n = float(n_str)
            th = float(th_str)
            
            speed, steer = Control_Code(n, th)
            act_msg = f"{speed},{steer}\n"
            local_socket.sendall(act_msg.encode('utf-8'))
            
            print(f"[ACTUATE] speed: {speed:.0f} | Servo: {steer:.1f}")


            time.sleep(0.05)

        except Exception as e:
            print(f"[ERROR] {e}")
            break
    pc_socket.close()
    print("Closed all connections.")
#####################################


#####################################
# === ENTRY POINT ===
if __name__ == "__main__":
    main()    
#####################################
