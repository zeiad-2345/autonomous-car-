import sys
import argparse
print("Starting...")
sys.stdout.flush()

from ultralytics import YOLO
import cv2
print("Imported everything")
sys.stdout.flush()

model = YOLO("src/perception/sign_recognition/best.pt")
print("Model loaded")
sys.stdout.flush()

cap = cv2.VideoCapture(0)
print(f"Cap opened: {cap.isOpened()}")
sys.stdout.flush()

if cap.isOpened():
    ret, frame = cap.read()
    print(f"Read frame: {ret}")
    if ret:
        results = model(frame, verbose=False)
        print(f"Inference complete: {len(results[0].boxes)} detections")
        sys.stdout.flush()

cap.release()
print("Done")
