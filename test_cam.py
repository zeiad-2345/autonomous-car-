import cv2
import time
print("Imported cv2")
cap = cv2.VideoCapture(0)
print(f"Cap opened: {cap.isOpened()}")
if cap.isOpened():
    ret, frame = cap.read()
    print(f"Read frame: {ret}")
cap.release()
print("Done")
