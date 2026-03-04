from ultralytics import YOLO
import cv2

model = YOLO("src/perception/sign_recognition/best.pt")
print(model.names)
print("------")
