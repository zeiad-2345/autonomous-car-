from ultralytics import YOLO
print("Imported YOLO")
model = YOLO("src/perception/sign_recognition/best.pt")
print("Loaded model")
