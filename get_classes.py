from ultralytics import YOLO
model = YOLO("src/perception/sign_recognition/best.pt")
print(model.names)
