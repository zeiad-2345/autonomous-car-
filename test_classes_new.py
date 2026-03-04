from ultralytics import YOLO
model = YOLO("src/perception/sign_recognition/bfmc_best.pt")
print(model.names)
