from roboflow import Roboflow
import shutil
import os

rf = Roboflow(api_key="L0G3avkLqRdIYleyaBk5")
project = rf.workspace("unlam-swoow").project("bosch-traffic-signs")
version = project.version(4)

print("Starting download...")
# This downloads the YOLOv8 dataset and the trained model weights if available
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")
# Usually it puts weights in {dataset.location}/runs/detect/train/weights/best.pt or something similar.
