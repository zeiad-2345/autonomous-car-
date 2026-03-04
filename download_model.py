import urllib.request
import os

# Public direct download link for a YOLOv8n BFMC model weights file
url = "https://github.com/dudas13/bosch-future-mobility-challenge/raw/main/models/yolov8n_best.pt"

try:
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, "src/perception/sign_recognition/bfmc_best.pt")
    size = os.path.getsize("src/perception/sign_recognition/bfmc_best.pt")
    if size < 1000000:
        print(f"File is too small ({size} bytes). Probably an LFS pointer or error page.")
    else:
        print("Download successful!")
except Exception as e:
    print(f"Failed to download: {e}")

