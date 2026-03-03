# RAVEN — Sign Recognition Module

Live detection of 9 European traffic signs for the Bosch Future Mobility Challenge.

## Signs Detected

| Sign | Description |
|------|-------------|
| 🛑 Stop | Red octagon |
| 🅿️ Parking | Blue square, white P |
| ⬧ Priority | Yellow diamond |
| 🚶 Crosswalk | Blue square, pedestrian figure |
| 🛣️ Highway entrance | Green rectangle, motorway symbol |
| 🚫 Highway exit | Green rectangle + red slash |
| 🔄 Roundabout | Blue circle, circular arrows |
| ⬆️ One-way | Blue square, upward arrow |
| ⛔ No-entry | Red circle, white horizontal bar |

---

## Quick Start (Pi 4 with Display)

### 1. Install dependencies

```bash
# On Raspberry Pi
cd ~/raven-brain-stack
pip install ultralytics opencv-python

# picamera2 should already be installed on Pi OS
```

### 2. Run live detection

```bash
# Using Pi Camera (default)
python3 src/perception/sign_recognition/live_sign_detector.py

# Using webcam (USB or Mac)
python3 src/perception/sign_recognition/live_sign_detector.py --webcam
```

### 3. Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot to `screenshots/` |
| `p` | Pause / Resume |

---

## Models

### Default: YOLOv8n (COCO)

Works **immediately** with no setup — but only detects **Stop signs** (the only European-style sign in COCO).

```bash
python3 live_sign_detector.py
# Automatically downloads yolov8n.pt on first run
```

### Recommended: Custom European Signs Model

For all 9 signs, use a pretrained European traffic signs model:

**Option A — Roboflow Universe (easiest, free):**
1. Create a free account at [roboflow.com](https://roboflow.com)
2. Find a European traffic signs YOLOv8 model on [Roboflow Universe](https://universe.roboflow.com)
3. Download the model weights (`.pt` file)
4. Run:
   ```bash
   python3 live_sign_detector.py --model path/to/eu_signs_best.pt
   ```

**Option B — Train your own (most accurate for BFMC):**
1. Capture images of the miniature track signs with `s` key during live view
2. Upload & annotate at [roboflow.com](https://roboflow.com) or [cvat.ai](https://cvat.ai)
3. Train on your Mac or Google Colab:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.train(data="signs.yaml", epochs=100, imgsz=640)
   ```
4. Copy `runs/detect/train/weights/best.pt` to the Pi

---

## CLI Options

```
--model PATH    Path to YOLO model file (default: yolov8n.pt)
--webcam        Force webcam instead of Pi Camera
--source PATH   Video file or image path for offline testing
--conf FLOAT    Confidence threshold, 0.0–1.0 (default: 0.5)
```

## Examples

```bash
# Test on a single image
python3 live_sign_detector.py --source test_photo.jpg

# Test on a video
python3 live_sign_detector.py --source track_video.mp4

# Lower confidence for more detections
python3 live_sign_detector.py --conf 0.3

# Use custom model with Pi Camera
python3 live_sign_detector.py --model eu_signs_best.pt
```

---

## Architecture

```
src/perception/sign_recognition/
├── sign_detector.py         # Original ROS-based detector
├── live_sign_detector.py    # ⬅ NEW: Standalone live detector
└── README.md                # ⬅ This file
```

The `live_sign_detector.py` is **standalone** — no ROS required. It:
- Auto-detects Pi Camera vs webcam
- Maps any model's labels to the 9 BFMC sign classes via `LABEL_MAP`
- Shows real-time bounding boxes + FPS on the monitor
- Prints detections to the terminal
