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
# Using Pi Camera (default) — uses best available model automatically
python3 src/perception/sign_recognition/live_sign_detector.py

# Using webcam (USB or Mac)
python3 src/perception/sign_recognition/live_sign_detector.py --webcam

# Specify a model explicitly
python3 src/perception/sign_recognition/live_sign_detector.py --model src/perception/sign_recognition/bfmc_best_shirts.pt --webcam
```

### 3. Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot to `screenshots/` |
| `p` | Pause / Resume |

---

## 🧠 Model Files — Detailed Guide

All production model weights live in `src/perception/sign_recognition/`. The training outputs (runs, plots, CSVs) live in `runs/detect/bfmc_models/`.

### Production Models (Use These)

| File | Description | mAP50 | mAP50-95 | When to Use |
|------|-------------|-------|----------|-------------|
| `bfmc_best_shirts.pt` | **⭐ RECOMMENDED.** Fine-tuned with negative mining to reject red shirts as stop signs. | 0.933 | 0.843 | Default for competition. Best accuracy + fewest false positives. |
| `bfmc_best.pt` | Base production model. 100 epochs on the original Bosch dataset only. | 0.927 | 0.837 | Fallback if shirt model behaves unexpectedly. |

### Checkpoint Models (For Resuming Training)

| File | Description | When to Use |
|------|-------------|-------------|
| `bfmc_last_shirts.pt` | Last checkpoint from the shirt fine-tuning run (epoch 10/10). | Resume shirt-aware training with more data. |
| `last.pt` | Last checkpoint from the original 100-epoch run (epoch 100/100). | Resume base training with more epochs or data. |

### Training Run Outputs (In `runs/detect/bfmc_models/`)

| Directory | What It Is |
|-----------|------------|
| `sign_detector/` | **Run 1 (Original).** 100 epochs, 561 images, 9 classes. Trained from `yolov8n.pt`. Contains `weights/best.pt`, `weights/last.pt`, `results.csv`, training plots. |
| `sign_detector_shirts/` | **Run 3 (Shirt Fine-tune).** 10 extra epochs on Run 1's `best.pt`, with 600 images (561 signs + 39 shirt backgrounds for negative mining). Contains final production weights. |
| `sign_detector2/` | **Run 2 (Abandoned).** Started from scratch with shirts but was stopped at epoch 14. Not used. Can be safely deleted. |

### Base Model

| File | Description |
|------|-------------|
| `training_workspace/yolov8n.pt` | Ultralytics YOLOv8 Nano pretrained on COCO (80 classes). Used as the starting point for Run 1. Only detects "stop sign" from the 9 BFMC target signs. Auto-downloaded on first use. |

---

## 🔬 Training History

### Run 1: Base Training (sign_detector)
- **Script:** `training_workspace/train_signs.py`
- **Base model:** `yolov8n.pt` (COCO pretrained)
- **Dataset:** Bosch Traffic Signs YOLOv8 (Roboflow export, 561 images, 9 classes)
- **Epochs:** 100
- **Device:** Apple M2 GPU (MPS)
- **Batch size:** 16 | Image size: 640×640
- **Output:** `runs/detect/bfmc_models/sign_detector/`
- **Result:** mAP50 = 0.927

### Run 3: Negative Mining Fine-tune (sign_detector_shirts)
- **Script:** `training_workspace/finetune_shirts.py`
- **Base model:** `bfmc_best.pt` (Run 1 output)
- **Dataset:** Same as Run 1 + 39 red shirt images with empty label files
- **Epochs:** 10 (fine-tune only)
- **Purpose:** Teach the model that red shirts ≠ stop signs
- **Output:** `runs/detect/bfmc_models/sign_detector_shirts/`
- **Result:** mAP50 = 0.933 (improved over base!)

### What is Negative Mining?
The 39 shirt images have **empty `.txt` label files** (0 bytes). In YOLO, an image with an empty label file means "this image exists but contains **no objects**." During training, when the model tries to detect a stop sign in a red shirt image, the empty label corrects it: "No, there's nothing here." Over many iterations, the model learns to distinguish between actual signs and visually similar non-sign objects (like red clothing).

---

## 🏗️ Training Workspace

Located at `training_workspace/`:

| File | Purpose |
|------|---------|
| `train_signs.py` | Start a fresh 100-epoch training run from `yolov8n.pt` |
| `resume_training.py` | Resume an interrupted training run from `last.pt` checkpoint |
| `finetune_shirts.py` | Fine-tune an existing model with the shirt negative mining dataset |
| `fix_yaml.py` | Fix absolute paths in `data.yaml` for your local machine |
| `Bosch Traffic Signs YOLOv8/` | Dataset directory (images + labels for train/valid/test) |
| `Bosch Traffic Signs YOLOv8/data.yaml` | Dataset configuration (class names, paths) |
| `venv/` | Python virtual environment with `ultralytics` installed |

### How to Retrain

```bash
cd training_workspace

# 1. Activate the virtual environment
source venv/bin/activate

# 2. Fix paths if on a new machine
python fix_yaml.py

# 3a. Fresh training (100 epochs from scratch)
python train_signs.py

# 3b. Resume interrupted training
python resume_training.py

# 3c. Fine-tune with negative mining images
python finetune_shirts.py
```

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

# Use the shirt-aware model with Pi Camera
python3 live_sign_detector.py --model src/perception/sign_recognition/bfmc_best_shirts.pt
```

---

## Architecture

```
src/perception/sign_recognition/
├── bfmc_best_shirts.pt      ⭐ Production model (shirt-aware, RECOMMENDED)
├── bfmc_best.pt             Base production model (100 epochs, no shirts)
├── bfmc_last_shirts.pt      Last checkpoint from shirt fine-tuning
├── last.pt                  Last checkpoint from base training
├── live_sign_detector.py    Standalone live detector (webcam/Pi Camera)
├── sign_detector.py         Original ROS-based detector class
├── threads/
│   └── threadSignDetection.py   Threaded detector for the RAVEN message bus
└── README.md                This file
```

The `live_sign_detector.py` is **standalone** — no ROS required. It:
- Auto-detects Pi Camera vs webcam
- Maps any model's labels to the 9 BFMC sign classes via `LABEL_MAP`
- Shows real-time bounding boxes + FPS on the monitor
- Prints detections to the terminal

### Per-Class Accuracy (bfmc_best_shirts.pt)

| Class | mAP50-95 |
|-------|----------|
| Stop | 94.0% |
| Priority | 92.0% |
| Parking | 89.5% |
| No Entry | 88.2% |
| Crosswalk | 83.2% |
| One Way | 81.6% |
| Highway Exit | 80.2% |
| Highway Entrance | 66.0% |
| **Overall mAP50** | **93.3%** |
