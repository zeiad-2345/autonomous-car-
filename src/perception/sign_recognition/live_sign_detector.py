#!/usr/bin/env python3
"""
RAVEN — Live Traffic Sign Detector
===================================
Detects 9 European traffic signs for the Bosch Future Mobility Challenge.

Signs:  Stop · Parking · Priority · Crosswalk · Highway Entrance
        Highway Exit · Roundabout · One-way · No-entry

Usage (on Pi with display):
    python3 live_sign_detector.py                       # Pi Camera
    python3 live_sign_detector.py --webcam              # Webcam fallback
    python3 live_sign_detector.py --model best.pt       # Custom trained model
    python3 live_sign_detector.py --source video.mp4    # Test on video file
    python3 live_sign_detector.py --source image.jpg    # Test on single image

Controls:
    q - Quit
    s - Save screenshot
    p - Pause / Resume

Requirements:
    pip install ultralytics opencv-python
    # On Pi only: pip install picamera2
"""

import argparse
import sys
import time
import os
from pathlib import Path
from datetime import datetime

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

try:
    from sign_filters import validate_detection
    FILTERS_AVAILABLE = True
except ImportError:
    try:
        from src.perception.sign_recognition.sign_filters import validate_detection
        FILTERS_AVAILABLE = True
    except ImportError:
        FILTERS_AVAILABLE = False

# Global flag — toggled by --no-filters CLI arg
USE_FILTERS = True


# ─── BFMC Target Signs ────────────────────────────────────────────────────────
# Color is BGR for OpenCV
BFMC_SIGNS = {
    "stop":              {"color": (0, 0, 255),    "label": "STOP"},
    "parking":           {"color": (255, 150, 0),  "label": "PARKING"},
    "priority":          {"color": (0, 215, 255),  "label": "PRIORITY"},
    "crosswalk":         {"color": (255, 200, 0),  "label": "CROSSWALK"},
    "highway_entrance":  {"color": (0, 180, 0),    "label": "HWY ENTER"},
    "highway_exit":      {"color": (0, 130, 0),    "label": "HWY EXIT"},
    "roundabout":        {"color": (255, 100, 0),  "label": "ROUNDABOUT"},
    "one_way":           {"color": (255, 50, 50),  "label": "ONE WAY"},
    "no_entry":          {"color": (50, 50, 255),  "label": "NO ENTRY"},
    "green":             {"color": (0, 255, 0),    "label": "GREEN"},
    "red":               {"color": (0, 0, 255),    "label": "RED"},
    "yellow":            {"color": (0, 255, 255),  "label": "YELLOW"},
    "redandyellow":      {"color": (0, 165, 255),  "label": "RED+YELLOW"},
}

# Maps various model label names → our canonical BFMC sign names.
# Add entries here when you switch to a new pretrained model.
LABEL_MAP = {
     "green":          "green",
    "red":            "red",
    "yellow":         "yellow",
    "redandyellow":   "redandyellow",
    "red_yellow":     "redandyellow",
    "red-yellow":     "redandyellow",
    "red and yellow": "redandyellow",
    
    # ── COCO labels (yolov8n.pt) ──
    "stop sign":            "stop",

    # ── Common Roboflow / GTSDB / GTSRB labels ──
    "stop":                 "stop",
    "parking":              "parking",
    "parking_sign":         "parking",
    "p":                    "parking",
    "priority":             "priority",
    "priority_road":        "priority",
    "priority road":        "priority",
    "give_way":             "priority",       # sometimes labeled this way
    "crosswalk":            "crosswalk",
    "pedestrian":           "crosswalk",
    "pedestrian_crossing":  "crosswalk",
    "pedestriancrossing":   "crosswalk",
    "highway":              "highway_entrance",
    "highway_entrance":     "highway_entrance",
    "motorway":             "highway_entrance",
    "motorway_begin":       "highway_entrance",
    "highway_exit":         "highway_exit",
    "motorway_end":         "highway_exit",
    "end_motorway":         "highway_exit",
    "end motorway":         "highway_exit",
    "roundabout":           "roundabout",
    "roundabout_sign":      "roundabout",
    "one_way":              "one_way",
    "one-way":              "one_way",
    "oneway":               "one_way",
    "one way":              "one_way",
    "no_entry":             "no_entry",
    "no-entry":             "no_entry",
    "no entry":             "no_entry",
    "noentry":              "no_entry",
    "do_not_enter":         "no_entry",
    "no_enter":             "no_entry",

    # ── Custom GTSRB Numbers ──
    "12": "priority",
    "14": "stop",
    "27": "crosswalk",
    "38": "highway_entrance", # keeping right
    "39": "highway_entrance", # keeping left
    "17": "no_entry", # horizontal bar
    "15": "no_entry", # blank circle
    "40": "roundabout",
    "35": "one_way",  # ahead only
    "33": "one_way",  # turn right ahead
    "13": "priority", # yield
}


# ─── Camera Abstraction ───────────────────────────────────────────────────────

def get_camera(use_webcam=False):
    """
    Returns an object with .read() -> (bool, frame) and .release().
    Tries Pi Camera first, falls back to webcam.
    """
    if not use_webcam:
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            config = cam.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            cam.configure(config)
            cam.start()
            # Small warm-up delay
            time.sleep(0.5)
            print("📷 Camera: Pi Camera Module")

            class _PiCam:
                def read(self):
                    frame = cam.capture_array()
                    return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                def release(self):
                    cam.stop()

            return _PiCam()
        except (ImportError, RuntimeError) as e:
            print(f"⚠️  Pi Camera not available ({e}), trying webcam...")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("📷 Camera: Webcam")
        return cap

    raise RuntimeError("❌ No camera found! Check your connections.")


# ─── Detection Logic ──────────────────────────────────────────────────────────

def map_label(raw_label, model_path):
    """Map a model's raw label to a BFMC sign name, or return raw if custom model."""
    normalized = raw_label.lower().strip()
    bfmc = LABEL_MAP.get(normalized)
    if bfmc:
        return bfmc, BFMC_SIGNS[bfmc]["color"], True

    # For custom models: show ALL detections (unknown labels in gray)
    if "yolov8n.pt" not in str(model_path):
        return normalized, (180, 180, 180), False

    return None, None, False


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = det["color"]
        label = det["display"]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Label background
        text = f"{label} {det['conf']:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def draw_hud(frame, fps, model_name, num_detections, paused=False):
    """Draw heads-up display overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "RAVEN Sign Detection", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    # Stats
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Model: {model_name}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Signs: {num_detections}", (200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Controls hint
    cv2.putText(frame, "q:Quit  s:Screenshot  p:Pause", (w - 300, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    return frame


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run(args):
    # Model
    model_path = args.model or "yolov8n.pt"
    print(f"🤖 Loading model: {model_path}")
    model = YOLO(model_path)
    model_name = Path(model_path).stem

    # Print model classes for debugging
    class_names = list(model.names.values())
    print(f"📋 Model knows {len(class_names)} classes: {class_names[:15]}{'...' if len(class_names) > 15 else ''}")

    # Check which BFMC signs this model can detect
    detectable = []
    for name in class_names:
        mapped = LABEL_MAP.get(name.lower().strip())
        if mapped:
            detectable.append(mapped)
    detectable = list(set(detectable))
    print(f"🎯 Can detect {len(detectable)}/9 BFMC signs: {detectable}")
    missing = [s for s in BFMC_SIGNS if s not in detectable]
    if missing:
        print(f"⚠️  Missing signs: {missing}")
        if model_path == "yolov8n.pt":
            print("   ℹ️  Using COCO model (stop sign only). For all 9 signs,")
            print("      provide a custom model with --model <path_to_best.pt>")
    print()

    # Source
    if args.source:
        # Image or video file
        img = cv2.imread(args.source)
        if img is not None:
            # Single image mode
            results = model(img, conf=args.conf, verbose=False)
            detections = _extract_detections(results, model, model_path, img)
            annotated = draw_detections(img.copy(), detections)
            annotated = draw_hud(annotated, 0, model_name, len(detections))

            for d in detections:
                print(f"  ✅ {d['display']:20s} ({d['conf']:.0%})")

            cv2.imshow("RAVEN Sign Detection", annotated)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"❌ Cannot open source: {args.source}")
            return
    else:
        cap = get_camera(use_webcam=args.webcam)

    # Live detection loop
    paused = False
    fps_time = time.time()
    fps_count = 0
    fps = 0.0
    last_frame = None

    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(exist_ok=True)

    print("━" * 55)
    print("  🚀  RAVEN Live Sign Detection — RUNNING")
    print("      q = Quit   s = Screenshot   p = Pause")
    print("━" * 55)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if args.source:
                        break
                    continue
                last_frame = frame.copy()

                # Inference
                results = model(frame, conf=args.conf, verbose=False)
                detections = _extract_detections(results, model, model_path, frame)

                # Draw
                annotated = draw_detections(frame, detections)

                # FPS
                fps_count += 1
                elapsed = time.time() - fps_time
                if elapsed >= 1.0:
                    fps = fps_count / elapsed
                    fps_count = 0
                    fps_time = time.time()

                annotated = draw_hud(annotated, fps, model_name, len(detections))

                # Console output for detected signs
                if detections:
                    for d in detections:
                        print(f"  🔍 {d['display']:20s} ({d['conf']:.0%})")

                cv2.imshow("RAVEN Sign Detection", annotated)
            else:
                # While paused, still show the frozen frame
                if last_frame is not None:
                    annotated = last_frame.copy()
                    annotated = draw_hud(annotated, fps, model_name, 0, paused=True)
                    cv2.imshow("RAVEN Sign Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and last_frame is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = screenshot_dir / f"raven_sign_{ts}.jpg"
                cv2.imwrite(str(path), last_frame)
                print(f"  📸 Screenshot saved: {path}")
            elif key == ord('p'):
                paused = not paused
                state = "⏸️  PAUSED" if paused else "▶️  RESUMED"
                print(f"  {state}")

    except KeyboardInterrupt:
        print("\n  ⛔ Interrupted.")

    cap.release()
    cv2.destroyAllWindows()
    print("  👋 Detection stopped.\n")


def _extract_detections(results, model, model_path, frame=None):
    """Extract and map detections from YOLO results.
    
    If USE_FILTERS is True and a frame is provided, each detection is
    validated against shape, color, and size filters before being accepted.
    """
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            raw_label = model.names[cls_id]
            conf = float(box.conf[0])

            mapped, color, is_bfmc = map_label(raw_label, model_path)
            if mapped is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ── Post-Detection Filters ──
            # Validate shape, color, and size to reject false positives
            # (e.g., red shirts, blue cars, tiny noise detections).
            if (USE_FILTERS and FILTERS_AVAILABLE and frame is not None
                    and is_bfmc):
                if not validate_detection(frame, mapped, (x1, y1, x2, y2)):
                    continue  # Rejected by filters

            display = BFMC_SIGNS[mapped]["label"] if is_bfmc else mapped.upper()

            detections.append({
                "sign": mapped,
                "display": display,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "color": color,
            })
    return detections


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAVEN Live Traffic Sign Detector — Bosch Future Mobility Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 live_sign_detector.py                     # Pi Camera + COCO model
  python3 live_sign_detector.py --webcam            # Mac/PC webcam
  python3 live_sign_detector.py --model best.pt     # Custom trained model
  python3 live_sign_detector.py --source test.jpg   # Single image test
  python3 live_sign_detector.py --conf 0.3          # Lower confidence threshold
  python3 live_sign_detector.py --no-filters         # Disable post-detection filters
        """
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLO .pt model (default: yolov8n.pt)")
    parser.add_argument("--webcam", action="store_true",
                        help="Force webcam instead of Pi Camera")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to video file or image for offline testing")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--no-filters", action="store_true",
                        help="Disable post-detection shape/color/size filters")
    args = parser.parse_args()

    if args.no_filters:
        USE_FILTERS = False
        print("⚠️  Post-detection filters DISABLED (raw YOLO output)")

    run(args)
