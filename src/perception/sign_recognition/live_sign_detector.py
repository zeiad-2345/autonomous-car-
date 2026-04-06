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
import threading
import time
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

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
TRAFFIC_LIGHT_LABELS = {"green", "red", "yellow", "redandyellow"}


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
    "traffic_light":     {"color": (255, 255, 0),  "label": "TRAFFIC LIGHT"},
    "pedestrian":        {"color": (255, 0, 255),  "label": "pedestrian"},
    
}

# Maps various model label names → our canonical BFMC sign names.
# Add entries here when you switch to a new pretrained model.
LABEL_MAP = {
    "pedestrian":     "prerson",
    "person":         "pedestrian",
     "green":          "green",
    "red":            "red",
    "yellow":         "yellow",
    "redandyellow":   "redandyellow",
    "red_yellow":     "redandyellow",
    "red-yellow":     "redandyellow",
    "red and yellow": "redandyellow",
    "traffic light":  "traffic_light",
    
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
import cv2
import numpy as np
import cv2
import numpy as np

##

##
from typing import Tuple


# ===========================================================================
# Sub-model A: Brightness
# ===========================================================================

def _brightness_vote(roi_inner: np.ndarray) -> Tuple[str, float]:
    """
    Returns (state, confidence) from brightness analysis.
    confidence is the gap between winner and second-place (0-1 range).
    """
    gray = cv2.cvtColor(roi_inner, cv2.COLOR_BGR2GRAY)

    p_low, p_high = np.percentile(gray, [5, 95])
    if p_high - p_low < 8:
        return "red", 0.0  # Flat ROI — very low confidence

    gray_s = np.clip(
        (gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)),
        0, 255
    ).astype(np.uint8)
    gray_s = cv2.GaussianBlur(gray_s, (3, 3), 0)

    ih, iw = gray_s.shape
    seg = max(ih // 3, 1)

    regions = {
        "red":    gray_s[0       : seg,     :],
        "yellow": gray_s[seg     : 2 * seg, :],
        "green":  gray_s[2 * seg : ih,      :],
    }

    def peak(r):
        return float(np.percentile(r, 90)) if r.size > 0 else 0.0

    raw = {k: peak(v) for k, v in regions.items()}
    total = sum(raw.values()) or 1.0
    norm = {k: v / total for k, v in raw.items()}  # normalize to [0,1]

    winner = max(norm, key=norm.get)
    winner_score = norm[winner]
    others = [v for k, v in norm.items() if k != winner]
    second = max(others) if others else 0.0

    # confidence = how much the winner dominates
    confidence = winner_score - second

    # redandyellow: top AND middle both bright
    if raw["red"] > 100 and raw["yellow"] > 100 and raw["green"] < 80:
        return "redandyellow", 0.6

    # Low-brightness fallback — unreliable, set low confidence
    if raw[winner] < 60:
        return "red", 0.15

    return winner, confidence


# ===========================================================================
# Sub-model B: HSV Color
# ===========================================================================

_HSV_RANGES = {
    "red": [
        {"h": (0,  10),   "s": (55, 255), "v": (30, 255)},  # lower red
        {"h": (160, 179), "s": (55, 255), "v": (30, 255)},  # upper red
    ],
    "yellow": [
        {"h": (20, 35),   "s": (50, 255), "v": (30, 255)},  # yellow
        {"h": (10, 20),   "s": (70, 255), "v": (30, 255)},  # orange → yellow
    ],
    "green": [
        {"h": (36, 85),   "s": (45, 255), "v": (30, 255)},
    ],
}


def _hsv_mask(hsv_seg: np.ndarray, color: str, dim: bool) -> float:
    h, w = hsv_seg.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for win in _HSV_RANGES[color]:
        v_lo = 15 if dim else win["v"][0]
        lo = np.array([win["h"][0], win["s"][0], v_lo],        dtype=np.uint8)
        hi = np.array([win["h"][1], win["s"][1], win["v"][1]], dtype=np.uint8)
        out = cv2.bitwise_or(out, cv2.inRange(hsv_seg, lo, hi))
    return float(np.count_nonzero(out)) / (h * w) if h * w > 0 else 0.0


def _hsv_vote(roi_inner: np.ndarray) -> Tuple[str, float]:
    """
    Returns (state, confidence) from HSV color analysis.
    """
    hsv = cv2.cvtColor(roi_inner, cv2.COLOR_BGR2HSV)
    dim = float(np.mean(hsv[:, :, 2])) < 60

    ih, iw = hsv.shape[:2]
    seg = max(ih // 3, 1)

    segs = {
        "red":    hsv[0       : seg,     :, :],
        "yellow": hsv[seg     : 2 * seg, :, :],
        "green":  hsv[2 * seg : ih,      :, :],
    }

    # Score: how much of each segment matches its expected color?
    primary = {
        "red":    _hsv_mask(segs["red"],    "red",    dim),
        "yellow": _hsv_mask(segs["yellow"], "yellow", dim),
        "green":  _hsv_mask(segs["green"],  "green",  dim),
    }

    # Cross-segment bonus (glow bleeds into adjacent segments in small ROIs)
    cross_red = sum(_hsv_mask(segs[s], "red",    dim) for s in segs)
    cross_yel = sum(_hsv_mask(segs[s], "yellow", dim) for s in segs)
    cross_grn = sum(_hsv_mask(segs[s], "green",  dim) for s in segs)

    final = {
        "red":    primary["red"]    * 2.5 + cross_red,
        "yellow": primary["yellow"] * 2.5 + cross_yel,
        "green":  primary["green"]  * 2.5 + cross_grn,
    }

    # redandyellow transition
    if primary["red"] > 0.04 and primary["yellow"] > 0.04 and primary["green"] < 0.03:
        return "redandyellow", 0.7

    total = sum(final.values()) or 1.0
    norm = {k: v / total for k, v in final.items()}

    winner = max(norm, key=norm.get)
    others = [v for k, v in norm.items() if k != winner]
    second = max(others) if others else 0.0
    confidence = norm[winner] - second

    # Nothing matched well
    if final[winner] < 0.03:
        return "red", 0.1

    return winner, confidence


# ===========================================================================
# Main Fusion Function
# ===========================================================================

def infer_traffic_light_state(frame: np.ndarray, bbox: tuple) -> str:
    """
    Hybrid fusion traffic light inference.

    Args:
        frame : BGR frame from camera
        bbox  : (x1, y1, x2, y2) bounding box

    Returns:
        "red" | "yellow" | "green" | "redandyellow" | "traffic_light"
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_f, w_f = frame.shape[:2]

    x1 = max(0, min(x1, w_f - 1))
    y1 = max(0, min(y1, h_f - 1))
    x2 = max(x1 + 1, min(x2, w_f))
    y2 = max(y1 + 1, min(y2, h_f))

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "traffic_light"

    rh, rw = roi.shape[:2]
    if rh < 12 or rw < 6:
        return "traffic_light"

    # Crop margins
    mx = max(1, int(rw * 0.18))
    my = max(1, int(rh * 0.05))
    roi_c = roi[my: rh - my, mx: rw - mx]
    if roi_c.size == 0:
        roi_c = roi

    # --- Run both sub-models ---
    b_state, b_conf = _brightness_vote(roi_c)
    h_state, h_conf = _hsv_vote(roi_c)

    # --- Fusion ---

    # Case 1: Both agree — high confidence output
    if b_state == h_state:
        return b_state

    # Case 2: One returned redandyellow — respect it if conf is decent
    if b_state == "redandyellow" and b_conf >= 0.4:
        return "redandyellow"
    if h_state == "redandyellow" and h_conf >= 0.4:
        return "redandyellow"

    # Case 3: Yellow vs Red disagreement
    # This is the #1 bug in the original code. When brightness says red
    # but HSV says yellow → TRUST HSV. Orange/dark-yellow lights look
    # bright-warm in grayscale but their hue is clearly not red.
    if {b_state, h_state} == {"red", "yellow"}:
        # HSV is the tiebreaker for this specific conflict
        return h_state  # trust the hue reading

    # Case 4: Green vs Yellow disagreement
    # Usually caused by yellow-green LEDs. Trust HSV (hue-aware).
    if {b_state, h_state} == {"green", "yellow"}:
        return h_state

    # Case 5: Red vs Green — very unusual, pick higher confidence
    # but default to red if roughly equal (safety bias)
    if {b_state, h_state} == {"red", "green"}:
        if h_conf > b_conf + 0.15:
            return h_state
        return "red"

    # Case 6: General disagreement — weighted vote
    # HSV gets a 1.3× weight because hue is more discriminative than brightness
    HSV_WEIGHT = 1.3
    scores: dict = {}
    for state in ("red", "yellow", "green", "redandyellow"):
        scores[state] = 0.0
    scores[b_state] += b_conf
    scores[h_state] += h_conf * HSV_WEIGHT

    return max(scores, key=scores.get)

##
##
def run(args):
    # Model
    model_path = args.model or "yolov8n.pt"
    print(f"🤖 Loading model: {model_path}")
    model = YOLO(model_path)
    model_name = Path(model_path).stem
    traffic_box_model = None
    if  "yolov8n.pt" not in str(model_path):
        try:
            traffic_box_model = YOLO("yolov8n.pt")
            print("🚦 Extra traffic-box detector enabled: yolov8n.pt")
        except Exception as e:
            print(f"⚠️ Could not load extra traffic-box detector: {e}")

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
            if True:
                extra = traffic_box_model(img, conf=args.conf, verbose=False)
                detections.extend(
                    _extract_detections(
                        extra, traffic_box_model, "yolov8n.pt", img,
                        allowed_labels={"traffic_light","pedestrian"},
                    )
                )
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
                if traffic_box_model is not None:
                    extra = traffic_box_model(frame, conf=args.conf, verbose=False)
                    detections.extend(
                        _extract_detections(
                            extra, traffic_box_model, "yolov8n.pt", frame,
                            allowed_labels={"traffic_light","pedestrian"},
                        )
                    )

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


# ─── Threaded Sign Detector ───────────────────────────────────────────────────

class LiveSignDetectorThread(threading.Thread):
    """SharedState-based threaded sign detector.

    Reads frames from ``state.get_latest_frame()`` and writes detections
    back via ``state.set_signs()``.  Mirrors the same pattern used by
    ``LaneDetectionThread``.
    """

    DETECT_HZ = 30  # max inference rate

    def __init__(
        self,
        state,
        model_path: str = "src/perception/sign_recognition/bfmc_best_shirts.pt",
        conf: float = 0.5,
        add_traffic_box: bool = True,
    ) -> None:
        super().__init__(name="LiveSignDetectorThread", daemon=True)
        self.state = state
        self.model_path = "src/perception/sign_recognition/bfmc_best_shirts.pt"
        self.conf = conf
        self.add_traffic_box = add_traffic_box
        self._interval = 1.0 / self.DETECT_HZ

        self.model = YOLO("src/perception/sign_recognition/bfmc_best_shirts.pt")
        # Always enable traffic-light box detection.
        if "yolov8n.pt" in self.model_path:
            self.traffic_box_model = self.model
        else:
            self.traffic_box_model = YOLO("yolov8n.pt")

    def run(self) -> None:
        print("[Signs] Starting…")
        last_time = 0.0

        while self.state.is_running():
            now = time.time()
            if now - last_time < self._interval:
                time.sleep(0.005)
                continue
            last_time = now

            frame = self.state.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            t0 = time.perf_counter()

            results = self.model(frame, conf=self.conf, verbose=False)
            detections = _extract_detections(results, self.model, self.model_path, frame)

            if self.traffic_box_model is not None:
                extra = self.traffic_box_model(frame, conf=self.conf, verbose=False)
                detections.extend(
                    _extract_detections(
                        extra,
                        self.traffic_box_model,
                        "yolov8n.pt",
                        frame,
                        allowed_labels={"traffic_light","pedestrian"},
                    )
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.state.set_signs({
                "detections": detections,
                "latency_ms": latency_ms,
            })

        print("[Signs] Stopped.")


def _extract_detections(results, model, model_path, frame=None, allowed_labels=None):
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
            if allowed_labels is not None and mapped not in allowed_labels:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if mapped == "traffic_light" and frame is not None:
                mapped = infer_traffic_light_state(frame, (x1, y1, x2, y2))
                if mapped in BFMC_SIGNS:
                    color = BFMC_SIGNS[mapped]["color"]
            if mapped == "pedestrian" and frame is not None:
                if mapped in BFMC_SIGNS:
                    color = BFMC_SIGNS[mapped]["color"]
               
               

            # ── Post-Detection Filters ──
            # Validate shape, color, and size to reject false positives
            # (e.g., red shirts, blue cars, tiny noise detections).
            if (USE_FILTERS and FILTERS_AVAILABLE and frame is not None
                    and is_bfmc and mapped in BFMC_SIGNS):
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
    parser.add_argument("--add-traffic-box", action="store_true",
                        help="Add COCO traffic-light boxes on top of your selected model")
    args = parser.parse_args()

    if args.no_filters:
        USE_FILTERS = False
        print("⚠️  Post-detection filters DISABLED (raw YOLO output)")

    run(args)