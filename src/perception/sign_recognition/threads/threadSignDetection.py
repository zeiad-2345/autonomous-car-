import time
import json
import base64

import cv2
import numpy as np

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import mainCamera, SignDetected
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from src.perception.sign_recognition.sign_filters import validate_detection
    FILTERS_AVAILABLE = True
except ImportError:
    FILTERS_AVAILABLE = False


# ─── Label mapping: model output → BFMC sign name ────────────────────────────
LABEL_MAP = {
    # COCO
    "stop sign":            "stop",
    # Roboflow / GTSDB / custom models
    "stop":                 "stop",
    "parking":              "parking",
    "parking_sign":         "parking",
    "p":                    "parking",
    "priority":             "priority",
    "priority_road":        "priority",
    "priority road":        "priority",
    "give_way":             "priority",
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
}


import socket
import struct

class threadSignDetection(ThreadWithStop):
    """Thread that continuously reads camera frames and runs YOLOv8 sign detection.

    Subscribes to the mainCamera message to get frames.
    Publishes SignDetected messages with the detected sign name and confidence.
    Streams annotated frames to the Dashboard via a TCP Socket.
    """

    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging

        # TCP Video Streaming Setup
        self.laptop_ip = '10.105.27.45'  # Default, should match Dashboard IP
        self.stream_socket = None
        self._init_video_stream()

        # Load model
        self.model = None
        self.model_path = "yolov8n.pt"
        self._load_model()

        # Confidence threshold
        self.conf_threshold = 0.5

        # Rate limiting: don't run inference on every single frame
        self.last_inference_time = 0
        self.inference_interval = 0.2  # seconds (5 FPS inference)

        # Subscribe to camera frames
        self.cameraSubscriber = messageHandlerSubscriber(
            self.queuesList, mainCamera, "lastOnly", True
        )

        # Publish detected signs
        self.signPublisher = messageHandlerSender(
            self.queuesList, SignDetected
        )

        super(threadSignDetection, self).__init__()

    def _init_video_stream(self):
        """Initializes a TCP connection to send annotated frames to the laptop."""
        try:
            self.stream_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Short timeout to avoid blocking the thread if the laptop isn't listening yet
            self.stream_socket.settimeout(0.5) 
            self.stream_socket.connect((self.laptop_ip, 5012))
            
            # Reset timeout so the recv('F') loop doesn't instantly die, but keep it low
            # so we drop frames instead of lagging the car if the network stutters.
            self.stream_socket.settimeout(0.2) 
            self.logging.info(f"✅ Video Stream: Connected to Laptop {self.laptop_ip}:5012")
        except Exception as e:
            self.logging.warning(f"Video Stream: Client not found on {self.laptop_ip}:5012 ({e}). Annotations will not be streamed.")
            self.stream_socket = None

    def _load_model(self):
        """Load the YOLO model. Tries custom model first, falls back to COCO."""
        if YOLO is None:
            self.logging.warning(
                "⚠️ ultralytics not installed! Sign detection disabled. "
                "Run: pip install ultralytics"
            )
            return

        import os
        # ── Model File Lookup (Priority Order) ──────────────────────────
        # The detector tries each path in order and uses the first one found.
        #
        # Model files explained:
        #   bfmc_best_shirts.pt  — RECOMMENDED. Fine-tuned from bfmc_best.pt
        #                          with 10 extra epochs of negative mining
        #                          (39 red shirt images as backgrounds).
        #                          Reduces false positives on red clothing.
        #                          Dataset: 600 images (561 signs + 39 shirts).
        #                          Final mAP50: 0.933, mAP50-95: 0.843.
        #
        #   bfmc_best.pt         — Base production model. 100 epochs trained
        #                          on Bosch Traffic Signs dataset (561 images,
        #                          9 classes). mAP50: 0.927, mAP50-95: 0.837.
        #                          Source: runs/detect/bfmc_models/sign_detector/
        #
        #   bfmc_last_shirts.pt  — Last checkpoint from the shirt fine-tuning
        #                          run (epoch 10/10). Use for resuming training.
        #
        #   last.pt              — Last checkpoint from the original 100-epoch
        #                          training run. Use for resuming training.
        #
        #   yolov8n.pt (fallback) — Ultralytics COCO pretrained (80 classes).
        #                           Only detects "stop sign" from the 9 BFMC signs.
        # ─────────────────────────────────────────────────────────────────
        custom_paths = [
            "src/perception/sign_recognition/bfmc_best_shirts.pt",  # Best + negative mining
            "src/perception/sign_recognition/bfmc_best.pt",         # Base 100-epoch model
            "models/sign_detector_best.pt",                          # Legacy path
            "models/best.pt",                                        # Legacy path
            "src/perception/sign_recognition/last.pt",               # Last checkpoint
        ]
        for path in custom_paths:
            if os.path.exists(path):
                self.model_path = path
                break

        try:
            self.model = YOLO(self.model_path)
            class_names = list(self.model.names.values())
            self.logging.info(
                f"✅ Sign Detection: loaded {self.model_path} "
                f"({len(class_names)} classes)"
            )
        except Exception as e:
            self.logging.error(f"❌ Failed to load YOLO model: {e}")
            self.model = None

    def thread_work(self):
        if self.model is None:
            time.sleep(1)
            return

        # Rate limiting
        now = time.time()
        if now - self.last_inference_time < self.inference_interval:
            return
        self.last_inference_time = now

        # Get latest camera frame
        frame_msg = self.cameraSubscriber.receive()
        if frame_msg is None:
            return

        # Decode the frame (camera sends base64 encoded JPEG)
        try:
            if isinstance(frame_msg, str):
                image_data = base64.b64decode(frame_msg)
                np_arr = np.frombuffer(image_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif isinstance(frame_msg, np.ndarray):
                frame = frame_msg
            else:
                return
        except Exception as e:
            if self.debugging:
                self.logging.warning(f"Frame decode error: {e}")
            return

        if frame is None:
            return

        # Run YOLO inference
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            if self.debugging:
                self.logging.warning(f"Inference error: {e}")
            return

        # Process detections
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                raw_label = self.model.names[cls_id].lower().strip()
                conf = float(box.conf[0])

                # Map to BFMC sign name
                bfmc_sign = LABEL_MAP.get(raw_label)
                if bfmc_sign is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ── Post-Detection Filters ──
                # Validate shape, color, and size to reject false positives
                # (e.g., red shirts, blue cars, tiny noise detections).
                if FILTERS_AVAILABLE:
                    if not validate_detection(frame, bfmc_sign, (x1, y1, x2, y2)):
                        if self.debugging:
                            self.logging.info(
                                f"  ❌ FILTERED: {bfmc_sign} ({conf:.0%}) "
                                f"at [{x1},{y1},{x2},{y2}]")
                        continue  # Rejected by filters

                detection = {
                    "sign": bfmc_sign,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                }

                # Publish to the message bus
                self.signPublisher.send(json.dumps(detection))

                # DRAW BOX!
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, bfmc_sign, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if self.debugging:
                    self.logging.info(
                        f"🔍 Sign: {bfmc_sign} ({conf:.0%}) "
                        f"at [{x1},{y1},{x2},{y2}]"
                    )

        # ── Stream Annotated Frame to Dashboard ──
        if self.stream_socket:
            try:
                # The laptop sends 'F' to request a frame
                req = self.stream_socket.recv(1)  
                if req == b'F':
                    # Resize to save WiFi bandwidth (Server expects 640x480)
                    frame_resized = cv2.resize(frame, (640, 480))
                    _, img_encoded = cv2.imencode('.jpg', frame_resized)
                    img_bytes = img_encoded.tobytes()
                    size = len(img_bytes)
                    
                    # Send 8-byte size header, then image data
                    self.stream_socket.sendall(size.to_bytes(8, byteorder='big'))
                    self.stream_socket.sendall(img_bytes)
                    
                    # Receive confirmation
                    status = self.stream_socket.recv(1024) 
            except socket.timeout:
                pass # Server didn't request a frame in time, just continue without blocking YOLO
            except Exception as e:
                self.logging.warning(f"Video Stream broken: {e}")
                self.stream_socket.close()
                self.stream_socket = None # Stop trying to stream until restart
