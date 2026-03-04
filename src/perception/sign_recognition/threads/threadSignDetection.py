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


class threadSignDetection(ThreadWithStop):
    """Thread that continuously reads camera frames and runs YOLOv8 sign detection.

    Subscribes to the mainCamera message to get frames.
    Publishes SignDetected messages with the detected sign name and confidence.

    Args:
        queueList: Dictionary of multiprocessing queues.
        logging: Logger instance.
        debugging: Enable debug output.
    """

    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging

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

    def _load_model(self):
        """Load the YOLO model. Tries custom model first, falls back to COCO."""
        if YOLO is None:
            self.logging.warning(
                "⚠️ ultralytics not installed! Sign detection disabled. "
                "Run: pip install ultralytics"
            )
            return

        import os
        # Check for custom model in common locations
        custom_paths = [
            "models/sign_detector_best.pt",
            "models/best.pt",
            "src/perception/sign_recognition/best.pt",
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
∫∫
        """Main work loop — receive camera frame, run inference, publish results."""
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

                detection = {
                    "sign": bfmc_sign,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                }

                # Publish to the message bus
                self.signPublisher.send(json.dumps(detection))

                if self.debugging:
                    self.logging.info(
                        f"🔍 Sign: {bfmc_sign} ({conf:.0%}) "
                        f"at [{x1},{y1},{x2},{y2}]"
                    )
