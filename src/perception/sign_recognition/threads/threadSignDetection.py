import time
import json
import base64
from pathlib import Path

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

TRAFFIC_LIGHT_LABELS = {"green", "red", "yellow", "redandyellow"}


LABEL_MAP = {
    # Traffic lights
    "green": "green",
    "red": "red",
    "yellow": "yellow",
    "redandyellow": "redandyellow",
    "red_yellow": "redandyellow",
    "red-yellow": "redandyellow",
    "red and yellow": "redandyellow",
    # Traffic signs
    "stop sign": "stop",
    "stop": "stop",
    "parking": "parking",
    "parking_sign": "parking",
    "p": "parking",
    "priority": "priority",
    "priority_road": "priority",
    "priority road": "priority",
    "give_way": "priority",
    "crosswalk": "crosswalk",
    "pedestrian": "crosswalk",
    "pedestrian_crossing": "crosswalk",
    "pedestriancrossing": "crosswalk",
    "highway": "highway_entrance",
    "highway_entrance": "highway_entrance",
    "motorway": "highway_entrance",
    "motorway_begin": "highway_entrance",
    "highway_exit": "highway_exit",
    "motorway_end": "highway_exit",
    "end_motorway": "highway_exit",
    "end motorway": "highway_exit",
    "roundabout": "round_about",
    "roundabout_sign": "round_about",
    "round_about": "round_about",
    "one_way": "one_way",
    "one-way": "one_way",
    "oneway": "one_way",
    "one way": "one_way",
    "no_entry": "no_entry",
    "no-entry": "no_entry",
    "no entry": "no_entry",
    "noentry": "no_entry",
    "do_not_enter": "no_entry",
    "no_enter": "no_entry",
}


class threadSignDetection(ThreadWithStop):
    """Thread that runs YOLOv8 sign and traffic-light detection on camera frames."""

    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging

        self.model = None
        self.model_path = "yolov8n.pt"
        self.conf_threshold = 0.5

        self.last_inference_time = 0
        self.inference_interval = 0.2

        self.cameraSubscriber = messageHandlerSubscriber(
            self.queuesList, mainCamera, "lastOnly", True
        )
        self.signPublisher = messageHandlerSender(self.queuesList, SignDetected)

        self._load_model()
        super(threadSignDetection, self).__init__()

    def _resolve_repo_root(self):
        return Path(__file__).resolve().parents[4]

    def _candidate_model_paths(self):
        root = self._resolve_repo_root()
        return [
            root / "src/perception/sign_recognition/bfmc_best_traffic_lights.pt",
            root / "src/perception/sign_recognition/bfmc_best_shirts.pt",
            root / "src/perception/sign_recognition/bfmc_best.pt",
            root / "src/perception/sign_recognition/best.pt",
            root / "runs/detect/bfmc_models/sign_detector_shirts/weights/best.pt",
            root / "runs/detect/bfmc_models/sign_detector/weights/best.pt",
            root / "src/perception/sign_recognition/last.pt",
            Path("yolov8n.pt"),
        ]

    def _load_model(self):
        if YOLO is None:
            self.logging.warning(
                "ultralytics not installed; sign detection disabled. Run: pip install ultralytics"
            )
            return

        for path in self._candidate_model_paths():
            if isinstance(path, Path) and path.exists():
                self.model_path = str(path)
                break
            if isinstance(path, str):
                self.model_path = path
                break

        try:
            self.model = YOLO(self.model_path)
            class_names = list(self.model.names.values())
            self.logging.info(
                f"Sign Detection loaded model: {self.model_path} ({len(class_names)} classes)"
            )
        except Exception as exc:
            self.logging.error(f"Failed to load YOLO model: {exc}")
            self.model = None

    def thread_work(self):
        if self.model is None:
            time.sleep(1)
            return

        now = time.time()
        if now - self.last_inference_time < self.inference_interval:
            return
        self.last_inference_time = now

        frame_msg = self.cameraSubscriber.receive()
        if frame_msg is None:
            return

        try:
            if isinstance(frame_msg, str):
                image_data = base64.b64decode(frame_msg)
                np_arr = np.frombuffer(image_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif isinstance(frame_msg, np.ndarray):
                frame = frame_msg
            else:
                return
        except Exception as exc:
            if self.debugging:
                self.logging.warning(f"Frame decode error: {exc}")
            return

        if frame is None:
            return

        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
        except Exception as exc:
            if self.debugging:
                self.logging.warning(f"Inference error: {exc}")
            return

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_label = str(self.model.names[cls_id]).lower().strip()
                conf = float(box.conf[0])

                mapped_label = LABEL_MAP.get(raw_label)
                if mapped_label is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if FILTERS_AVAILABLE and mapped_label not in TRAFFIC_LIGHT_LABELS:
                    if not validate_detection(frame, mapped_label, (x1, y1, x2, y2)):
                        if self.debugging:
                            self.logging.info(
                                f"Filtered {mapped_label} ({conf:.0%}) at [{x1},{y1},{x2},{y2}]"
                            )
                        continue

                detection = {
                    "sign": mapped_label,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                }
                self.signPublisher.send(json.dumps(detection))

                if self.debugging:
                    self.logging.info(
                        f"Detected {mapped_label} ({conf:.0%}) at [{x1},{y1},{x2},{y2}]"
                    )
