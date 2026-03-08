#!/usr/bin/env python3

import json
from pathlib import Path

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


LABEL_MAP = {
    "green": "green",
    "red": "red",
    "yellow": "yellow",
    "redandyellow": "redandyellow",
    "red_yellow": "redandyellow",
    "red-yellow": "redandyellow",
    "red and yellow": "redandyellow",
    "stop": "stop",
    "stop sign": "stop",
    "parking": "parking",
    "parking_sign": "parking",
    "priority": "priority",
    "crosswalk": "crosswalk",
    "highway_entrance": "highway_entrance",
    "highway_exit": "highway_exit",
    "roundabout": "round_about",
    "round_about": "round_about",
    "one_way": "one_way",
    "no_entry": "no_entry",
}


def _resolve_model_path() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "src/perception/sign_recognition/bfmc_best_traffic_lights.pt",
        repo_root / "src/perception/sign_recognition/bfmc_best_shirts.pt",
        repo_root / "src/perception/sign_recognition/bfmc_best.pt",
        repo_root / "runs/detect/bfmc_models/sign_detector/weights/best.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return "yolov8n.pt"


class SignDetector:
    def __init__(self):
        rospy.init_node("sign_recognition_node")
        self.bridge = CvBridge()
        self.model_path = _resolve_model_path()
        self.model = YOLO(self.model_path)

        self.image_sub = rospy.Subscriber(
            "/camera/image_raw",
            Image,
            self.image_callback,
            queue_size=1,
        )

        self.pub = rospy.Publisher(
            "/detected_sign",
            String,
            queue_size=10,
        )

        rospy.loginfo(f"Sign recognition node started with model: {self.model_path}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logwarn(f"Image conversion failed: {exc}")
            return

        results = self.model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_label = str(self.model.names[cls_id]).lower().strip()
                mapped_label = LABEL_MAP.get(raw_label)
                if mapped_label is None:
                    continue

                payload = {"detected_object": mapped_label}
                self.pub.publish(String(json.dumps(payload)))
                rospy.loginfo(f"Detected: {payload}")


if __name__ == "__main__":
    SignDetector()
    rospy.spin()
