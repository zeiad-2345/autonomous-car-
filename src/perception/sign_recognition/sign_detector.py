#!/usr/bin/env python3

import rospy
import cv2
import json
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# YOLOv8
from ultralytics import YOLO


class SignDetector:
    def __init__(self):
        rospy.init_node("sign_recognition_node")

        self.bridge = CvBridge()

        # Load YOLOv8 pretrained model
        # This model already knows STOP signs
        self.model = YOLO("yolov8n.pt")

        # Subscribe to camera
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw",
            Image,
            self.image_callback,
            queue_size=1
        )

        # Publish detected sign label
        self.pub = rospy.Publisher(
            "/detected_sign",
            String,
            queue_size=10
        )

        rospy.loginfo("✅ Sign Recognition Node Started")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"Image conversion failed: {e}")
            return

        results = self.model(frame, verbose=False)

        detected = None

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                # We only care about these
                if label in ["stop sign", "parking meter"]:
                    detected = label
                    break

        if detected:
            msg = {
                "detected_object": detected.replace(" ", "_")
            }
            self.pub.publish(String(json.dumps(msg)))
            rospy.loginfo(f"🛑 Detected: {msg}")


if __name__ == "__main__":
    SignDetector()
    rospy.spin()
