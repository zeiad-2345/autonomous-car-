#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class VideoStreamHandler:
    def __init__(self):
        # 1. Initialize the ROS1 Node
        rospy.init_node('video_stream_handler', anonymous=True)
        
        # 2. The "Handshake": bridge ROS images to OpenCV (NumPy)
        self.bridge = CvBridge()

        # 3. Subscriber: Listen to the Gazebo Camera
        # NOTE: In BFMC Gazebo, the topic is usually /camera/rgb/image_raw
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        
        rospy.loginfo("RAVEN Eye: Video Stream Handler Initialized...")

    def callback(self, data):
        try:
            # 4. Convert ROS Image message to OpenCV format (bgr8)
            # This turns 'data' into a standard NumPy array!
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # --- TEAM A STARTS HERE ---
        # At this point, 'cv_image' is a NumPy array. 
        # You can now pass it to Task 002a (IPM) or 002b (Segmentation).
        
        # Vibe Check: Display the feed (For testing on Linux/Sim machines)
        cv2.imshow("RAVEN Eye - Gazebo Feed", cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        handler = VideoStreamHandler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
