#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class LaneSegmenter:
    def __init__(self):
        rospy.init_node('lane_segmenter', anonymous=True)
        self.bridge = CvBridge()
        
        # Subscribes to the Gazebo Camera
        # We listen to /camera/rgb/image_raw directly to get the feed
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        
        # Publishes the BINARY MASK for Team A's next tasks (Lateral Offset)
        self.mask_pub = rospy.Publisher("/raven/perception/lane_mask", Image, queue_size=1)
        
        rospy.loginfo("RAVEN Eye: Lane Segmenter Initialized...")

    def get_roi(self, frame):
        """Masks everything except the road directly in front of the car."""
        height, width = frame.shape[:2]
        # Define a trapezoid covering the lower half of the view
        # We focus on the bottom part where the track is
        polygon = np.array([[
            (0, height), 
            (width, height), 
            (int(width * 0.8), int(height * 0.6)), 
            (int(width * 0.2), int(height * 0.6))
        ]], np.int32)
        
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, polygon, (255, 255, 255))
        return cv2.bitwise_and(frame, mask)

    def callback(self, data):
        try:
            # 1. Convert ROS to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        # 2. Pre-process: Grayscale and Blur to remove 'noise'
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Apply ROI (Focus only on the road)
        # Note: Since we greyscaled, we need to handle channels correctly if get_roi expects color
        # Our get_roi uses the shape of 'frame'. If we pass gray (1 channel), mask will be 1 channel.
        roi_image = self.get_roi(blurred)
        
        # 4. Color Thresholding (Isolate White Lanes)
        # BFMC lanes are usually high-intensity white
        _, binary_mask = cv2.threshold(roi_image, 200, 255, cv2.THRESH_BINARY)
        
        # 5. Publish the result for Task 002c (Lateral Offset)
        try:
            self.mask_pub.publish(self.bridge.cv2_to_imgmsg(binary_mask, "mono8"))
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Publish Error: {e}")

        # Optional: Display for debug (if running on desktop)
        # cv2.imshow("RAVEN Eye - Lane Mask", binary_mask)
        # cv2.waitKey(1)

if __name__ == '__main__':
    try:
        segmenter = LaneSegmenter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
