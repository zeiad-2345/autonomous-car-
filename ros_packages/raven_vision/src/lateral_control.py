#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError

class LateralController:
    def __init__(self):
        rospy.init_node('lateral_controller', anonymous=True)
        self.bridge = CvBridge()
        
        # Subscribe to the BINARY MASK from Task 002b
        self.mask_sub = rospy.Subscriber("/raven/perception/lane_mask", Image, self.callback)
        
        # Publish the ERROR (distance from center) for the PID Controller
        self.error_pub = rospy.Publisher("/raven/control/lateral_error", Float32, queue_size=1)
        
        rospy.loginfo("RAVEN Brain: Lateral Controller Initialized...")

    def calculate_error(self, binary_mask):
        """
        Calculates the distance between the image center and the lane center.
        """
        height, width = binary_mask.shape
        
        # We only care about the bottom part of the mask (closest to car)
        # Taking a slice of the bottom 20% of the image
        # This reduces noise from upcoming turns and focuses on immediate positioning
        scan_slice = binary_mask[int(height*0.8):, :]
        
        # Find all white pixels (lane markings)
        M = cv2.moments(scan_slice)
        
        if M["m00"] > 0:
            # Calculate centroid of the white pixels
            cX = int(M["m10"] / M["m00"])
            
            # Image Center X
            image_center = width // 2
            
            # Error = Distance from Center
            # Positive Error = Car is to the LEFT of lane center (needs to steer RIGHT)
            # Negative Error = Car is to the RIGHT of lane center (needs to steer LEFT)
            # We assume the camera is mounted locally central.
            error_pixels = image_center - cX
            
            # Simple conversion to 'centimeters' (This is an ESTIMATE for the PID)
            # You will need to tune this SCALING_FACTOR based on real-world tests/calibration
            # For now, we normalize it to a range roughly -1.0 to 1.0 or similar useful value
            # Let's say image width is 640. Max error is +/- 320.
            # 320 pixels ~ 15cm offset? (Just a guess)
            SCALE_FACTOR = 0.05 
            error_cm = error_pixels * SCALE_FACTOR
            
            return float(error_cm)
            
        return 0.0

    def callback(self, data):
        try:
            # Convert ROS Image to OpenCV (Mono8)
            mask = self.bridge.imgmsg_to_cv2(data, "mono8")
            
            # Calculate Error
            error = self.calculate_error(mask)
            
            # Publish Error to PID
            self.error_pub.publish(error)
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

if __name__ == '__main__':
    try:
        controller = LateralController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
