#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32

class SteeringMapper:
    """
    Task [005b]: Translates vision error (cm) into steering angle (degrees).
    Implements a PD Controller to smooth the response.
    """
    def __init__(self):
        rospy.init_node('steering_mapper', anonymous=True)
        
        # PID Constants (Tunable)
        self.Kp = rospy.get_param('~Kp', 0.8)  # Proportional
        self.Kd = rospy.get_param('~Kd', 0.1)  # Derivative
        
        self.prev_error = 0.0
        self.max_steer = 25.0 # Degrees
        
        # 1. Listen to the Lateral Error from Vision (Task 002c)
        self.error_sub = rospy.Subscriber("/raven/control/lateral_error", Float32, self.callback)
        
        # 2. Publish the Calculated Steer Angle for the Serial Link (Task 005a)
        self.steer_pub = rospy.Publisher("/raven/control/steer_angle", Float32, queue_size=1)
        
        rospy.loginfo("RAVEN Control: Steering Mapper Initialized...")

    def callback(self, msg):
        error = msg.data # in cm (ish)
        
        # PD Control Logic
        derivative = error - self.prev_error
        output_value = (self.Kp * error) + (self.Kd * derivative)
        
        self.prev_error = error
        
        # Clamp to physical hardware limits
        output_value = max(min(output_value, self.max_steer), -self.max_steer)
        
        # Publish
        self.steer_pub.publish(output_value)

if __name__ == '__main__':
    try:
        mapper = SteeringMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
