#!/usr/bin/env python3
import rospy
import time
from std_msgs.msg import String, Float32, Bool

class RavenFSM:
    """
    Task [006a]: FSM Framework (The Brain)
    Task [006b]: Sign Response Logic (e.g. Stop Sign)
    """
    
    # States
    STATE_IDLE = "IDLE"
    STATE_DRIVE = "DRIVE"
    STATE_STOP_WAIT = "STOP_WAIT"
    STATE_PARKING = "PARKING"

    def __init__(self):
        rospy.init_node('raven_fsm', anonymous=True)
        
        self.current_state = self.STATE_IDLE
        self.resume_time = 0
        self.default_speed = 15.0
        
        # Inputs
        # 1. Sign Detections (from Team A - Task 008b)
        self.sign_sub = rospy.Subscriber("/raven/perception/sign", String, self.sign_callback)
        
        # 2. User Commands (Start/Stop) - typically from Dashboard
        self.cmd_sub = rospy.Subscriber("/raven/dashboard/command", String, self.command_callback)
        
        # Outputs
        # 1. Target Speed for the Serial Link
        self.speed_pub = rospy.Publisher("/raven/control/target_speed", Float32, queue_size=1)
        
        # 2. State Telemetry
        self.state_pub = rospy.Publisher("/raven/fsm/state", String, queue_size=1)
        
        # Main Loop (10Hz)
        self.rate = rospy.Rate(10)
        
        rospy.loginfo("RAVEN Brain: FSM Initialized [IDLE]")

    def sign_callback(self, msg):
        sign = msg.data.lower()
        if sign == "stop_sign":
            self.handle_stop_sign()
        elif sign == "parking_sign":
            self.current_state = self.STATE_PARKING

    def command_callback(self, msg):
        cmd = msg.data.lower()
        if cmd == "start":
            self.current_state = self.STATE_DRIVE
        elif cmd == "stop":
            self.current_state = self.STATE_IDLE

    def handle_stop_sign(self):
        if self.current_state == self.STATE_DRIVE:
            rospy.loginfo("FSM: STOP SIGN DETECTED -> Waiting 3s")
            self.current_state = self.STATE_STOP_WAIT
            self.resume_time = time.time() + 3.0

    def run(self):
        while not rospy.is_shutdown():
            target_speed = 0.0
            
            # --- FSM LOGIC ---
            if self.current_state == self.STATE_IDLE:
                target_speed = 0.0
                
            elif self.current_state == self.STATE_DRIVE:
                target_speed = self.default_speed
                
            elif self.current_state == self.STATE_STOP_WAIT:
                target_speed = 0.0
                # Check timer
                if time.time() > self.resume_time:
                    rospy.loginfo("FSM: Resuming Drive")
                    self.current_state = self.STATE_DRIVE
                    
            elif self.current_state == self.STATE_PARKING:
                # Placeholder for parking maneuver
                target_speed = 5.0 
            
            # Publish
            self.speed_pub.publish(target_speed)
            self.state_pub.publish(self.current_state)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        fsm = RavenFSM()
        fsm.run()
    except rospy.ROSInterruptException:
        pass
