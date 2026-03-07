# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import json
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import SignDetected, SpeedMotor, SteerMotor
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender

class threadPlanner(ThreadWithStop):
    """
    The Hybrid Planner Node for Skynet.
    This thread acts as the 'brain' that bridges Perception (YOLO/Camera) with Action (Arduino Motors).
    It subscribes to high-level semantic queues (like SignDetected), applies driving rules,
    and publishes low-level actuator commands to the serial handler.

    Args:
        queuesList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queuesList, logger, debugger=False):
        super(threadPlanner, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger

        # 1. Perception Subscribers (Listening to YOLO and other sensors)
        self.signSubscriber = messageHandlerSubscriber(self.queuesList, SignDetected, "lastOnly", True)

        # 2. Actuator Senders (Commanding the Arduino)
        self.speedSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)

        # State tracking
        self.current_speed = 0.0
        self.current_steer = 0.0
        
        # Stop-sign logic: if we stop, we want to pause before continuing
        self.stop_timer = 0
        self.is_stopped = False

        print("\033[1;97m[ Planner ] :\033[0m \033[1;92mINFO\033[0m - Hybrid Planner Node Initialized")

    def thread_work(self):
        """
        The main control loop. Executes driving rules based on perception inputs.
        """
        # ==========================================
        # 1. READ PERCEPTION INPUTS
        # ==========================================
        sign_msg_raw = self.signSubscriber.receive()

        # ==========================================
        # 2. APPLY DRIVING RULES (The Hybrid Map/Logic)
        # ==========================================
        
        # Handle Traffic Signs
        if sign_msg_raw is not None:
            try:
                # The message is expected to be a serialized JSON string: {"sign": "stop", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}
                sign_data = json.loads(sign_msg_raw)
                sign_name = sign_data.get("sign")
                
                if self.debugger:
                    self.logger.info(f"Planner received sign: {sign_name}")

                if sign_name == "stop" and not self.is_stopped:
                    print("\033[1;97m[ Planner ] :\033[0m \033[1;91mACTION\033[0m - STOP sign detected. Stopping car.")
                    self.current_speed = 0.0
                    self.is_stopped = True
                    self.stop_timer = 30  # Stop for ~3 seconds (assuming 10hz loop)
                    
                elif sign_name == "highway_entrance":
                    print("\033[1;97m[ Planner ] :\033[0m \033[1;92mACTION\033[0m - Highway Entrance. Increasing speed.")
                    self.current_speed = 30.0  # Speed up
                    
                elif sign_name == "highway_exit":
                    print("\033[1;97m[ Planner ] :\033[0m \033[1;93mACTION\033[0m - Highway Exit. Decreasing speed.")
                    self.current_speed = 15.0  # Slow down
                    
                elif sign_name == "parking":
                    print("\033[1;97m[ Planner ] :\033[0m \033[1;94mACTION\033[0m - Parking sign detected. Commencing park sequence.")
                    # Implement parking maneuver here
                    self.current_speed = 0.0
                    
            except json.JSONDecodeError:
                print(f"\033[1;97m[ Planner ] :\033[0m \033[1;93mWARNING\033[0m - Failed to parse SignDetected message: {sign_msg_raw}")

        # Handle simple stop timer logic
        if self.is_stopped:
            self.stop_timer -= 1
            if self.stop_timer <= 0:
                print("\033[1;97m[ Planner ] :\033[0m \033[1;92mACTION\033[0m - Stop cleared. Resuming driving.")
                self.is_stopped = False
                self.current_speed = 15.0  # Default cruise speed

        # ==========================================
        # 3. PUBLISH LOW-LEVEL COMMANDS
        # ==========================================
        # We only send commands if they theoretically changed to avoid destroying the serial bandwidth
        # In a real E2E system, we might stream these at 30hz continuously.
        self.speedSender.send(self.current_speed)
        # self.steerSender.send(self.current_steer)
