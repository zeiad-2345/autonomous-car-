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

import cv2
import threading
import base64
try:
    import picamera2
except ImportError:
    picamera2 = None
import time
import os
import numpy as np

# ROS Imports (Conditional)
try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from src.utils.messages.allMessages import (
    mainCamera,
    serialCamera,
    Recording,
    Record,
    Brightness,
    Contrast,
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import StateChange
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.statemachine.systemMode import SystemMode

class threadCamera(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadCamera, self).__init__(pause=0.001)
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_rate = 5
        self.recording = False

        # Simulation Mode Flag
        self.is_simulation = os.getenv("RAVEN_SIMULATION", "false").lower() == "true"
        
        self.video_writer = ""

        self.recordingSender = messageHandlerSender(self.queuesList, Recording)
        self.mainCameraSender = messageHandlerSender(self.queuesList, mainCamera)
        self.serialCameraSender = messageHandlerSender(self.queuesList, serialCamera)

        self.subscribe()
        self._init_camera()
        self.queue_sending()
        self.configs()

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""

        self.recordSubscriber = messageHandlerSubscriber(self.queuesList, Record, "lastOnly", True)
        self.brightnessSubscriber = messageHandlerSubscriber(self.queuesList, Brightness, "lastOnly", True)
        self.contrastSubscriber = messageHandlerSubscriber(self.queuesList, Contrast, "lastOnly", True)
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, StateChange, "lastOnly", True)

    def queue_sending(self):
        """Callback function for recording flag."""
        if self._blocker.is_set():
            return
        self.recordingSender.send(self.recording)
        threading.Timer(1, self.queue_sending).start()

    # ================================ RUN ================================================
    def thread_work(self):
        """This function will run while the running flag is True. 
        It captures the image from camera and make the required modifies 
        and then it send the data to process gateway."""
        
        # In simulation, wait for image from ROS callback
        if self.is_simulation:
            if not self.latest_ros_image:
                 time.sleep(0.1)
                 return
            
            # Use the latest image from ROS
            mainRequest = self.latest_ros_image
            # Just resize or crop for "lores" if needed, or use same.
            # Real cam does: main=(2048, 1080), lores=(512, 270)
            serialRequest = cv2.resize(mainRequest, (512, 270))
        
        # In real mode, if camera is missing, skip
        elif self.camera is None:
            time.sleep(0.1)
            return
        else:
             try:
                # Real Camera Capture
                mainRequest = self.camera.capture_array("main")
                serialRequest = self.camera.capture_array("lores")
             except Exception as e:
                print(f"\033[1;97m[ Camera ] :\033[0m \033[1;91mERROR\033[0m - Capture failed: {e}")
                return
            
        try:
            recordRecv = self.recordSubscriber.receive()
            if recordRecv is not None: 
                self.recording = bool(recordRecv)
                if recordRecv == False:
                    if self.video_writer:
                        self.video_writer.release() # type: ignore
                else:
                    fourcc = cv2.VideoWriter_fourcc( # type: ignore
                        *"XVID"
                    )  # You can choose different codecs, e.g., 'MJPG', 'XVID', 'H264', etc.
                    self.video_writer = cv2.VideoWriter(
                        "output_video" + str(time.time()) + ".avi",
                        fourcc,
                        self.frame_rate,
                        (2048, 1080),
                    )

        except Exception as e:
            print(f"\033[1;97m[ Camera ] :\033[0m \033[1;91mERROR\033[0m - {e}")

        try:
            if self.recording == True and self.video_writer:
                self.video_writer.write(mainRequest) # type: ignore

            # Convert for serial (Sim images are already BGR usually, but let's check source)
            # Picamera 'lores' is YUV420, so it needs conversion. 
            # ROS images are converted to BGR in callback. 
            if not self.is_simulation:
                serialRequest = cv2.cvtColor(serialRequest, cv2.COLOR_YUV2BGR_I420) # type: ignore

            _, mainEncodedImg = cv2.imencode(".jpg", mainRequest) # type: ignore
            _, serialEncodedImg = cv2.imencode(".jpg", serialRequest) # type: ignore

            mainEncodedImageData = base64.b64encode(mainEncodedImg).decode("utf-8") # type: ignore
            serialEncodedImageData = base64.b64encode(serialEncodedImg).decode("utf-8") # type: ignore

            if self._blocker.is_set():
                return

            self.mainCameraSender.send(mainEncodedImageData)
            self.serialCameraSender.send(serialEncodedImageData)
        except Exception as e:
            print(f"\033[1;97m[ Camera ] :\033[0m \033[1;91mERROR\033[0m - Processing failed: {e}")

    # ================================ STATE CHANGE HANDLER ========================================
    def state_change_handler(self):
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            modeDict = SystemMode[message].value["camera"]["thread"]

            if "resolution" in modeDict:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - Resolution changed to {modeDict['resolution']}")

    # ================================ INIT CAMERA ========================================
    def _init_camera(self):
        """This function will initialize the camera object. It will make this camera object have two chanels "lore" and "main"."""
        
        self.camera = None
        self.latest_ros_image = None
        self.bridge = None

        if self.is_simulation:
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;94mINFO\033[0m - Simulation Mode Detected. Initializing ROS Subscriber...")
            if not ROS_AVAILABLE:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - ROS not found. Cannot run simulation mode.")
                return

            try:
                # We assume roscore is running and node is initialized in main or here. 
                # Since this is a thread inside a multiprocess, it's safer if the process initialized the node or we do it anonymously here if not done.
                # However, usually one node per process. processCamera.py doesn't seem to init node.
                # Let's try checking if node is initialized, if not init it.
                if rospy.get_name() == "/unnamed":
                    rospy.init_node('raven_brain_camera', anonymous=True)
                
                self.bridge = CvBridge()
                # Topic from raven-sim/README or standard
                self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.ros_callback)
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - Subscribed to /camera/rgb/image_raw")
            except Exception as e:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - Failed to init ROS: {e}")
            return

        # REAL HARDWARE MODE
        try:
            if picamera2 is None:
                 print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - picamera2 lib not found.")
                 return

            # check if camera is available
            if len(picamera2.Picamera2.global_camera_info()) == 0:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - No camera detected. Camera functionality will be disabled.")
                self.camera = None
                return
            
            self.camera = picamera2.Picamera2()
            config = self.camera.create_preview_configuration(
                buffer_count=1,
                queue=False,
                main={"format": "RGB888", "size": (2048, 1080)},
                lores={"size": (512, 270)},
                encode="lores",
            )
            self.camera.configure(config) # type: ignore
            self.camera.start()
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - Camera initialized successfully")
        except Exception as e:
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - Failed to initialize camera: {e}")
            self.camera = None

    def ros_callback(self, data):
        """Callback for ROS Image messages"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.latest_ros_image = cv_image
        except Exception as e:
            # Avoid spamming logs in high freq
            pass

    # =============================== STOP ================================================
    def stop(self):
        if self.recording and self.video_writer:
            self.video_writer.release() # type: ignore
        if self.camera is not None:
            self.camera.stop()
        super(threadCamera, self).stop()

    # =============================== CONFIG ==============================================
    def configs(self):
        """Callback function for receiving configs on the pipe."""
        if self._blocker.is_set():
            return
        if self.brightnessSubscriber.is_data_in_pipe():
            message = self.brightnessSubscriber.receive()
            if self.debugger:
                self.logger.info(str(message))
            if self.camera:
                self.camera.set_controls(
                    {
                        "AeEnable": False,
                        "AwbEnable": False,
                        "Brightness": max(0.0, min(1.0, float(message))), # type: ignore
                    }
                )
        if self.contrastSubscriber.is_data_in_pipe():
            message = self.contrastSubscriber.receive() # de modificat marti uc camera noua 
            if self.debugger:
                self.logger.info(str(message))
            if self.camera:
                self.camera.set_controls(
                    {
                        "AeEnable": False,
                        "AwbEnable": False,
                        "Contrast": max(0.0, min(32.0, float(message))), # type: ignore
                    }
                )
        threading.Timer(1, self.configs).start()