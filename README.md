# BFMC - Brain Project

The project contains all the provided code for the RPi, more precisely:
- Firmware for communicating with the Nucleo and control the robot movements (Speed with constant current consumption, speed with constant speed, braking, moving and steering);
- Firmware for gathering data from the sensors (IMU and Camera);
- API's for communicating with the environmental servers at Bosch location;
- Simulated servers for the API's.

## 🏗️ RAVEN/Skynet Architecture (ROS 1 vs Native Python)

The `raven-brain-stack` utilizes a hybrid architectural approach. The core driving logic operates on a lightweight, high-speed custom **Python `multiprocessing.Queue`** framework, while specific isolated perception nodes leverage **ROS 1 Noetic**.

```mermaid
graph TD
    subgraph ROS 1 Noetic Environment
        G[Gazebo / Real Camera] -- /camera/rgb/image_raw --> LS[lane_segmentation.py]
        LS -- /raven/perception/lane_mask --> TF[Task: Lateral Offset]
        
        note1[Used purely for specific isolated<br>perception tasks requiring ROS tools]
    end

    subgraph Native Python Multiprocessing Environment
        C[threadCamera.py] -- Camera Queue --> SD[threadSignDetection.py]
        SD -- SignDetected Queue --> P[threadPlanner.py]
        P -- SpeedMotor/SteerMotor Queue --> SW[threadWrite.py]
        SR[threadRead.py] -- ImuData Queue --> P
        
        note2[The core Skynet brain.<br>High speed, low overhead, purely native Python (ZeroMQ-style).]
    end

    SW -- Serial USB --> MCU[Arduino RP2040 / Nucleo]
    MCU -- Serial USB --> SR

    classDef ros fill:#22314E,stroke:#4E70A6,color:#fff
    classDef zmq fill:#1A4024,stroke:#3D9955,color:#fff
    classDef hw fill:#5A3612,stroke:#B46D24,color:#fff
    
    class G,LS,TF,note1 ros
    class C,SD,P,SW,SR,note2 zmq
    class MCU hw
```

## 🕹️ Remote Control Support (New)

The Brain now includes a dedicated process (`processDashboard`) that listens for SocketIO commands from `raven-computer` and forwards them to the embedded controller.

- **Port**: 5005 (SocketIO)
- **Commands**: `SpeedMotor`, `SteerMotor`, `Klem` (System State)
- **Feedback**: Publishes Battery and IMU data back to the Dashboard.

### macOS Compatibility
The stack has been patched to support macOS environments:
- **Multiprocessing**: Fixed `spawn` start method issues.
- **Serial**: Added fallback detection for Arduino devices.
- **Networking**: `ip_manager` adapted for macOS `ifconfig`.

## 🔌 Serial Communication (Arduino RP2040)
The Brain communicates with the low-level Arduino embedded controller using a custom asynchronous serial protocol via USB.

- **Script:** `src/serial_controller.py`
- **Protocol:** `#key:value;;` (Commands) / `@key:value;;\r\n` (Telemetry)
- **Features:** 
    - Background Daemon thread continuously parses `@imu` and `@encoder` data.
    - Non-blocking main thread accepts user keyboard input to send `#speed` and `#steer` commands instantly.

### Running the Controller
```bash
python3 src/serial_controller.py
```
*(Note: Change `SERIAL_PORT` inside the script if running directly from a Mac instead of the Pi).*

---

## 🧠 Sign Detection AI (Task 008b)

YOLOv8-based real-time traffic sign detection trained specifically for the 9 BFMC miniature signs. Runs on both the Raspberry Pi 4 and Mac (MPS accelerated).

### Detected Signs
Stop · Parking · Priority · Crosswalk · Highway Entrance · Highway Exit · Roundabout · One-way · No-entry

### Model Files (`src/perception/sign_recognition/`)

| File | Description | Accuracy |
|------|-------------|----------|
| `bfmc_best_shirts.pt` | ⭐ **RECOMMENDED.** Fine-tuned with negative mining (red shirt rejection). | mAP50: 93.3% |
| `bfmc_best.pt` | Base model. 100 epochs on Bosch dataset (561 images). | mAP50: 92.7% |
| `bfmc_last_shirts.pt` | Last checkpoint from shirt fine-tuning (for resuming). | — |
| `last.pt` | Last checkpoint from base training (for resuming). | — |

### Training History

| Run | Script | Epochs | Dataset | Output |
|-----|--------|--------|---------|--------|
| **Run 1** (Base) | `train_signs.py` | 100 | 561 images, 9 classes | `sign_detector/` |
| **Run 3** (Shirts) | `finetune_shirts.py` | +10 fine-tune | 600 images (561 signs + 39 shirt backgrounds) | `sign_detector_shirts/` |

Run 3 uses **negative mining**: red shirt images with empty label files teach the model "no sign here," reducing false positives on red clothing.

### Quick Usage

```bash
# Live detection with webcam (Mac)
python3 src/perception/sign_recognition/live_sign_detector.py --model src/perception/sign_recognition/bfmc_best_shirts.pt --webcam

# Live detection with Pi Camera (Raspberry Pi)
python3 src/perception/sign_recognition/live_sign_detector.py --model src/perception/sign_recognition/bfmc_best_shirts.pt
```

### Post-Detection Filters (Salma's Suggestion)
After YOLO detects a sign, a filter pipeline validates it using:
- **Shape Filter:** Checks bounding box aspect ratio (signs ≈ square, cars ≈ wide+flat)
- **Color Filter:** Verifies dominant HSV hue matches expected sign color (lighting-independent)
- **Size Filter:** Rejects tiny noise and frame-filling false positives

Filters are enabled by default. Disable with `--no-filters` for debugging.

> Full details: [`src/perception/sign_recognition/README.md`](src/perception/sign_recognition/README.md)

---

## The documentation is available in more details here:
[Documentation](https://bosch-future-mobility-challenge-documentation.readthedocs-hosted.com/)

---

## Feature: Video Stream Handler (001b)

### Overview
The `001b-video-stream-handler` branch implements the "Eyes" of the RAVEN platform for the ROS1 Noetic stack. It bridges the simulation world (Gazebo) with the computer vision stack (OpenCV).
- **Subscribes to**: `/camera/rgb/image_raw` (Gazebo Camera)
- **converts**: ROS `sensor_msgs/Image` -> OpenCV `numpy.ndarray` (BGR8)
- **preview**: Displays a live feed in a "RAVEN Eye" window (Desktop only).

### Run the Node
```bash
#  Make sure you are in the ROS environment (ros_packages/raven_vision)
python3 ros_packages/raven_vision/src/video_stream_handler.py
```
*Note: A running `roscore` and Gazebo simulation are required.*

---

## Feature: IPM Matrix Calculation (002a)

### Overview
The `002a-ipm-matrix-calc` branch introduces tools to calculate the Inverse Perspective Mapping (IPM) matrix. This transforms the camera's perspective view into a top-down "Bird's-Eye View," essential for:
- Accurate lane width measurement
- Obstacle distance estimation
- Ground plane projection

### Local Development (Mac/PC)
To test the calibration tools without a Raspberry Pi:
1.  Initialize the environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements-mac.txt
    ```
2.  The `collect_images.py` script will automatically fall back to your webcam if the Pi Camera is not found.

### Tools Added
Located in `calibration/`:

1.  **`collect_images.py`**
    - **Purpose**: Captures checkerboard images from the Raspberry Pi Camera.
    - **Usage**:
        ```bash
        python3 calibration/collect_images.py
        ```
    - **Controls**: Press 's' to save a frame, 'q' to quit.

2.  **`calibrate_ipm.py`**
    - **Purpose**: Calculates Intrinsic (Camera) and Extrinsic (Homography) matrices.
    - **Process**:
        1.  Detects checkerboards in collected images for intrinsic calibration.
        2.  Opens a "perspective image" (lane view).
        3.  User manually clicks 4 points on the ground plane (Clockwise: TL, TR, BR, BL).
        4.  Generates `calibration/data/calib_data.json`.
    - **Usage**:
        ```bash
        python3 calibration/calibrate_ipm.py --perspective_img <path_to_lane_image>
        ```

3.  **`verify_ipm.py`**
    - **Purpose**: Applies the calculated IPM matrix to a test image to verify correctness.
    - **Usage**:
        ```bash
        python3 calibration/verify_ipm.py <path_to_test_image>
        ```

- `camera_matrix`: Intrinsic parameters
- `dist_coeffs`: Distortion coefficients
- `homography_matrix`: The IPM transformation matrix

---

## 🚀 Skynet — Step-by-Step Deployment Guide

### What Was Built

Three fully-working standalone scripts have been integrated into one coherent system.

| Script | Role | Status |
| --- | --- | --- |
| `src/serial_controller.py` | `SerialController` class + standalone keyboard control | ✅ Refactored |
| `src/skynet.py` | **Master integration** — 3 threads, all wired up | ✅ NEW |
| `services/rpi-wifi-fallback/frame_receiver_server.py` | Laptop video window | ✅ Updated to push-mode |
| `src/perception/sign_recognition/live_sign_detector.py` | Standalone YOLO demo | ✅ Unchanged |
| `src/perception/sign_recognition/sign_filters.py` | Detection filters | ✅ Unchanged |

### Architecture at a Glance

```
Pi                                          Mac (SSH)
─────────────────────────────────           ───────────────────────
src/skynet.py                               frame_receiver_server.py
  ├── PerceptionThread                      (port 5012)
  │     Camera → YOLO + filters              ↑ annotated frames (TCP)
  │     → publishes sign label
  │                                         SSH Terminal:
  ├── PlannerThread                          @imu, @encoder printed live
  │     sign → SIGN_RULES table
  │     → publishes speed/steer
  │
  └── SerialThread ──USB──► Arduino RP2040
        #speed / #steer →
        ← @imu / @encoder
```

### Step 1: Hardware Setup
1. Connect the **Arduino RP2040** to the Pi via USB (`/dev/ttyACM0`).
2. Connect the **Pi Camera** ribbon cable to the CSI port.
3. Make sure Pi and Mac are on the **same network** (lab WiFi or Pi hotspot).

### Step 2: Install Dependencies (one time, on Pi)

```bash
pip install ultralytics opencv-python pyserial picamera2
```

On Mac (for the video viewer):
```bash
pip install opencv-python
```

### Step 3: Run on the Mac first (opens the video window)

```bash
python3 services/rpi-wifi-fallback/frame_receiver_server.py --display
```

The window will block until the Pi connects.

### Step 4: Run skynet.py on the Pi (via SSH)

```bash
ssh captive@<PI_IP>
cd ~/raven-brain-stack
python3 src/skynet.py --laptop-ip <YOUR_MAC_IP>
```

**Common flags:**
```bash
python3 src/skynet.py --no-stream      # Skip video streaming
python3 src/skynet.py --no-arduino     # Bench-test without Arduino
python3 src/skynet.py --conf 0.35      # Lower YOLO confidence
python3 src/skynet.py --no-filters     # Disable post-detection filters
```

### Step 5: What you will see

**Mac OpenCV window:** live camera feed, colored bounding boxes, labels like `STOP 94%`.

**SSH terminal:**
```
[Perception] ✅ Pi Camera initialized
[Perception] ✅ YOLO model loaded: src/perception/sign_recognition/bfmc_best_shirts.pt
[Serial]     ✅ Connected to Arduino on /dev/ttyACM0
[Planner]    🚦 STOP                → speed=  0  steer=  0
[IMU] roll=2.1 pitch=-0.4 yaw=89.2
[Planner]    ⏱  Stop duration elapsed → resuming cruise
```

### Step 6: Standalone scripts still work

```bash
# Mac webcam YOLO test
python3 src/perception/sign_recognition/live_sign_detector.py --webcam

# Manual Arduino keyboard control
python3 src/serial_controller.py

# Mac video viewer only
python3 services/rpi-wifi-fallback/frame_receiver_server.py --display
```

### Sign Reaction Rules

Edit `SIGN_RULES` at the top of `src/skynet.py`:

```python
SIGN_RULES = {
    "stop":             {"speed": 0,   "steer": 0, "duration_s": 3},   # Stop 3 seconds
    "highway_entrance": {"speed": 30,  "steer": 0, "duration_s": 0},   # Speed up
    "highway_exit":     {"speed": 12,  "steer": 0, "duration_s": 0},   # Slow down
    "crosswalk":        {"speed": 10,  "steer": 0, "duration_s": 0},   # Crawl
    ...
}
DEFAULT_SPEED = 15  # Cruise speed between signs
```

### Troubleshooting

| Problem | Fix |
| --- | --- |
| `[Serial] ❌ Failed to open /dev/ttyACM0` | Check USB cable, try `ls /dev/ttyACM*` |
| Video window doesn't open | Start `frame_receiver_server.py` on Mac **before** `skynet.py` on Pi |
| YOLO too slow | Lower confidence: `--conf 0.3`, or `--no-filters` |
| Wrong laptop IP | Run `ip a` on Mac to find IP, pass with `--laptop-ip` |
| `picamera2` not found | `pip install picamera2`, or use `--webcam-index 0` |
