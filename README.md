# BFMC - Brain Project

The project contains all the provided code for the RPi, more precisely:
- Firmware for communicating with the Nucleo and control the robot movements (Speed with constant current consumption, speed with constant speed, braking, moving and steering);
- Firmware for gathering data from the sensors (IMU and Camera);
- API's for communicating with the environmental servers at Bosch location;
- Simulated servers for the API's.

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

### Calibration Artifacts
The tools generate `calibration/data/calib_data.json` containing:
- `camera_matrix`: Intrinsic parameters
- `dist_coeffs`: Distortion coefficients
- `homography_matrix`: The IPM transformation matrix

