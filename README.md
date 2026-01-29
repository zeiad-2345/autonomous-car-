# BFMC - Brain Project

The project contains all the provided code for the RPi, more precisely:
- Firmware for communicating with the Nucleo and control the robot movements (Speed with constant current consumption, speed with constant speed, braking, moving and steering);
- Firmware for gathering data from the sensors (IMU and Camera);
- API's for communicating with the environmental servers at Bosch location;
- Simulated servers for the API's.


## The documentation is available in more details here:
[Documentation](https://bosch-future-mobility-challenge-documentation.readthedocs-hosted.com/)

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

