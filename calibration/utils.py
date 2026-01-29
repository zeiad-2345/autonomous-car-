import json
import numpy as np
import cv2
import os

CALIB_DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "calib_data.json")

def save_calibration_data(ret, mtx, dist, rvecs, tvecs, homography_matrix, file_path=CALIB_DATA_FILE):
    """Saves calibration matrices to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "homography_matrix": homography_matrix.tolist()
    }
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Calibration data saved to {file_path}")

def load_calibration_data(file_path=CALIB_DATA_FILE):
    """Loads calibration matrices from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No calibration data found at {file_path}")
        
    with open(file_path, "r") as f:
        data = json.load(f)
        
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])
    homography = np.array(data["homography_matrix"])
    
    return mtx, dist, homography

def apply_ipm(image, homography_matrix, output_size=(640, 480)):
    """Applies the Inverse Perspective Mapping using the homography matrix."""
    warped_img = cv2.warpPerspective(image, homography_matrix, output_size)
    return warped_img
