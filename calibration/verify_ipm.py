import cv2
import numpy as np
import argparse
from utils import load_calibration_data, apply_ipm

def verify(image_path):
    try:
        mtx, dist, homography = load_calibration_data()
    except FileNotFoundError:
        print("Run calibration first!")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return

    # 1. Undistort
    h, w = img.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # 2. Warp
    warped = apply_ipm(undst, homography, output_size=(w, h))

    # Show side-by-side
    combined = np.hstack((undst, warped))
    cv2.namedWindow("Original (Undistorted) vs IPM", cv2.WINDOW_NORMAL)
    cv2.imshow("Original (Undistorted) vs IPM", combined)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to test image")
    args = parser.parse_args()
    verify(args.image)
