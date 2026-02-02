import cv2
import numpy as np
import glob
import argparse
import os
from utils import save_calibration_data

# Checkerboard dimensions (internal corners) - Typical BFMC pattern
# User can override via args
CHECKERBOARD = (8, 6) 
SQUARE_SIZE = 2.5 # cm (example)

def calibrate_intrinsic(image_dir, rows, cols):
    """
    Performs intrinsic camera calibration using checkerboard images.
    """
    # Termination criteria for corner sub-pix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE # Scale by square size

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    if not images:
        print(f"No images found in {image_dir}")
        return None, None

    print(f"Found {len(images)} images for intrinsic calibration.")
    
    gray = None
    shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"Chessboard found in {fname}")
        else:
            print(f"Chessboard NOT found in {fname}")

    if not objpoints:
        print("Calibration failed: No valid chessboards found.")
        return None, None

    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    print(f"Intrinsic calibration complete. RMSE: {ret}")
    return mtx, dist

def get_mouse_points(event, x, y, flags, params):
    """Mouse callback to select 4 points."""
    points = params['points']
    image = params['image']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 Points", image)

def calibrate_perspective(image_path, mtx, dist):
    """
    Calculates homography matrix from 4 manually selected points.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return None

    # Undistort first
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # UI for Point Selection
    print("Please select 4 points on the ground plane in CLOCKWISE order (TL, TR, BR, BL).")
    print("Press any key when done.")
    
    points = []
    cv2.imshow("Select 4 Points", undst)
    cv2.setMouseCallback("Select 4 Points", get_mouse_points, {'points': points, 'image': undst})
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        print("Error: Need exactly 4 points.")
        return None

    src_pts = np.float32(points)

    # Destination definition
    # Assuming standard lane width or projecting to a fixed view
    # Here we map the selected trapezoid to a rectangle satisfying standard aspect ratio or fixed size
    # For simplicity, we map to the full image width, spanning bottom to top
    
    # Simple approach: Map to a "Bird's Eye" rectangle
    # Width = Distance between bottom points (approx)
    # Height = Distance between top and bottom (approx)
    # Or just map to full 640x480 for visualization
    
    dst_pts = np.float32([
        [0, 0],       # Top-Left
        [w, 0],       # Top-Right
        [w, h],       # Bottom-Right
        [0, h]        # Bottom-Left
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M, UndistortInfo(mtx, dist, newcameramtx)

class UndistortInfo:
    def __init__(self, mtx, dist, newcameramtx):
        self.mtx = mtx
        self.dist = dist
        self.newcameramtx = newcameramtx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Camera & IPM")
    parser.add_argument("--dir", type=str, default="calibration/images", help="Directory with checkerboard images")
    parser.add_argument("--perspective_img", type=str, required=True, help="Image for perspective calibration (lane view)")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard internal rows")
    parser.add_argument("--cols", type=int, default=8, help="Checkerboard internal cols")
    
    args = parser.parse_args()

    # 1. Intrinsic
    mtx, dist = calibrate_intrinsic(args.dir, args.rows, args.cols)
    if mtx is None:
        exit(1)

    # 2. Extrinsic (Perspective)
    homography, undist_info = calibrate_perspective(args.perspective_img, mtx, dist)
    
    if homography is not None:
        # Save all results
        # Note: We save the original mtx/dist. 
        # The runtime application should undistort -> warp
        # For a truly optimized IPM, one might combine maps, but this is clearer.
        save_calibration_data(0, mtx, dist, [], [], homography)
