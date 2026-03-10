#!/usr/bin/env python3

import cv2
import numpy as np
import socket
from collections import deque
import math
import warnings

warnings.simplefilter('ignore', np.RankWarning)

#####################################
# === Configuration ===
HOST = '0.0.0.0'
PORT = 4000
FRAME_WIDTH = 1280
FRAME_HEIGHT = 960
REAL_LANE_WIDTH_CM = 35
LANE_HISTORY = 5

# ROI parameters
ROI_TOP_FRACTION = 0.7
ROI_BOTTOM_LEFT = 0.0
ROI_BOTTOM_RIGHT = 1.0
ROI_TOP_LEFT = 0.4
ROI_TOP_RIGHT = 0.6

# Morphology
NOISE_KERNEL = (3,3)
CONNECT_KERNEL = (5,5)
VERTICAL_KERNEL = (9,3)
OPEN_ITER = 2
CLOSE_ITER = 2
VERTICAL_CLOSE_ITER = 1

# Sliding window
NWINDOWS = 9
WINDOW_MARGIN = 60
MINPIX = 20

STRAIGHT_STD_THRESHOLD = 5

# Lane history for smoothing
left_lane_history = deque(maxlen=LANE_HISTORY)
right_lane_history = deque(maxlen=LANE_HISTORY)
#####################################

# === ROI / Trapezoid ===
def trapezoid_vertices(img):
    h,w = img.shape[:2]
    return np.array([[
        (int(ROI_BOTTOM_LEFT*w), h),
        (int(ROI_TOP_LEFT*w), int(ROI_TOP_FRACTION*h)),
        (int(ROI_TOP_RIGHT*w), int(ROI_TOP_FRACTION*h)),
        (int(ROI_BOTTOM_RIGHT*w), h)
    ]], dtype=np.int32)

def draw_roi_overlay(frame, vertices):
    overlay = frame.copy()
    cv2.fillPoly(overlay, vertices, (120,120,120))
    combined = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.polylines(combined, vertices, True, (0,0,255), 2)
    return combined

# === Thresholding + Morphology ===
def apply_thresholds(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    noise_kernel = np.ones(NOISE_KERNEL, np.uint8)
    connect_kernel = np.ones(CONNECT_KERNEL, np.uint8)
    vertical_kernel = np.ones(VERTICAL_KERNEL, np.uint8)
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, noise_kernel, iterations=OPEN_ITER)
    connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, connect_kernel, iterations=CLOSE_ITER)
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, vertical_kernel, iterations=VERTICAL_CLOSE_ITER)
    return connected

def mask_roi(binary, vertices):
    mask = np.zeros_like(binary)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(binary, mask)

# === BEV Transformation ===
def get_perspective_transform(frame):
    h,w = frame.shape[:2]
    src = np.float32([
        [ROI_BOTTOM_LEFT*w, h],
        [ROI_TOP_LEFT*w, ROI_TOP_FRACTION*h],
        [ROI_TOP_RIGHT*w, ROI_TOP_FRACTION*h],
        [ROI_BOTTOM_RIGHT*w, h]
    ])
    dst = np.float32([
        [0, h],
        [0, 0],
        [w, 0],
        [w, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def warp_bev(binary):
    M, Minv = get_perspective_transform(binary)
    bev = cv2.warpPerspective(binary, M, (binary.shape[1], binary.shape[0]), flags=cv2.INTER_LINEAR)
    return bev, Minv

# === Sliding Window Lane Detection ===
def sliding_window_lane_detection(binary):
    bev, Minv = warp_bev(binary)
    histogram = np.sum(bev[bev.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(bev.shape[0]/NWINDOWS)
    nonzero = bev.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for window in range(NWINDOWS):
        win_y_low = bev.shape[0] - (window+1)*window_height
        win_y_high = bev.shape[0] - window*window_height
        win_xleft_low = leftx_current - WINDOW_MARGIN
        win_xleft_high = leftx_current + WINDOW_MARGIN
        win_xright_low = rightx_current - WINDOW_MARGIN
        win_xright_high = rightx_current + WINDOW_MARGIN

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > MINPIX:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MINPIX:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    if len(left_lane_inds)==0 or len(right_lane_inds)==0:
        return None, None, Minv

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    left_fit, right_fit = None, None
    if len(leftx) > 10:
        left_fit = np.polyfit(lefty, leftx, 1 if np.std(leftx)<STRAIGHT_STD_THRESHOLD else 2)
    if len(rightx) > 10:
        right_fit = np.polyfit(righty, rightx, 1 if np.std(rightx)<STRAIGHT_STD_THRESHOLD else 2)

    return left_fit, right_fit, Minv

# === Lane Geometry & Curvature ===
def compute_lane_geometry(frame, left_fit, right_fit):
    if left_fit is None or right_fit is None:
        return 0,0,"Unknown",0

    y_eval = frame.shape[0]
    leftx = np.polyval(left_fit, y_eval)
    rightx = np.polyval(right_fit, y_eval)
    lane_center = (leftx + rightx)/2
    car_center = frame.shape[1]/2
    offset_pixels = car_center - lane_center
    lane_width_pixels = rightx - leftx
    cm_per_pixel = REAL_LANE_WIDTH_CM / lane_width_pixels
    offset_cm = offset_pixels * cm_per_pixel

    # Angle between bottom center and 60% height
    y1 = frame.shape[0]
    y2 = int(frame.shape[0]*0.6)
    center1 = (np.polyval(left_fit,y1) + np.polyval(right_fit,y1))/2
    center2 = (np.polyval(left_fit,y2) + np.polyval(right_fit,y2))/2
    dx = center1 - center2
    dy = y1 - y2
    angle = np.degrees(np.arctan2(dx, dy))

    # Lane curvature radius
    lane_type = "Straight"
    radius = 0
    ym_per_pix = 30/720
    xm_per_pix = REAL_LANE_WIDTH_CM / lane_width_pixels
    left_fit_cr = np.polyfit(np.array([y1,y2])*ym_per_pix, np.polyval(left_fit,[y1,y2])*xm_per_pix,2)
    right_fit_cr = np.polyfit(np.array([y1,y2])*ym_per_pix, np.polyval(right_fit,[y1,y2])*xm_per_pix,2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)/np.abs(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)/np.abs(2*right_fit_cr[0])
    radius = (left_curverad + right_curverad)/2

    if radius < 70:  # threshold to classify as curved
        lane_type = "Curved"

    return offset_cm, angle, lane_type, radius

# === Draw Dashed Center Line ===
def draw_dashed_line(img, x_vals, y_vals):
    for i in range(0,len(x_vals)-10,20):
        pt1 = (int(x_vals[i]), int(y_vals[i]))
        pt2 = (int(x_vals[i+10]), int(y_vals[i+10]))
        cv2.line(img, pt1, pt2, (0,0,0), 3)

# === Draw Lanes with BEV warp back ===
def draw_lanes(frame, left_fit, right_fit, Minv):
    if left_fit is None or right_fit is None:
        return frame

    ploty = np.linspace(0, frame.shape[0]-1, frame.shape[0])
    overlay = np.zeros_like(frame)

    leftx = np.polyval(left_fit, ploty)
    rightx = np.polyval(right_fit, ploty)

    left_lane_history.append(leftx)
    right_lane_history.append(rightx)
    leftx_smooth = np.mean(left_lane_history, axis=0)
    rightx_smooth = np.mean(right_lane_history, axis=0)

    pts_left = np.vstack([leftx_smooth, ploty]).T.astype(np.float32).reshape(-1,1,2)
    pts_right = np.flipud(np.vstack([rightx_smooth, ploty]).T.astype(np.float32)).reshape(-1,1,2)
    lane_pts = np.vstack([pts_left, pts_right])

    lane_pts_warped = cv2.perspectiveTransform(lane_pts, Minv)
    cv2.fillPoly(overlay, [np.int32(lane_pts_warped)], (0,255,0))

    left_line_warped = cv2.perspectiveTransform(pts_left, Minv)
    right_line_warped = cv2.perspectiveTransform(pts_right, Minv)
    cv2.polylines(overlay, [np.int32(left_line_warped)], False, (255,0,0),4)
    cv2.polylines(overlay, [np.int32(right_line_warped)], False, (255,0,0),4)

    centerx = (leftx_smooth + rightx_smooth)/2
    center_pts = np.vstack([centerx, ploty]).T.astype(np.float32).reshape(-1,1,2)
    center_pts_warped = cv2.perspectiveTransform(center_pts, Minv)
    draw_dashed_line(overlay, center_pts_warped[:,0,0], center_pts_warped[:,0,1])

    result = cv2.addWeighted(frame,1,overlay,0.7,0)
    return result

# === Lane Detection Demo ===
def demo_Lane_Detect(frame):
    vertices = trapezoid_vertices(frame)
    bw = apply_thresholds(frame)
    bw_roi = mask_roi(bw, vertices)
    left_fit, right_fit, Minv = sliding_window_lane_detection(bw_roi)
    lanes_frame = draw_lanes(frame, left_fit, right_fit, Minv)
    final_frame = draw_roi_overlay(lanes_frame, vertices)
    offset, angle, lane_type, radius = compute_lane_geometry(frame, left_fit, right_fit)

    cv2.putText(final_frame, f"Offset: {offset:.2f} cm", (30,50), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(final_frame, f"Angle: {angle:.2f} deg", (30,80), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(final_frame, f"Lane Type: {lane_type}", (30,110), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    if lane_type=="Curved":
        cv2.putText(final_frame, f"Radius: {radius:.1f} m", (30,140), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv2.imshow("Original", frame)
    cv2.imshow("Binary", bw)
    cv2.imshow("Binary ROI", bw_roi)
    cv2.imshow("Lane Detection", final_frame)
    return offset, angle, lane_type, radius

# === Perception ===
def Perception_Code(frame):
    return demo_Lane_Detect(frame)

# === Pull frame from socket ===
def pull_frame(conn):
    try:
        conn.sendall(b'F')
        size_data = conn.recv(8)
        if len(size_data) < 8: return None
        img_size = int.from_bytes(size_data, byteorder='big')
        img_bytes = b''
        while len(img_bytes) < img_size:
            packet = conn.recv(4096)
            if not packet: return None
            img_bytes += packet
        frame_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        return frame
    except Exception as e:
        print(f"[SERVER] Error: {e}")
        return None

# === Main Server ===
def main(args=None):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"[SERVER] Listening on {HOST}:{PORT}...")
    conn, addr = server_socket.accept()
    print(f"[SERVER] Connected to {addr}")

    try:
        while True:
            frame = pull_frame(conn)
            if frame is not None:
                offset, angle, lane_type, radius = Perception_Code(frame)
                Lane_msg = f"{offset},{angle}\n"
                conn.sendall(Lane_msg.encode('utf-8'))
                print("offset_cm:", offset, "angle_deg:", angle, "lane_type:", lane_type, "radius_m:", radius)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("[SERVER] No frame received or decoding failed.")
                break
    except KeyboardInterrupt:
        print("\n[SERVER] Interrupted by user.")
    finally:
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()
        print("Connection Terminated!")

#####################################
if __name__ == '__main__':
    main()
