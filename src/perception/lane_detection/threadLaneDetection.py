#!/usr/bin/env python3
"""
RAVEN — Native Lane Detection Thread
=====================================
Detects lane markings using pure OpenCV — NO ROS required.

Strategy (two fallbacks — most robust first):
  1. Sliding-window polynomial fit (best on straight + winding roads)
  2. Hough-line fallback (used when sliding window fails)

Track specifications (BFMC):
  City lane:    35 cm wide, 2 cm marking, 4.5 cm dash pattern
  Highway lane: 37 cm wide, 4 cm marking, 9 cm dash pattern

Publishes to SharedState:
  state.lane_error   — lateral offset from lane centre (-1.0 left … +1.0 right)
  state.lane_heading — angular offset from lane direction (radians)
  state.lane_type    — 'city' | 'highway' | 'unknown'

Usage in skynet.py:
    lane = LaneDetectionThread(state, args)
    lane.start()
"""

import threading
import time
import math
import cv2
import numpy as np
from collections import deque

# ── Known track parameters ─────────────────────────────────────────────────
LANE_WIDTH_CITY_CM     = 35.0   # cm
LANE_WIDTH_HIGHWAY_CM  = 37.0   # cm
LINE_WIDTH_CM          = 2.5    # cm  ← ACTUAL measured marking width (both solid & dashed)
DASH_GAP_CM            = 4.5    # cm  ← Gap between dashes (~4.4–4.5cm measured)
DASH_ON_CITY_CM        = 4.5    # cm
DASH_ON_HIGHWAY_CM     = 9.0    # cm

# ── Image pipeline constants ───────────────────────────────────────────────
# ROI: only look at the bottom part of the frame (road ahead, not sky)
# Lower the top fraction to include more of the road for detection
ROI_TOP_FRACTION   = 0.70
# IPM (bird's-eye) warp source/destination corners (tune per camera mount)
# These are fractions of frame width/height — adjust after calibration
IPM_SRC = np.float32([
    [0.00, 1.00],   # bottom-left
    [1.00, 1.00],   # bottom-right
    [0.85, ROI_TOP_FRACTION],  # top-right
    [0.23, ROI_TOP_FRACTION],  # top-left
])
IPM_DST = np.float32([
    [0.00, 1.00],
    [1.00, 1.00],
    [1.00, 0.00],
    [0.00, 0.00],
])

# Sliding window
N_WINDOWS     = 9
WINDOW_MARGIN = 70    # px half-width around window centroid (wider search)
MIN_PIX       = 20    # minimum pixels to recenter window (relaxed)

# Contour/area thresholds for the sliding-window point extraction
CONTOUR_MIN_AREA = 30

LANE_DETECT_HZ = 15   # run at 15 Hz (city: moves 1.3cm/frame; highway: 2.7cm/frame)

# Lane smoothing history
LANE_HISTORY = 10
left_lane_history = deque(maxlen=LANE_HISTORY)
right_lane_history = deque(maxlen=LANE_HISTORY)

# Real-world lane width for offset calculation
REAL_LANE_WIDTH_CM = 35.0

# Camera horizontal offset: camera is slightly LEFT of car center.
# Positive value shifts the assumed car position to the RIGHT in the image.
# 0.05 = 5% of image width (~32px on 640px frame). Tune on track.
CAMERA_CENTER_OFFSET = -0.25

# Look-ahead: evaluate lane centre at this fraction of the warped image height
# 1.0 = right under the car (reactive only), 0.5 = midway (anticipatory)
LOOKAHEAD_Y_FRAC = 0.55


class LaneDetectionThread(threading.Thread):
    """
    Reads frames from SharedState (populated by PerceptionThread),
    runs the lane detection pipeline, and writes lateral error + heading
    back to SharedState for the PlannerThread to consume.
    """

    def __init__(self, state, args):
        super().__init__(name="LaneDetectionThread", daemon=True)
        self.state = state
        self.args  = args
        self._interval = 1.0 / LANE_DETECT_HZ
        self._h = None
        self._w = None
        self._M = None   # IPM warp matrix (computed once from first frame)
        self._Minv = None

        # State for handling detection failures
        self.missing_counter = 0
        self.is_in_curve = False
        self.error_history = deque(maxlen=LANE_HISTORY)
        self.left_history = deque(maxlen=LANE_HISTORY)
        self.right_history = deque(maxlen=LANE_HISTORY) 
        self.heading_history = deque(maxlen=LANE_HISTORY)

        # EMA smoothing for output stability
        self._ema_alpha = 0.4   # 0.0 = full smoothing, 1.0 = no smoothing
        self._ema_error = 0.0
        self._ema_heading = 0.0
        self.last_valid_error = 0.0
        self.lane_width_px = None
        # The smoothing_factor is the "coefficient" for the EMA.
        # A higher value makes it more responsive to new data, a lower value makes it smoother.
        self.smoothing_factor = 0.2
        self.smoothed_error = 0.0
        self.smoothed_heading = 0.0
        self.smoothed_lfit = None
        self.smoothed_rfit = None
        # Number of frames to "coast" using memory before resetting
        self.FAILURE_COAST_FRAMES = 15  # ~1.0s at 15Hz
    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        print("[Lane] Starting…")
        last_time = 0

        while self.state.is_running():
            now = time.time()
            if now - last_time < self._interval:
                time.sleep(0.005)
                continue
            last_time = now

            frame = self.state.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            try:
                result = self._pipeline(frame)
                if result is not None:
                   self.state.set_lane(result)
            except Exception as e:
                # Never crash the car over a detection failure
                print(f"[Lane] Pipeline Exception: {e}")

        print("[Lane] Stopped.")

    # ── Full Pipeline ─────────────────────────────────────────────────────

    def _pipeline(self, frame):
        """
        Returns dict:
            {error, heading, lane_type, left_fit, right_fit, Minv, frame_shape}
        """
        h, w = frame.shape[:2]
        if self._M is None:
            self._compute_warp(w, h)
            self._h, self._w = h, w

        # 1. Convert to bird's eye (IPM)
        bird = cv2.warpPerspective(frame, self._M, (w, h))

        # 2. Threshold for white lane markings
        binary = self._threshold(bird)

        # Create a BGR visualization image to draw boxes on safely
        out_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 3. Detect lane type (city vs highway) based on line width ratio
        lane_type = self._classify_lane_type(binary)

        # 4. Try sliding-window, fall back to Hough
        result = self._sliding_window(binary, out_img)
        if result is None:
            result = self._hough_fallback(binary, out_img)
       
        if result is None:
            print("[Lane] No lane detected in this frame.")
            self.missing_counter += 1
            error, heading = 0.0, 0.0  # Default safety values
            left_fit, right_fit = None, None

            if self.missing_counter >4:
                # Short-term failure: coast using memory
                eh = list(self.error_history)
                hh = list(self.heading_history)
                lh = list(self.left_history)
                rh = list(self.right_history)

                # Use frames 3-6 from history if available
                if len(eh) >= 9: eh = eh[5:9] # type: ignore
                if len(hh) >= 9: hh = hh[5:9] # type: ignore
                if len(lh) >= 9: lh = lh[5:9] # type: ignore
                if len(rh) >= 9: rh = rh[5:9] # type: ignore

                # --- Calculate fallback values from sliced history ---
                error_from_history = float(np.mean(eh)) if len(eh) > 0 else 0.0

           
                if not self.is_in_curve or error_from_history < 0.1:  # If we're likely on a straight, use generic straight fallback
                    print("[Lane] Fallback: STRAIGHT (small historical error)")
                    error = 0.0
                    heading = 0.0
                    # Provide generic straight fits for visualization
                    
                    left_fit = np.array([0.0, 0.0, self._w * 0.2])
                    right_fit = np.array([0.0, 0.0, self._w * 0.85])
                else: # Otherwise, it's a curve, so use the historical values
                    print("[Lane] Fallback: CURVE (large error)")
                    error = error_from_history
                    if len(hh) > 0: heading = float(np.mean(hh))
                    if len(lh) > 0: left_fit = np.mean(lh, axis=0)
                    if len(rh) > 0: right_fit = np.mean(rh, axis=0)
            # else: Long-term failure, use default error=0, heading=0, fits=None

            return {"error": float(error), "heading": float(heading), "lane_type": lane_type,
                    "left_fit": left_fit, "right_fit": right_fit,
                    "Minv": self._Minv, "frame_shape": (h, w), "binary": out_img}
        else:
            # --- SUCCESSFUL DETECTION ---
            self.missing_counter = 0
            error, heading, lfit, rfit = result

            # Only trust sliding window (high confidence) to update the memory
            if lfit is not None or rfit is not None:
                self.error_history.append(error)
                self.heading_history.append(heading)
                if lfit is not None:
                    self.left_history.append(lfit)
                if rfit is not None:
                    self.right_history.append(rfit)

            # Determine if we are in a curve from the polynomial fit.
            l_curv = abs(lfit[0]) if lfit is not None and len(lfit) == 3 else 0
            r_curv = abs(rfit[0]) if rfit is not None and len(rfit) == 3 else 0
            CURVATURE_THRESHOLD = 1e-4
            
            self.is_in_curve = l_curv > CURVATURE_THRESHOLD or r_curv > CURVATURE_THRESHOLD
            
            # EMA smoothing to reduce jitter
            
            # In straight lines, use heavy smoothing (alpha=0.15) to create a "saved fixed center"
            # In curves, use light smoothing (alpha=0.8) to stay responsive
            alpha = 0.8 if self.is_in_curve else 0.15
            
            if self.smoothed_error == 0.0 and self.smoothed_heading == 0.0:
                self.smoothed_error, self.smoothed_heading = error, heading
            else:
                self.smoothed_error = (alpha * error) + ((1.0 - alpha) * self.smoothed_error)
                self.smoothed_heading = (alpha * heading) + ((1.0 - alpha) * self.smoothed_heading)

            return {"error": float(self.smoothed_error), "heading": float(self.smoothed_heading), "lane_type": lane_type,       
            "left_fit": lfit, "right_fit": rfit,
                "Minv": self._Minv, "frame_shape": (h, w), "binary": out_img}

    # ── Step 1: IPM warp ──────────────────────────────────────────────────

    def _compute_warp(self, w, h):
        src = (IPM_SRC * np.float32([w, h])).astype(np.float32)
        dst = (IPM_DST * np.float32([w, h])).astype(np.float32)
        self._M    = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)

    # ── Step 2: Threshold ─────────────────────────────────────────────────

    def _threshold(self, img):
        """White-line threshold: HSV S-channel low, V-channel high."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # White: low saturation, high value
        lower = np.array([0,   0,   180], dtype=np.uint8)
        upper = np.array([180, 60,  255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower, upper)
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN,  kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        return mask_white

    # ── Step 3: Lane type classifier ──────────────────────────────────────

    def _classify_lane_type(self, binary):
        """
        Estimate whether we're on city (4.5cm dashes) or highway (9cm dashes)
        by measuring the gap pattern in the binary image column projection.
        """
        col_sum = binary.sum(axis=1).astype(np.float32)
        # Normalise
        if col_sum.max() == 0:
            return "unknown"
        col_sum /= col_sum.max()
        # Count transitions (on→off) per vertical pixel — more transitions = city dashes
        transitions = np.sum(np.diff((col_sum > 0.3).astype(int)) > 0)
        # Highway has longer dashes → fewer transitions in the same image height
        return "highway" if transitions < 4 else "city"

    # ── Step 4a: Sliding window ───────────────────────────────────────────

    def _sliding_window(self, binary, out_img=None):
        """
        Classic sliding-window lane finder.
        Returns (lateral_error, heading_error, viz_img) or None.
        """
      
        h, w = binary.shape
        # Histogram of bottom half to find starting points
        hist = binary[h // 2:, :].sum(axis=0).astype(np.float32)
        
        # Dynamically shift the histogram split based on history to catch sharp curves
        mid = w // 2
        if len(self.error_history) > 0:
            last_error = self.error_history[-1]
            # Shift by up to 35% of the screen width depending on how hard we are turning
            shift = int(last_error * (w * 0.35))
            mid = max(w // 4, min(3 * w // 4, mid + shift))
            
        lx_base = int(np.argmax(hist[:mid]))
        rx_base = int(mid + np.argmax(hist[mid:]))

        l_pix, l_y = [], []
        r_pix, r_y = [], []
        valid_l_windows=0
        valid_r_windows=0
        window_height = 40
        y = h
        while y > 0:
            y0 = max(0, y - window_height)

            l_added = False
            # Left window
            x1 = max(0, lx_base - WINDOW_MARGIN)
            x2 = min(w, lx_base + WINDOW_MARGIN)
            img = binary[y0:y, x1:x2]
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cxx = []
            for contour in contours:
                if cv2.contourArea(contour) < CONTOUR_MIN_AREA:
                    continue
                m = cv2.moments(contour)
                if m.get('m00', 0) != 0:
                    cx = int(m['m10'] / m['m00'])
                    cxx.append(cx)
            if len(cxx) > 0:
                avg_cx = int(np.mean(cxx))
                lx_base = x1 + avg_cx
                l_pix.append(lx_base)
                l_y.append(y0 + window_height // 2)
                valid_l_windows += 1
                l_added = True

            # Right window
            r_added = False
            xr1 = max(0, rx_base - WINDOW_MARGIN)
            xr2 = min(w, rx_base + WINDOW_MARGIN)
            img = binary[y0:y, xr1:xr2]
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cxx = []
            for contour in contours:
                if cv2.contourArea(contour) < CONTOUR_MIN_AREA:
                    continue
                m = cv2.moments(contour)
                if m.get('m00', 0) != 0:
                    cx = int(m['m10'] / m['m00'])
                    cxx.append(cx)
            if len(cxx) > 0:
                avg_cx = int(np.mean(cxx))
                rx_base = xr1 + avg_cx
                r_pix.append(rx_base)
                r_y.append(y0 + window_height // 2)
                valid_r_windows += 1
                r_added = True

            # Remove points if the detected lane width is very narrow
            if l_added and r_added:
                if (rx_base - lx_base) < (w * 0.4):  # threshold: less than 40% of frame width
                    l_pix.pop()
                    l_y.pop()
                    valid_l_windows -= 1
                    r_pix.pop()
                    r_y.pop()
                    valid_r_windows -= 1
                    # Invalidate so we don't draw the dropped points
                    l_added = False
                    r_added = False

            if out_img is not None:
                if l_added:
                    cv2.rectangle(out_img, (max(0, l_pix[-1] - WINDOW_MARGIN), y0), (min(w, l_pix[-1] + WINDOW_MARGIN), y), (255, 0, 0), 2)
                    cv2.circle(out_img, (l_pix[-1], y0 + window_height // 2), 4, (0, 0, 255), -1)
                if r_added:
                    cv2.rectangle(out_img, (max(0, r_pix[-1] - WINDOW_MARGIN), y0), (min(w, r_pix[-1] + WINDOW_MARGIN), y), (255, 0, 0), 2)
                    cv2.circle(out_img, (r_pix[-1], y0 + window_height // 2), 4, (0, 255, 0), -1)

            y -= window_height
        
        # After scanning all windows, ensure we found sufficient points
        if valid_l_windows < 3 and valid_r_windows < 3:
            return None

        lx = np.array(l_pix)
        ly = np.array(l_y)
        rx = np.array(r_pix)
        ry = np.array(r_y)
       
            
        # Fit 2nd-degree polynomial y = f(x) → but we fit x = f(y) for near-vertical lanes
        try:
            if len(lx) >= 3:
                lfit = np.polyfit(ly, lx, 2)
            else:
                lfit = None
            if len(rx) >= 3:
                rfit = np.polyfit(ry, rx, 2)
            else:
                rfit = None
        except np.linalg.LinAlgError:
            return None

        # If only one side was found, generate a hypothetical opposite fit
        # by horizontally shifting the detected polynomial. The shift is
        # estimated as a fraction of image width (heuristic) so we can
        # still render a plausible lane when one side is occluded.
        # The assumed real lane width is REAL_LANE_WIDTH_CM but without
        # camera calibration we convert it to pixels heuristically.
        
        if lfit is not None and rfit is None:
            shift_px = float(w) * 1.00  # heuristic pixel lane width
            # Copy coefficients and add shift to constant term
            rfit = np.array(lfit, copy=True)
            if rfit.size == 3:
                rfit[2] = rfit[2] + shift_px
            elif rfit.size == 2:
                rfit[1] = rfit[1] + shift_px
        elif rfit is not None and lfit is None:
            shift_px = float(w) * 1.00
            lfit = np.array(rfit, copy=True)
            if lfit.size == 3:
                lfit[2] = lfit[2] - shift_px
            elif lfit.size == 2:
                lfit[1] = lfit[1] - shift_px
       
        # Lane centre at LOOK-AHEAD point (not bottom — gives time to react)
        y_eval = float(h) 
        if lfit is not None and rfit is not None :
            l_x_bottom = np.polyval(lfit, y_eval)
            r_x_bottom = np.polyval(rfit, y_eval)
            
            lane_centre = (l_x_bottom + r_x_bottom) / 2.0
        elif lfit is not None:
            # Only left lane visible — shift center further right
            lane_centre = np.polyval(lfit, y_eval) + w * 0.90/2.0
        elif rfit is not None:
            # Only right lane visible — shift center further left
            lane_centre = np.polyval(rfit, y_eval) - w * 0.90/2.0
        else:
            return None

        img_centre = (w / 2.0) + (w * CAMERA_CENTER_OFFSET)  # shift target to the right, moves car left
        # Normalised error: -1 = car at left edge, +1 = car at right edge
        error = (lane_centre - img_centre) / (w / 2.0)

        # Approximate heading from polynomial slope at bottom
        if lfit is not None and rfit is not None:
            l_slope = float(2 * lfit[0] * y_eval + lfit[1])
            r_slope = float(2 * rfit[0] * y_eval + rfit[1])
            slope = (l_slope + r_slope) / 2.0
        elif lfit is not None:
            slope = float(2 * lfit[0] * y_eval + lfit[1])
        else:
            slope = float(2 * rfit[0] * y_eval + rfit[1])

        heading = math.atan2(slope, 1.0)  # radians
        if lfit is None and rfit is None:
            return None


        return error, heading, lfit, rfit

    # ── Step 4b: Hough fallback ───────────────────────────────────────────

    def _hough_fallback(self, binary, out_img=None):
        """
        Simple Hough-line approach as a fallback when sliding window has insufficient pixels.
        """
       
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=30, minLineLength=30, maxLineGap=20)
        if lines is None:
            return None

        h, w = binary.shape
        left_x, right_x = [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Filter near-horizontal lines
            if abs(slope) < 0.3:
                continue
                
            if out_img is not None:
                cv2.line(out_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
            midx = (x1 + x2) / 2
            if midx < w / 2:
                left_x.append(midx)
            else:
                right_x.append(midx)

        if not left_x and not right_x:
            return None

        l_centre = float(np.mean(left_x))  if left_x  else None
        r_centre = float(np.mean(right_x)) if right_x else None

        if l_centre and r_centre:
            lane_centre = (l_centre + r_centre) / 2.0
        elif l_centre:
            lane_centre = l_centre + w * 0.45
        else:
            lane_centre = r_centre - w * 0.45

        img_centre = (w / 2.0) + (w * CAMERA_CENTER_OFFSET)
        error   = (lane_centre - img_centre) / (w / 2.0)
        heading = 0.0  # Hough fallback does not compute heading reliably
        return error, heading, None, None


# ══════════════════════════════════════════════════════════════════════════════
# Visualization helpers  (called from main display loop, NOT from the thread)
# ══════════════════════════════════════════════════════════════════════════════

def draw_roi_overlay(frame):
    """Draw the IPM source trapezoid on the frame."""
    h, w = frame.shape[:2]
    pts = (IPM_SRC * np.float32([w, h])).astype(np.int32)
    pts_reshaped = pts.reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts_reshaped], (120, 120, 120))
    combined = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.polylines(combined, [pts_reshaped], True, (0, 0, 255), 2)
    return combined


def draw_dashed_line(img, x_vals, y_vals, color=(0, 0, 0), thickness=3, dash_len=10, gap_len=10):
    """Draw a dashed line given arrays of x and y coordinates."""
    n = len(x_vals)
    i = 0
    while i + dash_len < n:
        pt1 = (int(x_vals[i]), int(y_vals[i]))
        pt2 = (int(x_vals[i + dash_len]), int(y_vals[i + dash_len]))
        cv2.line(img, pt1, pt2, color, thickness)
        i += dash_len + gap_len


def draw_lanes(frame, left_fit, right_fit, Minv):
    """
    Draw green lane fill, blue boundary lines, and dashed center line.
    Uses the thread's polynomial fits + inverse perspective matrix.
    """
    if left_fit is None or right_fit is None:
        return frame
    if Minv is None:
        return frame

    h, w = frame.shape[:2]
    ploty = np.linspace(0, h - 1, h)
    overlay = np.zeros_like(frame)

    leftx = np.polyval(left_fit, ploty)
    rightx = np.polyval(right_fit, ploty)

    # Smooth with history
    left_lane_history.append(leftx)
    right_lane_history.append(rightx)
    leftx_smooth = np.mean(left_lane_history, axis=0)
    rightx_smooth = np.mean(right_lane_history, axis=0)

    # Green fill between lanes
    pts_left = np.vstack([leftx_smooth, ploty]).T.astype(np.float32).reshape(-1, 1, 2)
    pts_right = np.flipud(np.vstack([rightx_smooth, ploty]).T.astype(np.float32)).reshape(-1, 1, 2)
    lane_pts = np.vstack([pts_left, pts_right])

    lane_pts_warped = cv2.perspectiveTransform(lane_pts, Minv)
    cv2.fillPoly(overlay, [np.int32(lane_pts_warped)], (0, 255, 0))

    # Blue boundary lines
    left_line_warped = cv2.perspectiveTransform(pts_left, Minv)
    right_line_warped = cv2.perspectiveTransform(pts_right, Minv)
    cv2.polylines(overlay, [np.int32(left_line_warped)], False, (255, 0, 0), 4)
    cv2.polylines(overlay, [np.int32(right_line_warped)], False, (255, 0, 0), 4)

    # Dashed center line
    centerx = (leftx_smooth + rightx_smooth) / 2
    center_pts = np.vstack([centerx, ploty]).T.astype(np.float32).reshape(-1, 1, 2)
    center_pts_warped = cv2.perspectiveTransform(center_pts, Minv)
    draw_dashed_line(overlay, center_pts_warped[:, 0, 0], center_pts_warped[:, 0, 1])

    result = cv2.addWeighted(frame, 1, overlay, 0.7, 0)
    return result


def draw_lane_hud(frame, lane_result):
    """
    Draw full lane HUD on frame using the thread's result dict.
    Shows: offset in cm, heading, lane type, normalised error.
    """
    if lane_result is None:
        cv2.putText(frame, "Lane: no data", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        return frame

    error = lane_result.get("error", 0.0)
    heading = lane_result.get("heading", 0.0)
    lane_type = lane_result.get("lane_type", "unknown")
    left_fit = lane_result.get("left_fit")
    right_fit = lane_result.get("right_fit")
    frame_shape = lane_result.get("frame_shape", (480, 640))

    heading_deg = math.degrees(heading)

    # Compute offset in cm if we have both fits
    offset_cm = 0.0
    if left_fit is not None and right_fit is not None:
        y_eval = float(frame_shape[0])
        leftx = np.polyval(left_fit, y_eval)
        rightx = np.polyval(right_fit, y_eval)
        lane_width_px = rightx - leftx
        if lane_width_px > 0:
            cm_per_px = REAL_LANE_WIDTH_CM / lane_width_px
            lane_centre = (leftx + rightx) / 2.0
            car_centre = (frame_shape[1] / 2.0) + (frame_shape[1] * CAMERA_CENTER_OFFSET)
            offset_cm = (car_centre - lane_centre) * cm_per_px

    cv2.putText(frame, f"Offset: {offset_cm:.2f} cm", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Heading: {heading_deg:.2f} deg", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Lane: {lane_type}", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Error: {error:+.3f}", (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame
