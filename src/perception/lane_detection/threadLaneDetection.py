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

# ── Known track parameters ─────────────────────────────────────────────────
LANE_WIDTH_CITY_CM     = 35.0   # cm
LANE_WIDTH_HIGHWAY_CM  = 37.0   # cm
LINE_WIDTH_CM          = 2.5    # cm  ← ACTUAL measured marking width (both solid & dashed)
DASH_GAP_CM            = 4.5    # cm  ← Gap between dashes (~4.4–4.5cm measured)
DASH_ON_CITY_CM        = 4.5    # cm
DASH_ON_HIGHWAY_CM     = 9.0    # cm

# ── Image pipeline constants ───────────────────────────────────────────────
# ROI: only look at the bottom 55% of the frame (road ahead, not sky)
ROI_TOP_FRACTION   = 0.45
# IPM (bird's-eye) warp source/destination corners (tune per camera mount)
# These are fractions of frame width/height — adjust after calibration
IPM_SRC = np.float32([
    [0.15, 1.00],   # bottom-left
    [0.85, 1.00],   # bottom-right
    [0.58, ROI_TOP_FRACTION],  # top-right
    [0.42, ROI_TOP_FRACTION],  # top-left
])
IPM_DST = np.float32([
    [0.10, 1.00],
    [0.90, 1.00],
    [0.90, 0.00],
    [0.10, 0.00],
])

# Sliding window
N_WINDOWS     = 9
WINDOW_MARGIN = 50    # px half-width around window centroid
MIN_PIX       = 30    # minimum pixels to recenter window

LANE_DETECT_HZ = 15   # run at 15 Hz (city: moves 1.3cm/frame; highway: 2.7cm/frame)


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
                self.state.set_lane(result)
            except Exception as e:
                # Never crash the car over a detection failure
                pass

        print("[Lane] Stopped.")

    # ── Full Pipeline ─────────────────────────────────────────────────────

    def _pipeline(self, frame):
        """
        Returns dict:
            {error: float, heading: float, lane_type: str, annotated: ndarray | None}
        """
        h, w = frame.shape[:2]
        if self._M is None:
            self._compute_warp(w, h)
            self._h, self._w = h, w

        # 1. Convert to bird's eye (IPM)
        bird = cv2.warpPerspective(frame, self._M, (w, h))

        # 2. Threshold for white lane markings
        binary = self._threshold(bird)

        # 3. Detect lane type (city vs highway) based on line width ratio
        lane_type = self._classify_lane_type(binary)

        # 4. Try sliding-window, fall back to Hough
        result = self._sliding_window(binary)
        if result is None:
            result = self._hough_fallback(binary)

        if result is None:
            return {"error": 0.0, "heading": 0.0, "lane_type": lane_type, "annotated": None}

        error, heading, viz = result
        return {"error": error, "heading": heading, "lane_type": lane_type, "annotated": viz}

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

    def _sliding_window(self, binary):
        """
        Classic sliding-window lane finder.
        Returns (lateral_error, heading_error, viz_img) or None.
        """
        h, w = binary.shape
        # Histogram of bottom half to find starting points
        hist = binary[h // 2:, :].sum(axis=0).astype(np.float32)
        mid  = w // 2
        lx_base = int(np.argmax(hist[:mid]))
        rx_base = int(mid + np.argmax(hist[mid:]))

        win_h = h // N_WINDOWS
        lx_cur, rx_cur = lx_base, rx_base
        l_pix, r_pix = [], []

        nz = binary.nonzero()
        nz_y, nz_x = np.array(nz[0]), np.array(nz[1])

        for win in range(N_WINDOWS):
            y_lo = h - (win + 1) * win_h
            y_hi = h - win * win_h

            # Left window
            good_l = ((nz_y >= y_lo) & (nz_y < y_hi) &
                      (nz_x >= lx_cur - WINDOW_MARGIN) &
                      (nz_x <  lx_cur + WINDOW_MARGIN)).nonzero()[0]
            # Right window
            good_r = ((nz_y >= y_lo) & (nz_y < y_hi) &
                      (nz_x >= rx_cur - WINDOW_MARGIN) &
                      (nz_x <  rx_cur + WINDOW_MARGIN)).nonzero()[0]

            l_pix.extend(good_l)
            r_pix.extend(good_r)

            if len(good_l) >= MIN_PIX:
                lx_cur = int(np.mean(nz_x[good_l]))
            if len(good_r) >= MIN_PIX:
                rx_cur = int(np.mean(nz_x[good_r]))

        if len(l_pix) < MIN_PIX * 2 and len(r_pix) < MIN_PIX * 2:
            return None  # Not enough evidence

        l_pix, r_pix = np.array(l_pix), np.array(r_pix)
        ly, lx = nz_y[l_pix], nz_x[l_pix]
        ry, rx = nz_y[r_pix], nz_x[r_pix]

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

        # Lane centre at bottom of image
        y_eval = float(h)
        if lfit is not None and rfit is not None:
            l_x_bottom = np.polyval(lfit, y_eval)
            r_x_bottom = np.polyval(rfit, y_eval)
            lane_centre = (l_x_bottom + r_x_bottom) / 2.0
        elif lfit is not None:
            # Only left lane visible — assume standard lane width
            lane_centre = np.polyval(lfit, y_eval) + w * 0.35 / 2.0
        elif rfit is not None:
            lane_centre = np.polyval(rfit, y_eval) - w * 0.35 / 2.0
        else:
            return None

        img_centre  = w / 2.0
        # Normalised error: -1 = car at left edge, +1 = car at right edge
        error = (lane_centre - img_centre) / (w / 2.0)

        # Approximate heading from polynomial slope at bottom
        if lfit is not None and rfit is not None:
            l_slope = float(2 * lfit[0] * y_eval + lfit[1])
            r_slope = float(2 * rfit[0] * y_eval + rfit[1])
            slope   = (l_slope + r_slope) / 2.0
        elif lfit is not None:
            slope = float(2 * lfit[0] * y_eval + lfit[1])
        else:
            slope = float(2 * rfit[0] * y_eval + rfit[1])

        heading = math.atan2(slope, 1.0)  # radians

        return error, heading, None  # viz skipped for performance

    # ── Step 4b: Hough fallback ───────────────────────────────────────────

    def _hough_fallback(self, binary):
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
            lane_centre = l_centre + w * 0.175
        else:
            lane_centre = r_centre - w * 0.175

        error   = (lane_centre - w / 2.0) / (w / 2.0)
        heading = 0.0  # Hough fallback does not compute heading reliably
        return error, heading, None
