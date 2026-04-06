"""
RAVEN — Post-Detection Sign Filters
=====================================
Validates YOLO detections using shape, color, and size heuristics
to reduce false positives (e.g., red shirts mistaken for stop signs,
blue cars mistaken for one-way signs).

Usage:
    from sign_filters import validate_detection

    # Inside your detection loop:
    if validate_detection(frame, sign_name, (x1, y1, x2, y2)):
        # Accept detection
    else:
        # Reject as false positive

Filter Pipeline:
    YOLO Detection → Shape Check → Color Check → Size Check → ✅ Accept
                         ↓              ↓             ↓
                      ❌ reject       ❌ reject     ❌ reject

Design Notes:
    - Uses HSV color space (NOT hex/RGB) so lighting changes don't matter.
      HSV separates Hue (color) from Value (brightness), meaning a stop sign
      in shadow and in sunlight both have the same Hue range.
    - Thresholds are intentionally GENEROUS — we only reject when the expected
      color is clearly absent (<5% of pixels). Better to let a marginal
      detection through than reject a real sign.
    - All thresholds are tunable via the SIGN_FILTERS dict below.
"""

import cv2
import numpy as np


# ─── Filter Configuration ────────────────────────────────────────────────────
# Each sign has expected properties. The filter rejects detections that
# clearly violate these expectations.
#
# aspect_ratio: Expected height/width ratio of the bounding box.
#               Most signs are square (~1.0). A car is wide+flat (~0.3).
# aspect_tol:   How much the aspect ratio can deviate (±).
# hsv_ranges:   List of (H_low, S_low, V_low, H_high, S_high, V_high).
#               Pixels matching ANY range count as "expected color."
#               Red wraps around 0°/180° in HSV, so it needs TWO ranges.
# color_min:    Minimum fraction of bounding box pixels that must match
#               the expected color. Set low (0.03 = 3%) to be generous.
# min_area_px:  Minimum bounding box area in pixels (reject tiny noise).
# max_area_frac: Maximum fraction of frame area (reject huge false positives).
# ──────────────────────────────────────────────────────────────────────────────

SIGN_FILTERS = {
    "stop": {
        "aspect_ratio": 1.0,
        "aspect_tol": 0.5,
        # Red in HSV wraps around 0°/180°, so we need two ranges
        "hsv_ranges": [
            (0, 30, 20, 12, 255, 255),      # Red low end (wider, lower S/V)
            (170, 30, 20, 179, 255, 255),    # Red high end (wider, lower S/V)
        ],
        "color_min": 0.02,
        "min_area_px": 150,
        "max_area_frac": 0.40,
    },
    "no_entry": {
        "aspect_ratio": 1.0,
        "aspect_tol": 0.5,
        "hsv_ranges": [
            (0, 30, 20, 12, 255, 255),
            (170, 30, 20, 179, 255, 255),
        ],
        "color_min": 0.02,
        "min_area_px": 150,
        "max_area_frac": 0.40,
    },
    "parking": {
        "aspect_ratio": 1.0,
        "aspect_tol": 0.5,
        "hsv_ranges": [
            (100, 50, 50, 130, 255, 255),    # Blue
        ],
        "color_min": 0.03,
        "min_area_px": 200,
        "max_area_frac": 0.40,
    },
    "crosswalk": {
        "aspect_ratio": 1.0,
        "aspect_tol": 0.6,
        "hsv_ranges": [
            (100, 50, 50, 130, 255, 255),    # Blue
        ],
        "color_min": 0.03,
        "min_area_px": 200,
        "max_area_frac": 0.40,
    },
    "roundabout": {
        "aspect_ratio": 1.0,
        "aspect_tol": 0.5,
        "hsv_ranges": [
            (100, 50, 50, 130, 255, 255),    # Blue
        ],
        "color_min": 0.03,
        "min_area_px": 200,
        "max_area_frac": 0.40,
    },
    "one_way": {
        "aspect_ratio": 1.2,
        "aspect_tol": 0.6,
        "hsv_ranges": [
            (100, 50, 50, 130, 255, 255),    # Blue
        ],
        "color_min": 0.03,
        "min_area_px": 200,
        "max_area_frac": 0.40,
    },
    "priority": {
        "aspect_ratio": 1.0,
        "aspect_tol": 0.5,
        "hsv_ranges": [
            (15, 50, 50, 40, 255, 255),      # Yellow / Orange
        ],
        "color_min": 0.03,
        "min_area_px": 200,
        "max_area_frac": 0.40,
    },
    "highway_entrance": {
        "aspect_ratio": 1.0,
        "aspect_tol": 0.6,
        "hsv_ranges": [
            (80, 30, 50, 130, 255, 255),     # Blue-Green range
        ],
        "color_min": 0.02,
        "min_area_px": 150,
        "max_area_frac": 0.40,
    },
    "highway_exit": {
        "aspect_ratio": 1.0,
        "aspect_tol": 0.6,
        "hsv_ranges": [
            (80, 30, 50, 130, 255, 255),     # Blue-Green range
        ],
        "color_min": 0.02,
        "min_area_px": 150,
        "max_area_frac": 0.40,
    },
    "pedestrian": { # Rules for a yellow diamond priority sign
        "aspect_ratio": 2.5,
        "aspect_tol": 5.0,
        "hsv_ranges": [
      # Yellow / Orange
        ],
        "color_min": 0.00,
        "min_area_px": 3500,
        "max_area_frac": 0.40,
    },
   
    
}


# ─── Filter Functions ─────────────────────────────────────────────────────────

def _check_aspect_ratio(bbox, expected_ratio, tolerance):
    """Check if bounding box aspect ratio (H/W) is within expected range.
    
    A stop sign is ~square (ratio ≈ 1.0).
    A car is wide and flat (ratio ≈ 0.3).
    This filter catches car-shaped false positives.
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    ratio = h / w

    low = expected_ratio - tolerance
    high = expected_ratio + tolerance

    return low <= ratio <= high


def _check_dominant_color(frame, bbox, hsv_ranges, min_fraction):
    """Check if enough pixels in the bounding box match the expected color.
    
    Converts the cropped region to HSV and counts pixels within any of the
    given HSV ranges. If fewer than min_fraction of pixels match, reject.
    
    HSV is used instead of RGB/hex because:
    - Hue encodes COLOR independently of brightness
    - A red sign in shadow and in sunlight both have Hue ≈ 0–10°
    - RGB would see them as completely different values
    """
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame.shape[:2]

    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]
    if total_pixels == 0:
        return False

    # Count pixels matching ANY of the HSV ranges
    combined_mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)
    for h_lo, s_lo, v_lo, h_hi, s_hi, v_hi in hsv_ranges:
        lower = np.array([h_lo, s_lo, v_lo])
        upper = np.array([h_hi, s_hi, v_hi])
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Denoise the combined mask to avoid speckle triggering the color test
    if combined_mask.size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.medianBlur(combined_mask, 5)

    matching_pixels = cv2.countNonZero(combined_mask)
    fraction = matching_pixels / total_pixels

    return fraction >= min_fraction



def _check_size(bbox, frame_shape, min_area_px, max_area_frac):
    """Check if bounding box area is within reasonable bounds.
    
    Rejects:
    - Tiny noise detections (< min_area_px pixels²)
    - Huge false positives (> max_area_frac of frame)
    """
    x1, y1, x2, y2 = bbox
    box_area = max((x2 - x1) * (y2 - y1), 0)
    frame_area = frame_shape[0] * frame_shape[1]

    if box_area < min_area_px:
        return False
    if frame_area > 0 and (box_area / frame_area) > max_area_frac:
        return False

    return True


# ─── Main Validation Function ─────────────────────────────────────────────────

def validate_detection(frame, sign_name, bbox, debug=False):
    """Validate a YOLO detection using shape, color, and size filters.
    
    Args:
        frame:     The original BGR image (numpy array).
        sign_name: Canonical BFMC sign name (e.g., "stop", "no_entry").
        bbox:      Tuple of (x1, y1, x2, y2) pixel coordinates.
        debug:     If True, print rejection reasons to console.
    
    Returns:
        True if the detection passes all filters (likely a real sign).
        False if any filter rejects it (likely a false positive).
    """
    rules = SIGN_FILTERS.get(sign_name)
    if rules is None:
        # Unknown sign type — no filter rules defined, let it pass
        return True

    # 1. Shape Filter (Aspect Ratio)
    if not _check_aspect_ratio(bbox, rules["aspect_ratio"], rules["aspect_tol"]):
        if debug:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            ratio = h / max(w, 1)
            print(f"  ❌ FILTER [{sign_name}] Shape rejected: "
                  f"ratio={ratio:.2f}, expected={rules['aspect_ratio']}±{rules['aspect_tol']}")
        return False

    # 2. Color Filter (Dominant Hue in HSV)
    if not _check_dominant_color(frame, bbox, rules["hsv_ranges"], rules["color_min"]):
        if debug:
            print(f"  ❌ FILTER [{sign_name}] Color rejected: "
                  f"expected color not found (< {rules['color_min']:.0%} of pixels)")
        return False

    # 3. Size Filter (Pixel Area)
    if not _check_size(bbox, frame.shape, rules["min_area_px"], rules["max_area_frac"]):
        if debug:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            print(f"  ❌ FILTER [{sign_name}] Size rejected: "
                  f"area={area}px², min={rules['min_area_px']}, "
                  f"max={rules['max_area_frac']:.0%} of frame")
        return False

    return True
