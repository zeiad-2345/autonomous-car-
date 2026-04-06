"""lane_detection package init

Expose a small, stable API and prefer the threaded implementation
(`threadLaneDetection`) so runtime uses the edited code during display.

This module also provides a `trapezoid_vertices` compatibility helper
so code importing that symbol from the package keeps working.
"""

from .threadLaneDetection import (
    LaneDetectionThread,
    draw_roi_overlay,
    draw_dashed_line,
    draw_lanes,
    IPM_SRC,
    ROI_TOP_FRACTION,
)

import numpy as _np

__all__ = [
    "LaneDetectionThread",
    "draw_roi_overlay",
    "draw_dashed_line",
    "draw_lanes",
    "trapezoid_vertices",
]


def trapezoid_vertices(img):
    """Return polygon vertices compatible with the older v11 API.

    Uses `IPM_SRC` (fractions) from the threaded implementation so the
    same ROI definition is used everywhere.
    """
    h, w = img.shape[:2]
    pts = (IPM_SRC * _np.float32([w, h])).astype(_np.int32)
    # reshape to the [ [ (x1,y1), ... ] ] format callers expect
    return pts.reshape(1, -1, 2)
