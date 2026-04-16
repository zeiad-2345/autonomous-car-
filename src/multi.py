from __future__ import annotations

import argparse
import threading
import time
import sys
from pathlib import Path
from typing import Optional

import cv2

# Ensure imports work when running: python src\multi
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]   # ...\drivex-brain-stack
_SRC_ROOT = _THIS.parent         # ...\drivex-brain-stack\src
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.perception.sign_recognition.live_sign_detector import (  # noqa: E402
    _extract_detections,
    draw_detections,
    draw_hud,
    get_camera,
    LiveSignDetectorThread,
)

# Lane detection (optional — may not be available yet)
try:
    from src.perception.lane_detection.threadLaneDetection import (
        LaneDetectionThread, draw_lanes, draw_roi_overlay, draw_lane_hud,
    )
    LANE_AVAILABLE = True
except ImportError:
    try:
        from perception.lane_detection.threadLaneDetection import (
            LaneDetectionThread, draw_lanes, draw_roi_overlay, draw_lane_hud,
        )
        LANE_AVAILABLE = True
    except ImportError:
        LANE_AVAILABLE = False
        
from src.shared_state import SharedState



def _draw_lane_info(frame, lane_result):
    """Overlay lane error, heading, and type on the frame."""
    if lane_result is None:
        cv2.putText(frame, "Lane: no data", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return frame
    error = lane_result.get("error", 0.0)
    heading = lane_result.get("heading", 0.0)
    lane_type = lane_result.get("lane_type", "unknown")
    cv2.putText(frame, f"Lane err: {error:+.2f}  head: {heading:+.3f}  [{lane_type}]",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def run_live(model_path: str, conf: float, webcam: bool, source: Optional[str], add_traffic_box: bool, state: Optional[SharedState] = None, remote: bool = False) -> None:
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Model not found: {model_path}")
        return

    # --- Shared state for lane detection thread ---
    if state is None:
        state = SharedState()

    # --- Sign detection thread ---
    detector = LiveSignDetectorThread(
        state,
        model_path=str(model_file),
        conf=conf,
        add_traffic_box=True,
    )
    detector.start()

    # --- Lane detection thread ---
    lane_thread = None
    if LANE_AVAILABLE:
        # LaneDetectionThread expects (state, args) — we pass a simple namespace
        lane_args = argparse.Namespace()
        lane_thread = LaneDetectionThread(state, lane_args)
        lane_thread.start()
        print("[multi] Lane detection started.")
    else:
        print("[multi] Lane detection not available.")

    cap = None
    using_video_file = False

    if remote:
        print("[multi] Remote mode — reading frames from SharedState (TCP bridge)")
    elif source:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Cannot open source: {source}")
            state.shutdown()
            return
        using_video_file = True
        # Print video info
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[multi] Video: {w}x{h}, {total} frames")
    else:
        cap = get_camera(use_webcam=webcam)
        using_video_file = False

    # Target processing resolution (resize large frames for speed)
    MAX_WIDTH = 640

    frame_id = 0
    fps_t0 = time.time()
    fps_count = 0
    fps = 0.0

    # Compute per-frame delay to match original video speed (1.5x playback)
    if using_video_file:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30
        wait_ms = max(1, int(1000 / (video_fps * 1.5)))
    else:
        wait_ms = 1

    _last_frame_id = None   # track frame identity to avoid re-processing stale frames

    curve_start_time = None
    is_in_curve_display = False

    try:
        while True:
            if cap is not None:
                # Local mode: read from camera / video file
                ok, frame = cap.read()
                if not ok:
                    if using_video_file:
                        break
                    continue

                # Resize large frames for speed
                h, w = frame.shape[:2]
                if w > MAX_WIDTH:
                    scale = MAX_WIDTH / w
                    frame = cv2.resize(frame, (MAX_WIDTH, int(h * scale)))

                # Share frame for both threads via SharedState
                state.set_latest_frame(frame.copy())
            else:
                # Remote mode: frames arrive via SharedState from TCP bridge
                if not state.is_running():
                    break
                frame = state.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                # Skip if same frame object (no new frame yet)
                if frame is _last_frame_id:
                    time.sleep(0.005)
                    continue
                _last_frame_id = frame

            frame_id += 1

            shown = frame

            # Draw sign detections
            sign_result = state.get_signs()
            detections_count = 0
            if sign_result is not None:
                shown = draw_detections(shown, sign_result["detections"])
                detections_count = len(sign_result["detections"])

            # Draw lane visuals from thread result (no expensive processing here)
            if LANE_AVAILABLE:
                lane_result = state.get_lane()
                if lane_result is not None:
                    left_fit = lane_result.get("left_fit")
                    right_fit = lane_result.get("right_fit")
                    Minv = lane_result.get("Minv")
                    # runtime debug: print presence of fits/Minv
                  ##  print(f"[multi] lane fits - left:{left_fit is not None}, right:{right_fit is not None}, Minv:{Minv is not None}")
                    # small on-frame label for quick visual feedback
                    status_text = f"L:{int(left_fit is not None)} R:{int(right_fit is not None)}"
                    cv2.putText(shown, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    shown = draw_lanes(shown, left_fit, right_fit, Minv)
                    shown = draw_roi_overlay(shown)
                
#                
                    binary_view = lane_result.get("binary")
                    if binary_view is not None:
                       cv2.imshow("Lane Binary View", binary_view)
                shown = draw_lane_hud(shown, lane_result)

                # ── CURVE VS STRAIGHT LOGIC (Window HUD) ──
                if lane_result is not None:
                    hud_error = lane_result.get("error", 0.0)
                    if abs(hud_error) > 0.55  :
                        if curve_start_time is None:
                            curve_start_time = time.time()
                        elif time.time() - curve_start_time >= .30:
                            is_in_curve_display = True
                    else:
                        curve_start_time = None
                        is_in_curve_display = False

                    # Send the timer-confirmed curve state to SharedState for skynet to use
                    state.set_curve_mode(is_in_curve_display)

                    if is_in_curve_display:
                        cv2.putText(shown, f"IN CURVE (Err: {hud_error:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(shown, f"STRAIGHT (Err: {hud_error:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # ──────────────────────────────────────────

            fps_count += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                fps = fps_count / dt
                fps_count = 0
                fps_t0 = time.time()

            shown = draw_hud(shown, fps, "threaded-sign+lane", detections_count)
            cv2.imshow("RAVEN Perception (Signs + Lane)", shown)

            if (cv2.waitKey(wait_ms) & 0xFF) == ord("q"):
                break
    finally:
        state.shutdown()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Threaded sign detector runner")
    parser.add_argument(
        "--model",
        type=str,
        default="src/perception/sign_recognition/bfmc_best_shirts.pt",
        help="Path to YOLO .pt model",
    )
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--source", type=str, default=None)
   

    args = parser.parse_args()

    run_live(
        model_path="src/perception/sign_recognition/bfmc_best_shirts.pt",
        conf=args.conf,
        webcam=args.webcam,
        source=args.source,
        add_traffic_box=True,
    )


if __name__ == "__main__":
    main()