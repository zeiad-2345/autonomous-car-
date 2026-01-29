import cv2
import time
import os
import argparse
import numpy as np

# Resolution matches threadCamera.py configuration
RESOLUTION = (640, 480) 

def collect_images(output_dir="calibration/images", interval=1.0):
    """
    Captures images from the camera for calibration.
    
    Args:
        output_dir (str): Directory to save images.
        interval (float): Minimum time between captures (if auto-capture was implemented, 
                          but here we use manual trigger).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    camera = None
    cap = None
    
    # Try initializing Picamera2 (RPi)
    try:
        import picamera2
        camera = picamera2.Picamera2()
        config = camera.create_preview_configuration(
            main={"format": "RGB888", "size": RESOLUTION},
            lores={"size": (320, 240)},
            encode="lores",
        )
        camera.configure(config)
        camera.start()
        print("Picamera2 started.")
    except (ImportError, RuntimeError) as e:
        print(f"Picamera2 not available ({e}). Falling back to standard OpenCV VideoCapture.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any camera.")
            return

    count = 0
    print(f"Press 's' to save an image. Press 'q' to quit.")
    print(f"Images will be saved to {output_dir}")

    try:
        while True:
            frame_bgr = None
            
            if camera:
                try:
                    frame = camera.capture_array("main")
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Capture error: {e}")
                    time.sleep(0.1)
                    continue
            elif cap:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("Failed to read frame.")
                    break

            if frame_bgr is None:
                continue

            cv2.imshow("Calibration Capture", frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = os.path.join(output_dir, f"calib_{count:03d}.jpg")
                cv2.imwrite(filename, frame_bgr)
                print(f"Saved {filename}")
                count += 1
                time.sleep(0.2) # Debounce
                
    except KeyboardInterrupt:
        pass
    finally:
        if camera:
            camera.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect calibration images.")
    parser.add_argument("--dir", type=str, default="calibration/images", help="Output directory")
    args = parser.parse_args()
    
    collect_images(output_dir=args.dir)
