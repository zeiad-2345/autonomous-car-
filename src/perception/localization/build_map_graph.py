#!/usr/bin/env python3
import sys, cv2, json, math

class MapBuilder:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        if self.img is None:
            print(f"Error: Could not load image {image_path}")
            sys.exit(1)
        self.state = "SCALE_1"
        self.scale_p1 = self.scale_p2 = self.origin = None
        self.cm_per_px = 1.0
        self.waypoints = []
        self.current_type = None
        cv2.namedWindow("MapBuilder", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("MapBuilder", self.click)

    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == "SCALE_1":
                self.scale_p1 = (x, y); self.state = "SCALE_2"
            elif self.state == "SCALE_2":
                self.scale_p2 = (x, y); d_px = math.hypot(x-self.scale_p1[0], y-self.scale_p1[1])
                try:
                    real_cm = float(input(f"Distance in pixels: {d_px:.1f}. Enter real-world CM: "))
                    self.cm_per_px = real_cm / d_px; self.state = "ORIGIN"
                except ValueError: print("Invalid number. Try again.")
            elif self.state == "ORIGIN":
                self.origin = (x, y); self.state = "WAYPOINTS"
                print("Origin set. Now select sign type (c, s, r, etc.) and click to place waypoints.")
            elif self.state == "WAYPOINTS" and self.current_type:
                # Calculate coordinates relative to origin (X right, Y up in cm)
                dx = (x - self.origin[0]) * self.cm_per_px
                dy = (self.origin[1] - y) * self.cm_per_px
                self.waypoints.append({
                    "id": f"{self.current_type}_{len(self.waypoints)}",
                    "type": self.current_type,
                    "x_cm": round(dx, 2),
                    "y_cm": round(dy, 2)
                })
                print(f"Placed {self.current_type} at ({dx:.1f}, {dy:.1f})")
            self.draw()

    def draw(self):
        tmp = self.img.copy()
        msg = f"State: {self.state} | SignType: {self.current_type or 'NONE (Press keys)'}"
        cv2.putText(tmp, msg, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        if self.scale_p1: cv2.circle(tmp, self.scale_p1, 5, (255,0,0), -1)
        if self.scale_p2: cv2.circle(tmp, self.scale_p2, 5, (255,0,0), -1)
        if self.origin: cv2.circle(tmp, self.origin, 8, (0,255,0), -1)
        
        for wp in self.waypoints:
            px = int(self.origin[0] + wp["x_cm"] / self.cm_per_px)
            py = int(self.origin[1] - wp["y_cm"] / self.cm_per_px)
            cv2.circle(tmp, (px, py), 5, (0,165,255), -1)
            cv2.putText(tmp, wp["type"], (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

        cv2.imshow("MapBuilder", tmp)

    def run(self):
        print("--- Raven Map Builder ---")
        print("1. Click two points to define scale.")
        print("2. Click the Start Pose (Origin).")
        print("3. Press keys to select sign type, then click to place:")
        print("   'c'=crosswalk, 's'=stop, 'r'=roundabout, 'p'=parking, 'h'=highway, 'y'=priority")
        print("4. Press 'q' to save and quit.")
        
        while True:
            self.draw()
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('c'): self.current_type = "crosswalk"
            elif k == ord('s'): self.current_type = "stop"
            elif k == ord('r'): self.current_type = "roundabout"
            elif k == ord('p'): self.current_type = "parking"
            elif k == ord('h'): self.current_type = "highway_entrance"
            elif k == ord('x'): self.current_type = "highway_exit"
            elif k == ord('y'): self.current_type = "priority"
            elif k == ord('z') and self.waypoints: self.waypoints.pop(); print("Last waypoint removed.")

        with open("map_waypoints.json", "w") as f:
            json.dump(self.waypoints, f, indent=4)
        print(f"Saved {len(self.waypoints)} waypoints to map_waypoints.json")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 build_map_graph.py <image_path>")
    else:
        MapBuilder(sys.argv[1]).run()
