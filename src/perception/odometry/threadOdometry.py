#!/usr/bin/env python3
"""
RAVEN — Odometry Thread
========================
Fuses IMU heading (yaw from `@imu`) with commanded speed to produce a
continuous (x, y, heading) position estimate via dead-reckoning.

No wheel encoder is currently published by the Arduino firmware (verified
by reading `robotstatemachine.cpp` — it publishes @speed, @steer, @brake but
NOT wheel ticks). Dead-reckoning therefore uses the *commanded* speed as a
velocity estimate. This introduces cumulative drift; sign-detection events
in `threadLocalization.py` will correct it once the map is loaded.

When the Arduino firmware is updated to publish @encoder:POS;; messages, swap
in the encoder-based distance in the `_integrate()` method below.

Track constraints used for sanity-clamping:
  Max speed city:    20 cm/s
  Max speed highway: 40 cm/s

Publishes to SharedState:
  state.pose  — {"x_cm": float, "y_cm": float, "heading_rad": float, "dist_cm": float}

Usage in skynet.py:
    odom = OdometryThread(state)
    odom.start()
"""

import threading
import time
import math


ODOM_HZ = 20   # 20 Hz integration loop — matches SerialThread update rate


class OdometryThread(threading.Thread):
    """
    Integrates speed and IMU yaw into a (x, y, heading) pose estimate.
    Thread-safe: reads from SharedState, writes to SharedState.
    """

    def __init__(self, state):
        super().__init__(name="OdometryThread", daemon=True)
        self.state = state
        self._interval = 1.0 / ODOM_HZ

        # Current pose (cm, radians)
        self._x   = 0.0
        self._y   = 0.0
        self._hdg = 0.0     # radians — 0 = forward (north on map)
        self._dist = 0.0    # total distance travelled (cm)

        self._last_t = None

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        print("[Odometry] Starting…")
        self._last_t = time.time()

        while self.state.is_running():
            now = time.time()
            dt = now - self._last_t
            if dt < self._interval:
                time.sleep(0.002)
                continue
            self._last_t = now

            self._integrate(dt)

        print("[Odometry] Stopped.")

    # ── Integration ───────────────────────────────────────────────────────

    def _integrate(self, dt: float):
        """
        Single Euler integration step.

        Heading source priority:
          1. IMU yaw (most reliable — from Arduino @imu:roll;pitch;yaw;ax;ay;az;;)
          2. Fallback to commanded steer angle if IMU not yet available

        Speed source:
          Commanded speed_cmd from SharedState.
          Replace with encoder-derived speed once firmware publishes @encoder.
        """
        # ── Heading from IMU ──────────────────────────────────────────────
        imu = self.state.imu_data
        if imu and "yaw" in imu:
            try:
                # IMU yaw is in degrees from the Arduino — convert to radians
                yaw_deg = float(imu["yaw"])
                self._hdg = math.radians(yaw_deg)
            except (ValueError, TypeError):
                pass   # Keep last known heading
        else:
            # Fallback: integrate commanded steer angle
            # Steer range ±25° maps to turn rate via vehicle kinematics
            # Approximate wheelbase: ~20 cm (BFMC car)
            speed_cmd, steer_cmd = self.state.get_command()
            WHEELBASE_CM = 20.0
            if abs(speed_cmd) > 0.01:
                steer_rad = math.radians(steer_cmd)
                turn_rate = (speed_cmd * math.tan(steer_rad)) / WHEELBASE_CM
                self._hdg += turn_rate * dt
                self._hdg  = _normalise_angle(self._hdg)

        # ── Speed / distance ──────────────────────────────────────────────
        speed_cmd, _ = self.state.get_command()

        # TODO: replace with encoder-derived speed when Arduino publishes @encoder
        # encoder = self.state.encoder_data
        # if encoder and "position" in encoder:
        #     speed_cm_s = _encoder_to_speed(encoder)  # implement once firmware ready
        # else:
        speed_cm_s = float(speed_cmd)   # treat commanded speed as actual (no slip model)

        # Sanity clamp — never integrate faster than physical max
        speed_cm_s = max(-40.0, min(40.0, speed_cm_s))

        d_cm = speed_cm_s * dt

        self._x    += d_cm * math.cos(self._hdg)
        self._y    += d_cm * math.sin(self._hdg)
        self._dist += abs(d_cm)

        # Publish
        self.state.set_pose({
            "x_cm":       round(self._x,   2),
            "y_cm":       round(self._y,   2),
            "heading_rad": round(self._hdg, 4),
            "dist_cm":    round(self._dist, 2),
        })

    def reset(self):
        """Reset pose to origin — call at start line."""
        with threading.Lock():
            self._x = self._y = self._dist = 0.0
            self._hdg = 0.0
            print("[Odometry] Pose reset to origin.")


# ── Helpers ───────────────────────────────────────────────────────────────

def _normalise_angle(a: float) -> float:
    """Wrap angle to (-π, π]."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a
