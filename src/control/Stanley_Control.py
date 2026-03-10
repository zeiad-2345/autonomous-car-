import math
#####################################
# ─────────────────────────────────────────────
#  Tunable Parameters
# ─────────────────────────────────────────────

# Stanley gains
STANLEY_K   = 1.2    # cross-track error gain — higher = more aggressive lateral correction
STANLEY_KS  = 2.0    # softening constant     — prevents division by zero at very low speed

# Speed control
BASE_SPEED        = 30.0   # cm/s nominal forward speed
SPEED_REDUCTION_K = 0.7    # how aggressively steering reduces speed (0 = no reduction, 1 = full)
MIN_FORWARD_SPEED = 10.0   # cm/s — don't slow below this while going forward

# Output limits
MAX_SPEED  =  50.0   # cm/s
MIN_SPEED  = -50.0   # cm/s
MAX_STEER  =  25.0   # degrees
MIN_STEER  = -25.0   # degrees

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


# ─────────────────────────────────────────────
#  Stanley Steering
# ─────────────────────────────────────────────

def stanley_steering(lateral_error_cm: float,
                     heading_error_deg: float,
                     speed_cm_s: float) -> float:

    heading_rad      = deg2rad(heading_error_deg)
    cross_track_term = math.atan2(STANLEY_K * lateral_error_cm,
                                  STANLEY_KS + abs(speed_cm_s))

    steer_rad = heading_rad + cross_track_term
    return rad2deg(steer_rad)

# ─────────────────────────────────────────────
#  Speed Controller
# ─────────────────────────────────────────────

def compute_speed(steering_deg: float) -> float:
    steer_fraction = abs(steering_deg) / MAX_STEER          # 0.0 → 1.0
    speed = BASE_SPEED * (1.0 - SPEED_REDUCTION_K * steer_fraction)
    speed = max(speed, MIN_FORWARD_SPEED)                   # floor for forward motion
    return speed

# ─────────────────────────────────────────────
#  Main Controller  (public API)
# ─────────────────────────────────────────────

def lane_controller(lateral_error_cm: float,
                    heading_error_deg: float) -> dict:

    # Sign convention: left drift/angle = positive input from CV.
    # Stanley internally uses right = positive, so we negate both inputs.
    raw_steer = stanley_steering(-lateral_error_cm,
                                 -heading_error_deg,
                                 BASE_SPEED)

    # Step 2 — clamp steering before feeding into speed controller
    steering_angle = clamp(raw_steer, MIN_STEER, MAX_STEER)

    # Step 3 — adapt speed based on how hard we're steering
    speed = compute_speed(steering_angle)
    speed = clamp(speed, MIN_SPEED, MAX_SPEED)

    return round(speed, 0), round(steering_angle, 0), lateral_error_cm, heading_error_deg

#####################################

#####################################
def Control_Code(distance_cm, angle_deg):

    speed, steer, lateral_error_cm, heading_error_deg = lane_controller(distance_cm, angle_deg)

    return speed, steer
#####################################
