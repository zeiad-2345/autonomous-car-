import math
#####################################
# ─────────────────────────────────────────────
#  Tunable Parameters
# ─────────────────────────────────────────────

# Stanley gains
STANLEY_K   = 4.0    # cross-track error gain
STANLEY_KS  = 0.5    # softening constant

# Adaptive PID gains on top of Stanley
KP_BOOST    = 3.0    # extra proportional gain scaled by error magnitude
KI          = 0.8    # integral gain (corrects steady drift)
KD          = 0.3    # derivative gain on heading change rate
I_MAX       = 40.0   # anti-windup clamp for integral term (degrees)

# Adaptive thresholds
SMALL_ERROR_CM = 3.0   # below this → gentle mode
LARGE_ERROR_CM = 8.0   # above this → aggressive mode

# Speed control
BASE_SPEED        = 15.0
SPEED_REDUCTION_K = 0.4
MIN_FORWARD_SPEED = 8.0

# Output limits
MAX_SPEED  =  50.0
MIN_SPEED  = -50.0
MAX_STEER  = 120.0
MIN_STEER  = -120.0

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
#  Adaptive Stanley + PID Controller
# ─────────────────────────────────────────────

# Persistent state for integral and derivative
_integral = 0.0
_prev_heading = 0.0

def adaptive_stanley_pid(lateral_error_cm: float,
                         heading_error_deg: float,
                         speed_cm_s: float) -> float:
    global _integral, _prev_heading

    # --- Stanley base ---
    heading_rad      = deg2rad(heading_error_deg)
    cross_track_term = math.atan2(STANLEY_K * lateral_error_cm,
                                  STANLEY_KS + abs(speed_cm_s))
    stanley_steer = rad2deg(heading_rad + cross_track_term)

    # --- Adaptive gain multiplier based on error magnitude ---
    abs_err = abs(lateral_error_cm)
    if abs_err < SMALL_ERROR_CM:
        adapt = 0.5   # gentle on straights
    elif abs_err > LARGE_ERROR_CM:
        adapt = 2.0   # aggressive on curves
    else:
        # linear interpolation between gentle and aggressive
        adapt = 0.5 + 1.5 * (abs_err - SMALL_ERROR_CM) / (LARGE_ERROR_CM - SMALL_ERROR_CM)

    # --- PID additions ---
    # P boost: extra proportional kick scaled adaptively
    p_term = KP_BOOST * adapt * lateral_error_cm

    # I term: accumulate error, clamp to prevent windup
    _integral += lateral_error_cm * 0.05  # dt ≈ 50ms (20Hz planner)
    _integral = clamp(_integral, -I_MAX / KI, I_MAX / KI)
    i_term = KI * _integral

    # D term: rate of heading change (damping)
    d_term = KD * (heading_error_deg - _prev_heading) / 0.05
    _prev_heading = heading_error_deg

    return stanley_steer + p_term + i_term + d_term


# ─────────────────────────────────────────────
#  Speed Controller
# ─────────────────────────────────────────────

def compute_speed(steering_deg: float) -> float:
    steer_fraction = abs(steering_deg) / MAX_STEER
    speed = BASE_SPEED * (1.0 - SPEED_REDUCTION_K * steer_fraction)
    speed = max(speed, MIN_FORWARD_SPEED)
    return speed

# ─────────────────────────────────────────────
#  Main Controller  (public API)
# ─────────────────────────────────────────────

def lane_controller(lateral_error_cm: float,
                    heading_error_deg: float) -> dict:

    raw_steer = adaptive_stanley_pid(lateral_error_cm,
                                     heading_error_deg,
                                     BASE_SPEED)

    steering_angle = clamp(raw_steer, MIN_STEER, MAX_STEER)
    speed = compute_speed(steering_angle)
    speed = clamp(speed, MIN_SPEED, MAX_SPEED)

    return round(speed, 0), round(steering_angle, 0), lateral_error_cm, heading_error_deg

#####################################

#####################################
def Control_Code(distance_cm, angle_deg):

    speed, steer, lateral_error_cm, heading_error_deg = lane_controller(distance_cm, angle_deg)

    return speed, steer
#####################################
