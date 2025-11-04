# ======================== Camera CONSTANTS ========================
# UI
SHOW_TRAIL            = True
TRAIL_LEN             = 20
DRAW_EVENT_LIFETIME_S = 1.6

# --- Mouse hover overlay ---
HOVER_ENABLED = True  # press 'h' to toggle

# Snapshots: ONLY for misses
SAVE_SNAPSHOTS_MISS   = True

# Tracking / Timing
INBOUND_TIMEOUT_S   = 30.0
STATE_RETENTION_S   = 25.0
REID_MAX_GAP_S      = 8.0
REID_DIST_PX        = 300
REID_IOU_THRESH     = 0.01
MAX_AGE_FRAMES      = 120

# Detection
SCORE_THR           = 0.35

# Anti-double-count (RED)
RED_DEBOUNCE_S         = 0.6
REARM_DIST_FROM_RED_PX = 35
MIN_V_SHIFT_PX         = 2

# Cross confirmation
CONSEC_NEEDED_ON_RED  = 1
CONSEC_NEEDED_ON_DEST = 1

# Global dedupe window along red
GLOBAL_TIME_WIN  = 0.30
GLOBAL_X_TOL     = 18

# Hot re-ID guard (to keep PID consistent right after IN)
HOT_REID_DIST_PX      = 140
HOT_REID_IOU          = 0.02
HOT_REID_TIMELOCK_S   = 2.0

# Direction gating
DIR_MIN_TRAVEL_PX     = 30

# Cooldown for 'counted' state to prevent immediate re-counting
COUNTED_COOLDOWN_SECONDS = 5.0
# Distance threshold (pixels) to consider a reappearance near the last counted position
REAPPEAR_DISTANCE_THRESHOLD = 100

# Cooldown for 'counted' state to prevent immediate re-counting
COUNTED_COOLDOWN_SECONDS = 5.0 # (ปรับเวลาได้ตามต้องการ)
REAPPEAR_DISTANCE_THRESHOLD = 150 # (ปรับระยะห่างได้ตามต้องการ)