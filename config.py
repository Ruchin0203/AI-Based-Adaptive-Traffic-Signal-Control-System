# ── Intersection ──────────────────────────────────────────────────────────────
ROADS = ["NORTH", "EAST", "SOUTH", "WEST"]   # Fixed phase order

# ── Timing (seconds) ─────────────────────────────────────────────────────────
CYCLE_BUDGET         = 120   # Total green-time budget per cycle across all 4 phases
MIN_GREEN            = 10    # Safety floor — every road gets at least this
MAX_GREEN            = 120   # Hard ceiling per phase (tier 7 top) — no single road hogs the cycle
YELLOW_TIME          = 3     # Amber warning before each RED
ALL_RED_GAP          = 1     # All-red clearance between consecutive phases
MAX_WAIT_CYCLES      = 3     # Starvation guard: force green if road waits this long

# ── Extension Pool ────────────────────────────────────────────────────────────
# Extra seconds available beyond CYCLE_BUDGET when traffic is unusually heavy.
# Keeps cycle max at CYCLE_BUDGET + EXTENSION_POOL = 170 s max.
EXTENSION_POOL       = 50    # Bonus seconds available per cycle for overflow
EXTENSION_MIN_CLAIM  = 5     # Minimum seconds worth claiming from pool
# A road claims from the pool if its weighted count exceeds this threshold
EXTENSION_THRESHOLD  = 25    # Weighted vehicle count above which extension kicks in

# u2500u2500 Live display refresh u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500
LIVE_REFRESH_INTERVAL = 2.0  # Seconds between dashboard vehicle count updates

# ── Consecutive-Zero Protection ───────────────────────────────────────────────
# If a road reports 0 vehicles for this many consecutive cycles → give it
# ZERO_GREEN seconds only, freeing budget for busier roads.
ZERO_GREEN           = 5     # Green time for confirmed-empty roads
ZERO_CYCLE_TRIGGER   = 3     # Consecutive zero cycles before road is treated as empty

# ── Fallback Fixed Timer (runs if AI/detector fails) ─────────────────────────
FALLBACK_GREEN       = 30    # Fixed green per road in fallback mode
FALLBACK_TRIGGER     = 3     # Consecutive all-zero cycles before fallback activates

# ── Vehicle Type Weighting ────────────────────────────────────────────────────
# YOLO COCO class id → (weight, label)
# Heavier / longer vehicles need more clearance time → higher weight.
VEHICLE_WEIGHT = {
    3:  0.5,   # motorcycle  — small, quick to clear
    2:  1.0,   # car         — baseline
    5:  2.5,   # bus         — long, slow to clear
    7:  3.0,   # truck       — heaviest
}
# Fallback weight for any class not listed above
VEHICLE_WEIGHT_DEFAULT = 1.0

# ── Detection ────────────────────────────────────────────────────────────────
DETECTION_INTERVAL    = 15    # Run YOLO every N frames  (2 Hz at 30 FPS)
CONFIDENCE_THRESHOLD  = 0.40  # Minimum YOLO confidence to count a vehicle
VEHICLE_CLASSES       = list(VEHICLE_WEIGHT.keys())   # COCO: car, motorcycle, bus, truck

# ── Video / Camera Sources ────────────────────────────────────────────────────
VIDEO_SOURCES = {
    "NORTH": "demo",
    "EAST":  "demo",
    "SOUTH": "demo",
    "WEST":  "demo",
}

# ── Demo Scenario ─────────────────────────────────────────────────────────────
DEMO_SCENARIO = "one_heavy"

# ── Window / Display ──────────────────────────────────────────────────────────
WINDOW_TITLE = "AI Traffic Signal Control"
FEED_W       = 470
FEED_H       = 270
STATS_H      = 75
FPS_CAP      = 30

# ── OpenCV colours (BGR) ──────────────────────────────────────────────────────
C_GREEN  = (50,  210,  50)
C_AMBER  = (0,   165, 255)
C_RED    = (45,   45, 215)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,     0,   0)
C_DARK   = (18,   20,  28)
C_PANEL  = (28,   32,  42)
C_LABEL  = (170, 170, 170)
C_ACCENT = (255, 175,  45)
C_GOOD   = (75,  215, 115)
C_CYAN   = (0,   220, 200)
C_WARN   = (0,   165, 255)   # orange — used for fallback mode warning

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_EVERY_N_SECS = 5
RESULTS_DIR      = "results"

# ── Fuel / Emissions estimation ───────────────────────────────────────────────
IDLE_FUEL_L_PER_HR = 0.60
CO2_KG_PER_LITRE   = 2.35
