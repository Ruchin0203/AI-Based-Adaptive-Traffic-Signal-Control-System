# =============================================================================
#  utils/demo_traffic.py  —  Synthetic Four-Way Traffic Generator
#  Produces animated road frames + realistic vehicle counts.
#  Works with NO camera and NO video file.
#  Author : Ruchin Patel  |  Adani University  |  B.Tech CSE 2024-25
# =============================================================================

import time
import math
import random
import numpy as np
import cv2
import config


# ── Per-scenario count functions ──────────────────────────────────────────────
def _noise(sigma=2):
    return int(random.gauss(0, sigma))

SCENARIOS = {
    "equal": {
        "NORTH": lambda t: max(0, 25 + _noise(3)),
        "EAST":  lambda t: max(0, 25 + _noise(3)),
        "SOUTH": lambda t: max(0, 25 + _noise(3)),
        "WEST":  lambda t: max(0, 25 + _noise(3)),
    },
    "one_heavy": {
        "NORTH": lambda t: max(0, 55 + _noise(4)),
        "EAST":  lambda t: max(0,  5 + _noise(2)),
        "SOUTH": lambda t: max(0,  5 + _noise(2)),
        "WEST":  lambda t: max(0,  4 + _noise(2)),
    },
    "two_heavy": {
        "NORTH": lambda t: max(0, 45 + _noise(4)),
        "EAST":  lambda t: max(0,  5 + _noise(2)),
        "SOUTH": lambda t: max(0, 44 + _noise(4)),
        "WEST":  lambda t: max(0,  5 + _noise(2)),
    },
    "three_heavy": {
        "NORTH": lambda t: max(0, 40 + _noise(4)),
        "EAST":  lambda t: max(0, 35 + _noise(3)),
        "SOUTH": lambda t: max(0, 38 + _noise(4)),
        "WEST":  lambda t: max(0,  4 + _noise(2)),
    },
    "dynamic": {
        "NORTH": lambda t: max(0, int(30 + 25 * math.sin(t / 20.0) + _noise(3))),
        "EAST":  lambda t: max(0, int(30 + 25 * math.cos(t / 20.0) + _noise(3))),
        "SOUTH": lambda t: max(0, int(30 - 20 * math.sin(t / 20.0) + _noise(3))),
        "WEST":  lambda t: max(0, int(20 + 15 * math.sin(t / 15.0 + 1) + _noise(2))),
    },
    "rush_hour": {
        "NORTH": lambda t: max(0, int(55 / (1 + math.exp(-(t % 120 - 40) / 8)) + 5 + _noise(3))),
        "EAST":  lambda t: max(0,  8 + _noise(3)),
        "SOUTH": lambda t: max(0, int(10 + 40 * (t % 120) / 120 + _noise(3))),
        "WEST":  lambda t: max(0,  6 + _noise(2)),
    },
}

SCENARIO_NAMES = list(SCENARIOS.keys())


# ── Vehicle type definitions ──────────────────────────────────────────────────
_VTYPES = [
    # (bgr_body,  bgr_roof,  w,  h, label)
    ((130, 110, 180), (160, 140, 210), 28, 16, "car"),
    ((70,  130, 200), (90,  160, 230), 26, 15, "car"),
    ((55,  165, 215), (75,  195, 245), 24, 14, "car"),
    ((160,  85, 130), (190, 115, 160), 26, 15, "car"),
    ((90,  140,  95), (120, 170, 125), 24, 15, "car"),
    ((145, 100,  75), (175, 130, 105), 42, 20, "truck"),
    ((80,  145, 100), (110, 175, 130), 38, 22, "bus"),
    ((165, 110,  85), (195, 140, 115), 44, 21, "truck"),
    ((155, 120,  90), (16,   16,  16), 20, 13, "motorcycle"),
    ((200, 165,  55), (16,   16,  16), 18, 12, "motorcycle"),
]


class FourWayDemoSource:
    """
    Generates animated synthetic road frames and per-road vehicle counts
    for one intersection approach.  No camera or file needed.
    """

    def __init__(self, road_name: str):
        self.road  = road_name
        self._t0   = time.time()
        self._seed = abs(hash(road_name)) % 9999
        random.seed(self._seed)
        self._phase      = "RED"          # current signal phase for animation
        self._frozen_t   = 0.0            # time reference when road went RED

    def set_scenario(self, scenario: str):
        config.DEMO_SCENARIO = scenario

    def set_phase(self, phase: str):
        """Called by detector loop to inform demo source of current signal phase."""        
        self._phase = phase

    # ── Public ──────────────────────────────────────────────────────────────

    def get_frame_and_count(self):
        """Returns (frame: np.ndarray [H,W,3 BGR],  count: int)"""
        t     = time.time() - self._t0
        count = self._get_count(t)
        phase = getattr(self, '_phase', 'RED')
        frame = self._render(count, t, phase)
        return frame, count

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_count(self, t: float) -> int:
        fn = SCENARIOS.get(config.DEMO_SCENARIO, SCENARIOS["equal"]).get(self.road)
        return max(0, min(int(fn(t)), 70)) if fn else 0

    def _render(self, count: int, t: float, phase: str = "RED") -> np.ndarray:
        W, H = config.FEED_W, config.FEED_H
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # ── Road surface ─────────────────────────────────────────────────────
        frame[:] = (54, 57, 62)                        # tarmac
        frame[:, :22]   = (82, 85, 88)                 # left kerb
        frame[:, W-22:] = (82, 85, 88)                 # right kerb

        # Dashed centre-line markings  (2 lane dividers)
        for lane_y in [H // 3, 2 * H // 3]:
            x = 24
            while x < W - 24:
                cv2.rectangle(frame, (x, lane_y - 2), (min(x + 26, W - 24), lane_y + 2),
                              (205, 175, 42), -1)
                x += 40

        # Stop line at bottom
        cv2.rectangle(frame, (22, H - 32), (W - 22, H - 28), (240, 240, 240), -1)

        # ── Animated vehicles ─────────────────────────────────────────────────
        rng   = np.random.default_rng(self._seed + int(t * 3))
        shown = min(count, 16)

        for i in range(shown):
            vt       = _VTYPES[i % len(_VTYPES)]
            body_col, roof_col, vw, vh, vname = vt

            # Two-lane distribution
            lane_y = (H // 4) + (i % 2) * (H // 2)
            base_x = 26 + int((i // 2) * (W - 56) / max(shown // 2, 1))
            # Vehicles move when GREEN/YELLOW, freeze when RED
            if phase in ("GREEN", "YELLOW"):
                scroll = int((t * 20 + i * 43) % (W - vw - 50))
            else:
                # Freeze at a deterministic position — vehicles queued at stop line
                scroll = int((i * 43) % (W - vw - 50))
            vx = 26 + scroll % (W - vw - 50)
            vy     = lane_y - vh // 2

            # Body
            cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), body_col, -1)
            cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (220, 220, 220), 1)

            # Roof / cab (lighter)
            rh = max(4, vh // 2 - 2)
            cv2.rectangle(frame, (vx + 3, vy + 2), (vx + vw - 3, vy + rh), roof_col, -1)

            # Headlights
            cv2.circle(frame, (vx + 4,      vy + vh - 3), 2, (240, 240, 100), -1)
            cv2.circle(frame, (vx + vw - 4, vy + vh - 3), 2, (240, 240, 100), -1)

            # YOLO-style detection box on first 5 vehicles
            if i < 5:
                conf  = round(0.88 + i * 0.02, 2)
                label = f"{vname} {conf}"
                cv2.rectangle(frame, (vx - 1, vy - 1), (vx + vw + 1, vy + vh + 1),
                              (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                cv2.rectangle(frame, (vx, vy - th - 6), (vx + tw + 4, vy),
                              (0, 185, 0), -1)
                cv2.putText(frame, label, (vx + 2, vy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)

        # ── HUD overlays ──────────────────────────────────────────────────────
        # Top bar
        cv2.rectangle(frame, (0, 0), (W, 22), (10, 12, 22), -1)
        cv2.putText(frame,
                    f"YOLOv8-nano  |  {self.road}  |  Detected: {count}  [DEMO]",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (95, 205, 100), 1)

        # Bottom count bar
        cv2.rectangle(frame, (0, H - 24), (W, H), (10, 12, 22), -1)
        cv2.putText(frame, f"Vehicles: {count}",
                    (8, H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.46, config.C_GOOD, 1)

        return frame
