# =============================================================================
#  utils/stats.py  —  CSV Statistics Logger
#  Author : Ruchin Patel  |  Adani University  |  B.Tech CSE 2024-25
# =============================================================================

import csv
import os
import time
from datetime import datetime
import config

HEADERS = [
    "timestamp", "cycle", "mode",
    "count_N", "count_E", "count_S", "count_W",
    "weighted_N", "weighted_E", "weighted_S", "weighted_W",
    "green_N", "green_E", "green_S", "green_W",
    "cycle_time_s", "ext_pool_remaining",
    "active_road", "phase_label",
    "fuel_saved_L", "co2_saved_kg",
    "avg_wait_adaptive_s", "avg_wait_fixed_s", "improvement_pct",
]


class StatsLogger:

    def __init__(self):
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(config.RESULTS_DIR, f"stats_{ts}.csv")
        self._fp  = open(self.path, "w", newline="")
        self._w   = csv.DictWriter(self._fp, fieldnames=HEADERS)
        self._w.writeheader()
        self._fp.flush()
        self._last = 0.0
        print(f"[stats] Logging → {self.path}")

    def log(self, state):
        now = time.time()
        if now - self._last < config.LOG_EVERY_N_SECS:
            return
        self._last = now

        a   = state.avg_wait_adapt
        f   = state.avg_wait_fixed
        pct = round((f - a) / f * 100, 1) if f > 0 else 0.0

        wc = getattr(state, "weighted_counts", {})

        self._w.writerow({
            "timestamp":           datetime.now().strftime("%H:%M:%S"),
            "cycle":               state.cycle_num,
            "mode":                getattr(state, "mode", "AI"),
            "count_N":             state.counts.get("NORTH", 0),
            "count_E":             state.counts.get("EAST",  0),
            "count_S":             state.counts.get("SOUTH", 0),
            "count_W":             state.counts.get("WEST",  0),
            "weighted_N":          f"{wc.get('NORTH', 0):.1f}",
            "weighted_E":          f"{wc.get('EAST',  0):.1f}",
            "weighted_S":          f"{wc.get('SOUTH', 0):.1f}",
            "weighted_W":          f"{wc.get('WEST',  0):.1f}",
            "green_N":             f"{state.green_times.get('NORTH', 0):.1f}",
            "green_E":             f"{state.green_times.get('EAST',  0):.1f}",
            "green_S":             f"{state.green_times.get('SOUTH', 0):.1f}",
            "green_W":             f"{state.green_times.get('WEST',  0):.1f}",
            "cycle_time_s":        f"{getattr(state, 'cycle_time_estimate', 0):.1f}",
            "ext_pool_remaining":  f"{getattr(state, 'extension_pool_remaining', 0):.1f}",
            "active_road":         state.active_road,
            "phase_label":         state.phase_label,
            "fuel_saved_L":        f"{state.total_fuel_L:.3f}",
            "co2_saved_kg":        f"{state.total_co2_kg:.3f}",
            "avg_wait_adaptive_s": f"{a:.2f}",
            "avg_wait_fixed_s":    f"{f:.2f}",
            "improvement_pct":     f"{pct}%",
        })
        self._fp.flush()

    def close(self) -> str:
        self._fp.close()
        print(f"[stats] Saved → {self.path}")
        return self.path
