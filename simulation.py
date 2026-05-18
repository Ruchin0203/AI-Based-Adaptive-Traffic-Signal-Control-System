import cv2
import numpy as np
import time
import config
from utils.demo_traffic import SCENARIO_NAMES

_GRID    = {"NORTH": (0, 0), "EAST": (0, 1), "SOUTH": (1, 0), "WEST": (1, 1)}
_SIG_COL = {"GREEN": config.C_GREEN, "YELLOW": config.C_AMBER, "RED": config.C_RED}
_ROAD_COL = {
    "NORTH": (60,  60, 220),
    "EAST":  (200, 130, 30),
    "SOUTH": (60, 175,  60),
    "WEST":  (30, 170, 210),
}

class Dashboard:

    def __init__(self, detectors: dict, manager):
        self._det  = detectors
        self._mgr  = manager
        self._t0   = time.time()
        self._sc_i = 0

        FW, FH = config.FEED_W, config.FEED_H
        GAP    = 4
        self._cw = FW * 2 + GAP
        self._ch = FH * 2 + GAP + config.STATS_H + 36

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, logger):
        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_TITLE, self._cw, self._ch)

        last_frame_t = time.time()
        frame_dt     = 1.0 / config.FPS_CAP

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("p"), ord("P")):
                self._mgr.toggle_pause()
            elif key in (ord("s"), ord("S")):
                self._next_scenario()

            now = time.time()
            if now - last_frame_t >= frame_dt:
                state  = self._mgr.state_snapshot()
                canvas = self._render(state)
                cv2.imshow(config.WINDOW_TITLE, canvas)
                logger.log(state)
                last_frame_t = now

        cv2.destroyAllWindows()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self, state) -> np.ndarray:
        FW, FH = config.FEED_W, config.FEED_H
        GAP    = 4
        canvas = np.zeros((self._ch, self._cw, 3), dtype=np.uint8)
        canvas[:] = config.C_DARK

        for road, (row, col) in _GRID.items():
            x0 = col * (FW + GAP)
            y0 = row * (FH + GAP)
            frame = self._det[road].get_frame()
            if frame is None:
                frame = np.full((FH, FW, 3), 30, dtype=np.uint8)
            frame = cv2.resize(frame, (FW, FH))
            self._overlay_road(frame, road, state)
            canvas[y0:y0 + FH, x0:x0 + FW] = frame

        mid_x = FW + GAP // 2
        mid_y = FH + GAP // 2
        cv2.line(canvas, (mid_x, 0), (mid_x, FH * 2 + GAP), (45, 50, 60), GAP)
        cv2.line(canvas, (0, mid_y), (self._cw, mid_y), (45, 50, 60), GAP)

        bar_y = FH * 2 + GAP
        self._draw_phase_banner(canvas, state, bar_y)
        self._draw_stats(canvas, state, bar_y + 36)

        # Fallback mode warning banner
        if state.mode == "FALLBACK":
            ow = canvas.copy()
            cv2.rectangle(ow, (0, 0), (self._cw, bar_y), (0, 0, 80), -1)
            cv2.addWeighted(ow, 0.25, canvas, 0.75, 0, canvas)
            warn = "  ⚠  FALLBACK MODE — Fixed 30s Timer Active  ⚠  "
            (ww, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(canvas, warn,
                        ((self._cw - ww) // 2, bar_y // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.C_WARN, 2)

        if self._mgr.is_paused:
            mask = canvas.copy()
            cv2.rectangle(mask, (0, 0), (self._cw, bar_y), (0, 0, 0), -1)
            cv2.addWeighted(mask, 0.55, canvas, 0.45, 0, canvas)
            cv2.putText(canvas, "  PAUSED  —  Press P to resume  ",
                        (self._cw // 2 - 190, bar_y // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 240), 2)

        return canvas

    # ── Per-road feed overlay ─────────────────────────────────────────────────

    def _overlay_road(self, frame, road, state):
        FW, FH  = config.FEED_W, config.FEED_H
        phase   = state.phases.get(road, "RED")
        sig_col = _SIG_COL[phase]
        rc      = _ROAD_COL[road]
        # Use live_counts for display — updated every frame independently of phase logic
        count    = state.live_counts.get(road, state.counts.get(road, 0))
        weighted = state.live_weighted.get(road, state.weighted_counts.get(road, 0.0))
        cdown   = state.countdowns.get(road, 0)
        budget  = state.green_times.get(road, 0)

        # Top bar
        cv2.rectangle(frame, (0, 0), (FW, 28), (8, 10, 18), -1)
        cv2.rectangle(frame, (0, 0), (6, 28), rc, -1)
        cv2.putText(frame, road, (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, config.C_WHITE, 2)

        # Phase badge
        badge = phase
        (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        bx = FW - bw - 46
        cv2.rectangle(frame, (bx - 4, 4), (FW - 36, 25), sig_col, -1)
        cv2.putText(frame, badge, (bx, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, config.C_BLACK, 2)

        # Signal light
        cx, cy = FW - 18, 40
        cv2.circle(frame, (cx, cy), 14, (25, 27, 35), -1)
        cv2.circle(frame, (cx, cy), 10, sig_col, -1)
        cv2.circle(frame, (cx, cy), 14, (70, 75, 85), 1)
        glow = np.zeros_like(frame)
        cv2.circle(glow, (cx, cy), 16, sig_col, -1)
        cv2.addWeighted(frame, 1.0, glow, 0.20, 0, frame)

        # Countdown timer
        if phase in ("GREEN", "YELLOW") and cdown > 0:
            txt = f"{cdown}s"
            fs  = 1.5
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
            tx = (FW - tw) // 2
            ty = FH // 2 + th // 2 + 10
            cv2.putText(frame, txt, (tx + 2, ty + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3)
            cv2.putText(frame, txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, sig_col, 3)

        # Bottom bar — show raw + weighted counts
        cv2.rectangle(frame, (0, FH - 26), (FW, FH), (8, 10, 18), -1)
        cv2.putText(frame, f"Vehicles: {count}  (w:{weighted:.1f})",
                    (8, FH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, config.C_GOOD, 1)
        if budget > 0:
            bt = f"Green: {budget:.0f}s"
            (tw2, _), _ = cv2.getTextSize(bt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(frame, bt, (FW - tw2 - 8, FH - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.C_ACCENT, 1)

        # Vehicle count bar (left edge)
        bar_max = FH - 60
        bar_h   = int(min(weighted, 60) / 60 * bar_max)
        b_y0    = FH - 26 - bar_h
        barcol  = config.C_GREEN if weighted < 20 else (config.C_AMBER if weighted < 40 else config.C_RED)
        cv2.rectangle(frame, (0, b_y0), (5, FH - 26), barcol, -1)

    # ── Phase progress banner ─────────────────────────────────────────────────

    def _draw_phase_banner(self, canvas, state, y0):
        h = 36
        cv2.rectangle(canvas, (0, y0), (self._cw, y0 + h), (14, 18, 28), -1)
        cv2.line(canvas, (0, y0), (self._cw, y0), (55, 65, 90), 2)

        active = state.active_road
        sig    = state.phases.get(active, "RED")
        col    = _SIG_COL[sig]
        label  = state.phase_label
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        lx = (self._cw - lw) // 2
        cv2.putText(canvas, label, (lx, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)

        # Cycle + mode (left)
        mode_col = config.C_WARN if state.mode == "FALLBACK" else \
                   config.C_CYAN if state.mode == "AI+EXT" else config.C_LABEL
        cv2.putText(canvas, f"Cycle: {state.cycle_num}  [{state.mode}]",
                    (10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, mode_col, 1)

        # Cycle estimate + ext pool (right)
        right_txt = (f"~{state.cycle_time_estimate:.0f}s cycle  "
                     f"Ext:{state.extension_pool_remaining:.0f}s")
        (rw, _), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(canvas, right_txt, (self._cw - rw - 10, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.C_ACCENT, 1)

        self._draw_phase_bar(canvas, state, y0 + h - 6, self._cw - 20, 6)

    def _draw_phase_bar(self, canvas, state, y, width, height):
        total = sum(state.green_times.values()) or (config.MIN_GREEN * 4)
        x = 10
        for road in config.ROADS:
            g   = state.green_times.get(road, config.MIN_GREEN)
            seg = max(4, int(g / total * width))
            col = _SIG_COL[state.phases.get(road, "RED")]
            cv2.rectangle(canvas, (x, y), (x + seg - 2, y + height), col, -1)
            if seg > 22:
                cv2.putText(canvas, road[0], (x + 3, y + height - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
            x += seg

    # ── Statistics bar ────────────────────────────────────────────────────────

    def _draw_stats(self, canvas, state, y0):
        h = config.STATS_H
        cv2.rectangle(canvas, (0, y0), (self._cw, y0 + h), (10, 12, 20), -1)
        cv2.line(canvas, (0, y0), (self._cw, y0), (50, 60, 90), 2)

        a   = state.avg_wait_adapt
        f   = state.avg_wait_fixed
        pct = (f - a) / f * 100 if f > 0 else 0
        elapsed = int(time.time() - self._t0)
        hh, mm, ss = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60

        items = [
            ("Wait Save",    f"{pct:+.1f}%"),
            ("Fuel Saved",   f"{state.total_fuel_L:.2f}L"),
            ("CO₂ Saved",  f"{state.total_co2_kg:.2f}kg"),
            ("AI Wait",      f"{a:.1f}s"),
            ("Fixed Wait",   f"{f:.1f}s"),
            ("Cycle",        f"{state.cycle_time_estimate:.0f}s"),
            ("Scenario",     config.DEMO_SCENARIO.upper()),
            ("Session",      f"{hh:02d}:{mm:02d}:{ss:02d}"),
            ("Keys",         "P=Pause S=Scene Q=Quit"),
        ]

        sx        = 6
        col_width = (self._cw - 12) // len(items)
        for label, val in items:
            cv2.putText(canvas, label, (sx, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, config.C_LABEL, 1)
            vc = config.C_GOOD if label == "Wait Reduction" else \
                 config.C_WARN if (label == "Cycle Time" and
                                   state.cycle_time_estimate > config.CYCLE_BUDGET) \
                 else config.C_ACCENT
            cv2.putText(canvas, val, (sx, y0 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, vc, 1)
            sx += col_width

    # ── Scenario cycling ──────────────────────────────────────────────────────

    def _next_scenario(self):
        self._sc_i = (self._sc_i + 1) % len(SCENARIO_NAMES)
        new = SCENARIO_NAMES[self._sc_i]
        config.DEMO_SCENARIO = new
        for d in self._det.values():
            if hasattr(d, "_demo") and d._demo is not None:
                d._demo.set_scenario(new)
        print(f"[simulation] Scenario → {new.upper()}")