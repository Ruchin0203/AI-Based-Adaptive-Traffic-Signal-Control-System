import threading
import time
import copy
import config


_TIERS = [
    (0,    10,  10),  # Tier 0: 0 vehicles        → MIN_GREEN floor (safety)
    (5,    12,  18),  # Tier 1: 1–5   weighted     → 12–18s  (~2.4s/vehicle)
    (15,   19,  30),  # Tier 2: 6–15  weighted     → 19–30s  (queue building)
    (25,   31,  45),  # Tier 3: 16–25 weighted     → 31–45s  (moderate load)
    (40,   46,  60),  # Tier 4: 26–40 weighted     → 46–60s  (heavy load)
    (60,   61,  80),  # Tier 5: 41–60 weighted     → 61–80s  (dense traffic)
    (90,   81, 110),  # Tier 6: 61–90 weighted     → 81–110s (peak congestion)
    (999, 111, 120),  # Tier 7: 91+   weighted     → 111–120s (max; ext pool handles rest)
]


def _tier_lookup(weighted_count: float):
    """Return (base_s, top_s, frac) for a given weighted vehicle count."""
    prev = 0
    for upper, base, top in _TIERS:
        if weighted_count <= upper:
            span = upper - prev
            frac = (weighted_count - prev) / span if span > 0 else 0.0
            return float(base), float(top), frac
        prev = upper
    # Beyond last tier
    _, base, top = _TIERS[-1]
    return float(base), float(top), 1.0


def tier_green_time(weighted_count: float) -> float:
    """Interpolated green time for a given weighted vehicle count."""
    base, top, frac = _tier_lookup(weighted_count)
    return base + frac * (top - base)


# ── State ─────────────────────────────────────────────────────────────────────

class State:
    """Complete snapshot of signal system state — safe to copy and read."""
    __slots__ = [
        "phases", "countdowns", "counts", "weighted_counts", "green_times",
        "live_counts", "live_weighted",  # refreshed every frame for display only
        "active_road", "cycle_num", "phase_label",
        "mode",                          # "AI" | "AI+EXT" | "FALLBACK"
        "extension_pool_remaining",
        "total_carry", "total_fuel_L", "total_co2_kg",
        "avg_wait_adapt", "avg_wait_fixed",
        "cycle_time_estimate",
    ]

    def __init__(self):
        self.phases                  = {r: "RED" for r in config.ROADS}
        self.countdowns              = {r: 0     for r in config.ROADS}
        self.counts                  = {r: 0     for r in config.ROADS}
        self.weighted_counts         = {r: 0.0   for r in config.ROADS}
        self.live_counts             = {r: 0     for r in config.ROADS}
        self.live_weighted           = {r: 0.0   for r in config.ROADS}
        self.green_times             = {r: 0.0   for r in config.ROADS}
        self.active_road             = config.ROADS[0]
        self.cycle_num               = 0
        self.phase_label             = "STARTING…"
        self.mode                    = "AI"
        self.extension_pool_remaining = float(config.EXTENSION_POOL)
        self.total_carry             = 0.0
        self.total_fuel_L            = 0.0
        self.total_co2_kg            = 0.0
        self.avg_wait_adapt          = 0.0
        self.avg_wait_fixed          = float(config.FALLBACK_GREEN * len(config.ROADS))
        self.cycle_time_estimate     = 0.0


# ── Controller ────────────────────────────────────────────────────────────────

class CycleManager:
    """
    Runs the four-phase N→E→S→W signal cycle on a daemon thread.
    Dashboard reads state via state_snapshot() without blocking the cycle.
    """

    def __init__(self, detectors: dict):
        self._detectors = detectors
        self._state     = State()
        self._lock      = threading.Lock()
        self._running   = False
        self._paused    = False
        self._thread    = None

        # Anti-starvation: how many cycles each road has been skipped
        self._waited    = {r: 0 for r in config.ROADS}

        # Consecutive-zero tracker per road
        self._zero_streak = {r: 0 for r in config.ROADS}

        # Fallback: consecutive all-zero cycles counter
        self._all_zero_cycles = 0

        # Accumulators
        self._cycle_n     = 0
        self._wait_sum_a  = 0.0
        self._fuel_total  = 0.0
        self._carry_total = 0.0

        self.on_cycle_done = None   # optional callback(State) after each full cycle

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        # Signal cycle thread
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        # Live display-count refresh thread (reads detectors every frame, ~30fps)
        self._live_thread = threading.Thread(target=self._live_refresh_loop, daemon=True)
        self._live_thread.start()

    def stop(self):
        self._running = False

    def toggle_pause(self):
        self._paused = not self._paused

    @property
    def is_paused(self):
        return self._paused

    def state_snapshot(self) -> State:
        with self._lock:
            return copy.copy(self._state)

    # ── Live display refresh (runs independently at ~30fps) ───────────────────

    def _live_refresh_loop(self):
        """
        Reads every detector continuously at ~30fps and writes into
        state.live_counts / state.live_weighted.
        These fields are ONLY for the dashboard display — the signal
        timing logic always samples fresh counts itself just before
        each phase and never reads from here.
        """
        interval = config.LIVE_REFRESH_INTERVAL
        while self._running:
            live_c = {}
            live_w = {}
            for road in config.ROADS:
                try:
                    det = self._detectors[road]
                    if hasattr(det, "get_weighted_count"):
                        raw, w = det.get_weighted_count()
                    else:
                        raw = det.get_count()
                        w   = float(raw)
                    live_c[road] = int(raw)
                    live_w[road] = float(w)
                except Exception:
                    live_c[road] = 0
                    live_w[road] = 0.0
            with self._lock:
                self._state.live_counts   = live_c
                self._state.live_weighted = live_w
            time.sleep(interval)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _main_loop(self):
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            try:
                self._run_one_cycle()
            except Exception as exc:
                # Never let an exception kill the signal controller
                print(f"[CycleManager] ERROR in cycle {self._cycle_n}: {exc}")
                time.sleep(1.0)

    # ── One full N→E→S→W cycle ────────────────────────────────────────────────

    def _run_one_cycle(self):
        self._cycle_n += 1

        # Reset per-cycle extension pool
        ext_pool_remaining = float(config.EXTENSION_POOL)

        # Detect if we should enter fallback mode
        fallback = self._check_fallback()

        # Track mode for display
        cycle_mode = "FALLBACK" if fallback else "AI"

        # Per-cycle budget tracker — deducted as each phase is assigned
        budget_remaining = float(config.CYCLE_BUDGET)

        # We'll store green times as they are decided (fresh per phase)
        green_decided   = {}
        counts_decided  = {}
        weighted_decided = {}


        hard_cap = float(config.CYCLE_BUDGET + config.EXTENSION_POOL)
        g_min    = float(config.MIN_GREEN)

        if not fallback:
            # Step 1: pre-compute budget shares from cycle-start snapshot
            snap_raw = {}; snap_weighted = {}; snap_tier = {}
            for r in config.ROADS:
                rw, ww        = self._sample_road(r)
                snap_raw[r]   = rw
                snap_weighted[r] = ww
                snap_tier[r]  = tier_green_time(ww)

            total_snap = sum(snap_tier.values()) or 1.0
            budget_share = {
                r: max(g_min, snap_tier[r] / total_snap * hard_cap)
                for r in config.ROADS
            }
            # Mark as extended if total snap exceeds normal budget
            if total_snap > float(config.CYCLE_BUDGET):
                cycle_mode = "AI+EXT"

        for phase_idx, road in enumerate(config.ROADS):
            if not self._running:
                return

            roads_after = config.ROADS[phase_idx + 1:]

            if fallback:
                raw_count = 0
                weighted  = 0.0
                g_time    = float(config.FALLBACK_GREEN)
            else:
                # Step 2: re-sample THIS road fresh — its queue grew while waiting
                raw_count, weighted = self._sample_road(road)

                # Consecutive-zero shortcut
                if (raw_count == 0 and
                        self._zero_streak.get(road, 0) >= config.ZERO_CYCLE_TRIGGER):
                    g_time = float(config.ZERO_GREEN)
                else:
                    fresh_tier = tier_green_time(weighted)
                    share_cap  = budget_share[road]
                    # Use fresh need but never exceed the pre-allocated share cap
                    g_time = min(fresh_tier, share_cap)
                    g_time = max(g_time, g_min)
                    # Starvation guard
                    if self._waited.get(road, 0) >= config.MAX_WAIT_CYCLES:
                        g_time = max(g_time, g_min)

            used_ext = False

            # Record decisions
            green_decided[road]    = g_time
            counts_decided[road]   = raw_count
            weighted_decided[road] = weighted

            # Update starvation counter
            self._waited[road] = 0

            # Update zero-streak
            if raw_count == 0:
                self._zero_streak[road] = self._zero_streak.get(road, 0) + 1
            else:
                self._zero_streak[road] = 0

            # Update shared state so dashboard shows current info
            with self._lock:
                self._state.counts          = dict(counts_decided)
                self._state.weighted_counts = dict(weighted_decided)
                self._state.green_times     = dict(green_decided)
                self._state.cycle_num       = self._cycle_n
                self._state.mode            = cycle_mode
                self._state.extension_pool_remaining = ext_pool_remaining

            # ── GREEN phase ───────────────────────────────────────────────────
            self._set_phase(road, "GREEN", g_time)
            self._tick(road, g_time)
            if not self._running:
                return

            # ── YELLOW phase ──────────────────────────────────────────────────
            self._set_phase(road, "YELLOW", config.YELLOW_TIME)
            self._tick(road, config.YELLOW_TIME)
            if not self._running:
                return

            # ── ALL-RED clearance ─────────────────────────────────────────────
            with self._lock:
                self._state.phases      = {r: "RED" for r in config.ROADS}
                self._state.phase_label = "ALL RED — clearance"
                self._state.countdowns  = {r: 0 for r in config.ROADS}
            self._sleep(config.ALL_RED_GAP)

            # Increment starvation for roads not yet served this cycle
            for r in roads_after:
                self._waited[r] = self._waited.get(r, 0) + 1

        # ── Check all-zero fallback trigger ───────────────────────────────────
        all_zero = all(counts_decided.get(r, 0) == 0 for r in config.ROADS)
        if all_zero:
            self._all_zero_cycles += 1
        else:
            self._all_zero_cycles = 0

        # ── End-of-cycle statistics ───────────────────────────────────────────
        total_green = sum(green_decided.values())
        n_roads     = len(config.ROADS)

        # ── Weighted wait metric ──────────────────────────────────────────────
        # Each vehicle on road R waits for the sum of the OTHER roads' green times.
        # We weight by vehicle count so 50 vehicles on North matter 50× more than
        # 1 vehicle on West. This makes the improvement % genuinely meaningful.
        #
        #   total_delay_AI    = Σ  counts[r] × Σ green[x]  (x ≠ r)
        #   total_delay_fixed = Σ  counts[r] × (n-1) × FALLBACK_GREEN
        #   weighted_wait_AI  = total_delay_AI    / total_vehicles
        #   weighted_wait_fix = total_delay_fixed / total_vehicles  (= 90s always)
        #
        total_vehicles = max(1, sum(counts_decided.values()))

        total_delay_ai = 0.0
        for r in config.ROADS:
            wait_r = sum(green_decided.get(x, 0) for x in config.ROADS if x != r)
            total_delay_ai += counts_decided.get(r, 0) * wait_r

        total_delay_fixed = sum(
            counts_decided.get(r, 0) * float(config.FALLBACK_GREEN) * (n_roads - 1)
            for r in config.ROADS
        )

        weighted_wait_ai    = total_delay_ai    / total_vehicles
        weighted_wait_fixed = total_delay_fixed / total_vehicles

        # Running average across all cycles
        self._wait_sum_a += weighted_wait_ai
        avg_a = self._wait_sum_a / self._cycle_n

        fuel  = self._est_fuel(counts_decided, green_decided)
        self._fuel_total += fuel
        co2   = self._fuel_total * config.CO2_KG_PER_LITRE

        with self._lock:
            self._state.total_fuel_L          = self._fuel_total
            self._state.total_co2_kg          = co2
            self._state.avg_wait_adapt        = avg_a
            self._state.avg_wait_fixed        = weighted_wait_fixed
            self._state.cycle_time_estimate   = total_green
            self._state.mode                  = cycle_mode

        if self.on_cycle_done:
            self.on_cycle_done(self.state_snapshot())

    # ── Green time decision ───────────────────────────────────────────────────

    def _decide_green(
        self, road: str, raw_count: int, weighted: float,
        budget_remaining: float, ext_pool: float
    ) -> tuple:
        """
        Returns (green_seconds, used_extension: bool, extension_seconds: float).
        Each road gets exactly its tier-table time — never inflated to fill budget.
        """
        g_min = float(config.MIN_GREEN)
        g_max = float(config.MAX_GREEN)

        # ── Consecutive-zero shortcut ─────────────────────────────────────────
        if (raw_count == 0 and
                self._zero_streak.get(road, 0) >= config.ZERO_CYCLE_TRIGGER):
            g_time = float(config.ZERO_GREEN)
            return max(g_time, g_min), False, 0.0

        # ── Starvation override ───────────────────────────────────────────────
        starvation = self._waited.get(road, 0) >= config.MAX_WAIT_CYCLES

        # ── Tier-based time ───────────────────────────────────────────────────
        # Use exactly what the tier table says — no surplus redistribution.
        # Only clamp downward if budget is genuinely exhausted (roads ahead
        # still need their MIN_GREEN safety floor reserved).
        tier_time = tier_green_time(weighted)

        # Use tier time directly — clamp only if budget is truly exhausted.
        # No min_reserve pressure: each road uses exactly what its traffic needs.
        # The g_min floor on every road is the only safety guarantee needed.
        g_time = min(tier_time, budget_remaining, g_max)
        g_time = max(g_time, g_min)

        # Starvation: force floor only, never inflate
        if starvation:
            g_time = max(g_time, g_min)

        # ── Extension pool: claim if heavily clamped and traffic is heavy ─────
        used_ext   = False
        ext_used   = 0.0
        shortfall  = tier_time - g_time   # how much we were forced to cut

        if (shortfall >= config.EXTENSION_MIN_CLAIM and
                weighted > config.EXTENSION_THRESHOLD and
                ext_pool >= config.EXTENSION_MIN_CLAIM):
            claim    = min(shortfall, ext_pool, g_max - g_time)
            claim    = max(claim, 0.0)
            g_time  += claim
            ext_used = claim
            used_ext = claim > 0

        return round(g_time, 1), used_ext, ext_used

    # ── Road sampling ─────────────────────────────────────────────────────────

    def _sample_road(self, road: str) -> tuple:
        """
        Returns (raw_count: int, weighted_count: float).
        Tries get_weighted_count() first (YOLO mode), falls back to get_count().
        """
        det = self._detectors[road]
        try:
            if hasattr(det, "get_weighted_count"):
                raw, weighted = det.get_weighted_count()
                return int(raw), float(weighted)
            else:
                raw = int(det.get_count())
                return raw, float(raw)   # demo: treat all vehicles as cars
        except Exception as exc:
            print(f"[CycleManager] Detector error on {road}: {exc}")
            return 0, 0.0

    # ── Fallback check ────────────────────────────────────────────────────────

    def _check_fallback(self) -> bool:
        """Return True if system should use fallback fixed-time mode."""
        return self._all_zero_cycles >= config.FALLBACK_TRIGGER

    # ── Phase helpers ─────────────────────────────────────────────────────────

    def _set_phase(self, road: str, phase: str, countdown: float):
        with self._lock:
            self._state.phases           = {r: "RED" for r in config.ROADS}
            self._state.phases[road]     = phase
            self._state.active_road      = road
            self._state.phase_label      = f"{road}  {phase}  ({countdown:.0f}s)"
            self._state.countdowns       = {r: 0 for r in config.ROADS}
            self._state.countdowns[road] = int(countdown)
        # Notify detectors of current phases so demo animation freezes on RED
        for r in config.ROADS:
            det = self._detectors.get(r)
            if det and hasattr(det, "set_phase"):
                det.set_phase(phase if r == road else "RED")

    def _tick(self, road: str, duration: float):
        """Block for `duration` seconds, updating countdown every 50 ms."""
        end_t = time.time() + duration
        while time.time() < end_t and self._running:
            if self._paused:
                end_t += 0.05
            remaining = max(0, end_t - time.time())
            with self._lock:
                self._state.countdowns[road] = int(remaining) + 1
            time.sleep(0.05)

    def _sleep(self, secs: float):
        end_t = time.time() + secs
        while time.time() < end_t and self._running:
            if self._paused:
                end_t += 0.05
            time.sleep(0.05)

    # ── Fuel estimation ───────────────────────────────────────────────────────

    def _est_fuel(self, counts: dict, green: dict) -> float:
        """
        Estimate fuel saved this cycle vs fixed-timer baseline.
        A vehicle idles while waiting for the other 3 roads' green phases.
        Fixed baseline: each road waits 3 × FALLBACK_GREEN.
        Adaptive: each road waits sum of the other 3 roads' actual green times.
        """
        fuel = 0.0
        for r in config.ROADS:
            others     = [x for x in config.ROADS if x != r]
            wait_fixed = (float(config.FALLBACK_GREEN) * len(others)
                          + len(others) * config.YELLOW_TIME
                          + len(others) * config.ALL_RED_GAP)
            wait_adapt = (sum(green.get(x, 0) for x in others)
                          + len(others) * config.YELLOW_TIME
                          + len(others) * config.ALL_RED_GAP)
            idle_saved = max(0.0, wait_fixed - wait_adapt)
            fuel      += counts.get(r, 0) * idle_saved * config.IDLE_FUEL_L_PER_HR / 3600.0
        return fuel
