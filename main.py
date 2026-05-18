import sys
import os
import argparse
import time

# ── Dependency check ──────────────────────────────────────────────────────────
_REQUIRED = {
    "cv2":        "opencv-python",
    "numpy":      "numpy",
    "matplotlib": "matplotlib",
}
_missing = []
for _mod, _pkg in _REQUIRED.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print("\n[ERROR] Missing packages.  Run:\n")
    print(f"pip install {' '.join(_missing)}\n")
    sys.exit(1)

# ── Project imports ───────────────────────────────────────────────────────────
import config
from detector      import create_detectors
from cycle_manager import CycleManager
from simulation    import Dashboard
from utils.stats   import StatsLogger
from utils.grapher import generate


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AI 4-Way Traffic Signal Control — Adani University"
    )
    p.add_argument(
        "--scenario", "-s",
        choices=["equal","one_heavy","two_heavy","three_heavy","dynamic","rush_hour"],
        default=None,
        help="Demo traffic scenario  (default: config.DEMO_SCENARIO)"
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.scenario:
        config.DEMO_SCENARIO = args.scenario

    banner = "=" * 62
    print(banner)
    print("  AI-Powered 4-Way Traffic Signal Control System")
    print("  Author  : Ruchin Patel")
    print("  College : Adani University, Ahmedabad, India")
    print("  Degree  : B.Tech CSE, 2024-25")
    print(banner)
    print(f"  Scenario       : {config.DEMO_SCENARIO.upper()}")
    print(f"  Cycle Budget   : {config.CYCLE_BUDGET} s  |  Extension Pool : +{config.EXTENSION_POOL} s  |  Max : {config.CYCLE_BUDGET + config.EXTENSION_POOL} s")
    print(f"  Green Range    : {config.MIN_GREEN} s – {config.MAX_GREEN} s  |  Yellow : {config.YELLOW_TIME} s  |  All-Red : {config.ALL_RED_GAP} s")
    print(f"  Fallback       : {config.FALLBACK_GREEN} s fixed  (triggers after {config.FALLBACK_TRIGGER} all-zero cycles)")
    print(f"  Vehicle Weights: motorcycle={config.VEHICLE_WEIGHT.get(3,0.5)}  car={config.VEHICLE_WEIGHT.get(2,1.0)}  bus={config.VEHICLE_WEIGHT.get(5,2.5)}  truck={config.VEHICLE_WEIGHT.get(7,3.0)}")
    print(banner)
    print("  P = Pause   S = Switch Scenario   Q = Quit + Report")
    print(banner + "\n")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ── 1. Vehicle detectors ──────────────────────────────────────────────────
    print("[main] Creating vehicle detectors…")
    detectors = create_detectors()
    for d in detectors.values():
        d.start()
    time.sleep(0.8)   # let threads warm up

    # ── 2. Signal cycle manager ───────────────────────────────────────────────
    print("[main] Starting cycle manager…")
    manager = CycleManager(detectors)

    # ── 3. Statistics logger ──────────────────────────────────────────────────
    print("[main] Starting stats logger…")
    logger  = StatsLogger()
    manager.on_cycle_done = lambda s: logger.log(s)
    manager.start()

    # ── 4. OpenCV Dashboard (blocking) ────────────────────────────────────────
    print("[main] Opening simulation dashboard…")
    print("[main] Window is ready — use keyboard controls above.\n")

    dashboard = Dashboard(detectors, manager)
    try:
        dashboard.run(logger)
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")
    finally:
        print("\n[main] Shutting down…")
        manager.stop()
        for d in detectors.values():
            d.stop()

        csv_path = logger.close()

        print("[main] Generating performance chart…")
        if csv_path and os.path.exists(csv_path):
            chart = generate(csv_path)
            if chart:
                print(f"[main] Chart saved → {chart}")
        else:
            print("[main] No data to chart (session too short).")

        # ── Session summary ───────────────────────────────────────────────────
        s = manager.state_snapshot()
        a, f = s.avg_wait_adapt, s.avg_wait_fixed
        pct  = round((f - a) / f * 100, 1) if f > 0 else 0

        print()
        print("=" * 50)
        print("  SESSION SUMMARY")
        print("=" * 50)
        print(f"  Cycles completed   : {s.cycle_num}")
        print(f"  Weighted wait (AI)   : {a:.1f} s")
        print(f"  Weighted wait (Fixed): {f:.1f} s")
        print(f"  Wait improvement   : {pct:+.1f}%")
        print(f"  Carry-over total   : {s.total_carry:.0f} s")
        print(f"  Fuel saved         : {s.total_fuel_L:.3f} L")
        print(f"  CO\u2082 saved           : {s.total_co2_kg:.3f} kg")
        print("=" * 50)
        print()


if __name__ == "__main__":
    main()
