# =============================================================================
#  utils/grapher.py  —  6-Panel Performance Chart Generator
#  Reads the CSV log and saves a PNG report.
#  Author : Ruchin Patel  |  Adani University  |  B.Tech CSE 2024-25
# =============================================================================

import os, csv
from datetime import datetime
import config

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False


def generate(csv_path: str) -> str | None:
    if not _MPL:
        print("[grapher] matplotlib not found — skipping chart.")
        return None

    rows = []
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"[grapher] Cannot read CSV: {e}")
        return None

    if len(rows) < 2:
        print("[grapher] Not enough data rows.")
        return None

    def F(key):  return [float(r[key]) for r in rows]
    def I(key):  return [int(r[key])   for r in rows]

    cycles   = I("cycle")
    cnt_N    = I("count_N"); cnt_E = I("count_E")
    cnt_S    = I("count_S"); cnt_W = I("count_W")
    grn_N    = F("green_N"); grn_E = F("green_E")
    grn_S    = F("green_S"); grn_W = F("green_W")
    fuel     = F("fuel_saved_L")
    co2      = F("co2_saved_kg")
    wa       = F("avg_wait_adaptive_s")
    wf       = F("avg_wait_fixed_s")
    ext_pool = F("ext_pool_remaining")

    FIXED = 30.0
    RC   = {"NORTH": "#E53935", "EAST": "#1E88E5", "SOUTH": "#43A047", "WEST": "#FB8C00"}

    fig = plt.figure(figsize=(18, 11), facecolor="#F8F9FC")
    fig.suptitle(
        "AI-Powered 4-Way Traffic Signal Control  —  Performance Report\n"
        "Ruchin Patel  |  Adani University  |  B.Tech CSE 2025–26",
        fontsize=13, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    def ax_style(ax, title, xlabel="Cycle", ylabel=""):
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35, color="#CCCCCC")
        ax.set_title(title, fontweight="bold", fontsize=10, pad=8)
        ax.set_xlabel(xlabel, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)

    # ── 1: Vehicle counts ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for road, data, col in [("North",cnt_N,RC["NORTH"]),("East",cnt_E,RC["EAST"]),
                              ("South",cnt_S,RC["SOUTH"]),("West",cnt_W,RC["WEST"])]:
        ax.plot(cycles, data, color=col, lw=1.6, label=road)
    ax.legend(fontsize=8, loc="upper right")
    ax_style(ax, "Vehicle Counts per Approach", ylabel="Vehicles")

    # ── 2: Adaptive vs Fixed green time ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for road, data, col in [("N adaptive",grn_N,RC["NORTH"]),("E adaptive",grn_E,RC["EAST"]),
                              ("S adaptive",grn_S,RC["SOUTH"]),("W adaptive",grn_W,RC["WEST"])]:
        ax.plot(cycles, data, color=col, lw=1.6, label=road)
    ax.axhline(FIXED, color="#888888", lw=1.2, linestyle="--", label="Fixed 30s")
    ax.legend(fontsize=7, loc="upper right")
    ax_style(ax, "Adaptive vs Fixed Green Time", ylabel="Seconds")

    # ── 3: Cumulative fuel saved ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.fill_between(cycles, fuel, alpha=0.25, color="#2E7D32")
    ax.plot(cycles, fuel, color="#2E7D32", lw=2)
    ax_style(ax, "Cumulative Fuel Saved (L)", ylabel="Litres")

    # ── 4: Wait time comparison ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(cycles, wa, color="#1565C0", lw=2, label="Adaptive")
    ax.plot(cycles, wf, color="#B71C1C", lw=1.5, linestyle="--", label="Fixed 30s")
    ax.fill_between(cycles, wf, wa, alpha=0.12, color="#1565C0")
    ax.legend(fontsize=9)
    ax_style(ax, "Avg Wait Time per Cycle (s)", ylabel="Seconds")

    # ── 5: Extension pool remaining ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(cycles, ext_pool, alpha=0.25, color="#6A1B9A")
    ax.plot(cycles, ext_pool, color="#6A1B9A", lw=2)
    ax_style(ax, "Extension Pool Remaining (s)", ylabel="Seconds")

    # ── 6: Session summary bars ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    last   = rows[-1]
    av     = float(last["avg_wait_adaptive_s"])
    fv     = float(last["avg_wait_fixed_s"])
    fl     = float(last["fuel_saved_L"])
    co2v   = float(last["co2_saved_kg"])
    pct    = last["improvement_pct"]

    labels = ["Wait\nFixed(s)", "Wait\nAdapt(s)", "Fuel\nSaved(L)", "CO₂\nSaved(kg)"]
    vals   = [fv, av, fl, co2v]
    cols   = ["#B71C1C","#1565C0","#2E7D32","#00838F"]
    bars   = ax.bar(labels, vals, color=cols, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_style(ax, f"Session Summary  ({pct} improvement)", xlabel="")

    # ── Save ──────────────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(config.RESULTS_DIR, f"results_{ts}.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[grapher] Chart → {out_path}")
    return out_path
