"""
make_charts.py
Draws the charts for your README into a "charts" folder.

Run after build_deliveries.py (and after train_model.py, for the last chart):
    python -m pip install matplotlib      (once)
    python make_charts.py

Colour rule used in the first three charts: BLUE = above the overall late rate,
GREY = at or below it. One accent colour, so the eye goes straight to the problem.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw straight to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "charts"
OUT.mkdir(exist_ok=True)

# ---------- Look ----------
SURFACE = "#fcfcfb"   # chart background (also keeps it readable on GitHub dark mode)
INK = "#0b0b0b"       # titles and values
INK_2 = "#52514e"     # subtitles and category labels
MUTED = "#898781"     # axis numbers
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"    # blue
QUIET = "#c3c2b7"     # grey for "everything else"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

# ---------- Data ----------
df = pd.read_csv(
    BASE / "deliveries.csv",
    parse_dates=["order_purchase_timestamp", "promised_date"],
)
overall = df["is_late"].mean()

df["customer_state"] = df["route"].str.split("->").str[1]
df["month"] = df["order_purchase_timestamp"].dt.to_period("M")
df["promised_days"] = (df["promised_date"] - df["order_purchase_timestamp"]).dt.days
df["promised_band"] = pd.cut(
    df["promised_days"],
    bins=[-1, 10, 20, 30, 1000],
    labels=["0-10 days", "11-20 days", "21-30 days", "31+ days"],
)


def rate_table(column, min_orders):
    """Orders and late rate per group. Small groups are hidden: their rates are noise."""
    t = df.groupby(column, observed=True)["is_late"].agg(orders="count", rate="mean")
    return t[t["orders"] >= min_orders]


def bar_colours(rates):
    return [ACCENT if r > overall else QUIET for r in rates]


def titles(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=30)
    # Subtitle sits a fixed 7 points above the plot, whatever the chart height.
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
                textcoords="offset points", fontsize=9.5, color=INK_2,
                ha="left", va="bottom")


def label_overall_line(ax):
    """Name the overall-rate line, just outside the right edge so it never hits a bar."""
    ax.annotate(f"All orders: {overall:.1%}", xy=(1, overall),
                xycoords=("axes fraction", "data"), xytext=(8, 0),
                textcoords="offset points", fontsize=9, color=INK_2,
                ha="left", va="center")


def style(ax, grid_axis):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved {path.name}")


# ---------- Chart 1: late rate by destination state ----------
t = rate_table("customer_state", 300).sort_values("rate")  # ascending: worst ends up on top
n = len(t)
fig, ax = plt.subplots(figsize=(8, 0.3 * n + 1.8))
ax.barh(t.index, t["rate"], height=0.62, color=bar_colours(t["rate"]))
ax.axvline(overall, color=INK_2, linewidth=1.2)
ax.text(overall + 0.003, 0, f"All orders: {overall:.1%}", fontsize=9, color=INK_2, va="center")
for i, r in enumerate(t["rate"]):
    if r > overall:
        ax.text(r + 0.003, i, f"{r:.1%}", fontsize=9, color=INK, va="center")
ax.set_xlim(0, t["rate"].max() * 1.12)
ax.set_ylim(-0.7, n - 0.3)
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.tick_params(axis="y", labelcolor=INK_2, labelsize=9.5)
style(ax, "x")
titles(ax, "Late rate by destination state",
       "Share of delivered orders that arrived after the promised date. "
       "Blue = above the overall rate. States with 300+ orders.")
save(fig, "late_by_state.png")

# ---------- Chart 2: late rate by month ----------
t = rate_table("month", 500).sort_index()
labels = [p.strftime("%b %y") for p in t.index]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
xs = range(len(t))
ax.bar(xs, t["rate"], width=0.62, color=bar_colours(t["rate"]))
ax.axhline(overall, color=INK_2, linewidth=1.2)
label_overall_line(ax)
for x, r in zip(xs, t["rate"]):
    if r > 0.10:  # label only the clear spikes
        ax.text(x, r + 0.003, f"{r:.1%}", fontsize=9, color=INK, ha="center", va="bottom")
ax.set_xticks(list(xs))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
ax.set_xlim(-0.7, len(t) - 0.3)
ax.set_ylim(0, t["rate"].max() * 1.15)
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
style(ax, "y")
titles(ax, "Late rate by month of purchase",
       "Share of delivered orders that arrived late. Blue = above the overall rate. "
       "Months with 500+ orders.")
save(fig, "late_by_month.png")

# ---------- Chart 3: late rate by promised delivery window ----------
t = rate_table("promised_band", 1).sort_index()
labels = [f"{band}\n{orders:,} orders" for band, orders in zip(t.index, t["orders"])]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
xs = range(len(t))
ax.bar(xs, t["rate"], width=0.55, color=bar_colours(t["rate"]))
ax.axhline(overall, color=INK_2, linewidth=1.2)
label_overall_line(ax)
for x, r in zip(xs, t["rate"]):
    ax.text(x, r + 0.002, f"{r:.1%}", fontsize=9.5, color=INK, ha="center", va="bottom")
ax.set_xticks(list(xs))
ax.set_xticklabels(labels, fontsize=9, color=INK_2)
ax.set_xlim(-0.6, len(t) - 0.4)
ax.set_ylim(0, t["rate"].max() * 1.2)
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
style(ax, "y")
titles(ax, "Late rate by delivery time promised at purchase",
       "Shorter promises are missed more often. Blue = above the overall rate.")
save(fig, "late_by_promise.png")

# ---------- Chart 4: what the model relies on ----------
importance_file = BASE / "results" / "feature_importance.csv"
if importance_file.exists():
    NAMES = {
        "promised_days": "Days promised at purchase",
        "customer_state": "Destination state",
        "ship_deadline_days": "Seller's handover deadline",
        "product": "Product category",
        "n_items": "Items in the order",
        "seller_state": "Seller's state",
        "same_state": "Seller and customer in same state",
        "seller_prev_late_rate": "Seller's past late rate",
        "distance_km": "Distance, seller to customer",
        "product_weight_g": "Product weight",
    }
    imp = pd.read_csv(importance_file)
    imp = imp[imp["importance"] > 0].head(8).iloc[::-1]  # biggest ends up on top
    names = [NAMES.get(f, f) for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(imp) + 1.9))
    ax.barh(names, imp["importance"], height=0.62, color=ACCENT)
    for i, v in enumerate(imp["importance"]):
        if i >= len(imp) - 3:  # label only the top three
            ax.text(v + imp["importance"].max() * 0.015, i, f"{v:.3f}", fontsize=9,
                    color=INK, va="center")
    ax.set_xlim(0, imp["importance"].max() * 1.12)
    ax.set_xlabel("Drop in model score when the column is shuffled", fontsize=9)
    ax.tick_params(axis="y", labelcolor=INK_2, labelsize=9.5)
    style(ax, "x")
    titles(ax, "What the model relies on most",
           "Measured on the honest time-split test. Bigger = matters more. "
           "Columns that did not help are left out.")
    save(fig, "feature_importance.png")
else:
    print("Skipped feature_importance.png: run train_model.py first.")
