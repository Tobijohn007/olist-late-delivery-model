"""
explore.py
First look at deliveries.csv: how often are orders late, and where?

Run after build_deliveries.py has created deliveries.csv:
    python explore.py
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE / "deliveries.csv",
    parse_dates=["order_purchase_timestamp", "promised_date", "actual_delivery_date"],
)

# ---------- Extra columns, only for exploring ----------
# route looks like "SP->RJ" (seller state -> customer state). Split it apart.
df["seller_state"] = df["route"].str.split("->").str[0]
df["customer_state"] = df["route"].str.split("->").str[1]
df["same_state"] = df["seller_state"] == df["customer_state"]

# Month the order was placed, e.g. "2017-11".
df["purchase_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

# How many days the customer was promised at the moment of purchase.
df["promised_days"] = (df["promised_date"] - df["order_purchase_timestamp"]).dt.days
df["promised_band"] = pd.cut(
    df["promised_days"],
    bins=[-1, 10, 20, 30, 1000],
    labels=["0-10 days", "11-20 days", "21-30 days", "31+ days"],
)

# Product weight in bands (kg).
df["weight_band"] = pd.cut(
    df["product_weight_g"] / 1000,
    bins=[0, 0.5, 2, 5, 10, 1000],
    labels=["under 0.5 kg", "0.5-2 kg", "2-5 kg", "5-10 kg", "10+ kg"],
)


def late_rate_by(column, min_orders=200, top=None, sort_by_index=False):
    """Number of orders and % late for each value of `column`.
    Groups with fewer than `min_orders` orders are hidden: a rate based on
    10 orders is just noise."""
    t = df.groupby(column, observed=True)["is_late"].agg(
        orders="count", late_pct="mean"
    )
    t = t[t["orders"] >= min_orders]
    t["late_pct"] = (t["late_pct"] * 100).round(1)
    if sort_by_index:
        t = t.sort_index()
    else:
        t = t.sort_values("late_pct", ascending=False)
    return t.head(top) if top else t


def show(title, table):
    print(f"\n=== {title} ===")
    print(table.to_string())


# ---------- 1. Overview ----------
late_share = df["is_late"].mean()
late_orders = df[df["is_late"] == 1]

print("=== OVERVIEW ===")
print(f"Delivered orders:                 {len(df):,}")
print(f"Late orders:                      {df['is_late'].sum():,} ({late_share:.1%})")
print(f"Accuracy of 'always on time':     {1 - late_share:.1%}   <- the score any model must beat")
print(f"Typical lateness (median, late):  {late_orders['delay_days'].median():.0f} days")
print(f"Very late (more than 7 days):     {(df['delay_days'] > 7).mean():.1%} of all orders")
print(f"Average days early (all orders):  {-df['delay_days'].mean():.1f}")

# ---------- 2. Where are orders late? ----------
show("Late % by customer state (destination), worst first", late_rate_by("customer_state", top=10))
show("Late % by seller state (origin), worst first", late_rate_by("seller_state", top=10))
show("Late % when seller and customer are in the same state", late_rate_by("same_state", min_orders=1))

# ---------- 3. What is late? ----------
show("Late % by product category, worst 10 (at least 300 orders)", late_rate_by("product", min_orders=300, top=10))
show("Late % by product weight", late_rate_by("weight_band", min_orders=1, sort_by_index=True))

# ---------- 4. When is it late? ----------
show("Late % by purchase month", late_rate_by("purchase_month", sort_by_index=True))
show("Late % by promised delivery window", late_rate_by("promised_band", min_orders=1, sort_by_index=True))

print("\nDone. Read each table and ask: where is the late % clearly above 6.8%?")
