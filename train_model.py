"""
train_model.py  (version 2)
Predict whether an order will arrive late, using ONLY information we know at the
moment the customer places the order.

New in version 2: distance, seller handover deadline, product volume, and the
seller's track record.

Run after the NEW build_deliveries.py has created deliveries.csv:
    python -m pip install scikit-learn      (once)
    python train_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE / "deliveries.csv",
    parse_dates=["order_purchase_timestamp", "promised_date", "actual_delivery_date"],
)

if "distance_km" not in df.columns:
    raise SystemExit(
        "deliveries.csv is from the old build script. Replace build_deliveries.py "
        "with the new version, run it again, then run this script."
    )

# ---------- 1. Build the features ----------
# Oldest order first, so the "last 7 days" calculation below looks backwards in time.
df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

df["seller_state"] = df["route"].str.split("->").str[0]
df["customer_state"] = df["route"].str.split("->").str[1]
df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)

# Days the customer was promised when they ordered.
df["promised_days"] = (df["promised_date"] - df["order_purchase_timestamp"]).dt.days
df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek  # 0 = Monday

# Congestion signal: how many orders were placed in the 7 days up to this order.
counts = pd.Series(1, index=df["order_purchase_timestamp"])
df["orders_last_7d"] = counts.rolling("7D").sum().to_numpy()

# Seller track record: how many orders this seller had ALREADY DELIVERED before
# this order was placed, and what share of those were late.
# We only count deliveries that finished before the purchase, because that is all
# anyone could have known on that day. Counting later ones would be cheating.
history = df[["supplier", "actual_delivery_date", "is_late"]].sort_values("actual_delivery_date")
history["seller_prev_orders"] = history.groupby("supplier").cumcount() + 1
history["seller_prev_late"] = history.groupby("supplier")["is_late"].cumsum()
matched = pd.merge_asof(
    df[["order_purchase_timestamp", "supplier"]],   # already sorted by purchase time
    history[["actual_delivery_date", "supplier", "seller_prev_orders", "seller_prev_late"]],
    left_on="order_purchase_timestamp",
    right_on="actual_delivery_date",
    by="supplier",
    allow_exact_matches=False,
)
# New sellers have no history: the count is 0 and the late rate stays empty.
df["seller_prev_late_rate"] = (matched["seller_prev_late"] / matched["seller_prev_orders"]).to_numpy()
df["seller_prev_orders"] = matched["seller_prev_orders"].fillna(0).to_numpy()

# NOT used on purpose:
#  - actual_delivery_date and delay_days: they ARE the answer. We only know them
#    after delivery, so using them would be cheating ("data leakage").
#  - supplier (the seller's ID): thousands of sellers, many with a handful of
#    orders. The model would memorise names instead of learning patterns. The
#    seller's track record above captures what matters.
NUMBER = [
    "price", "freight_value", "n_items", "product_weight_g", "product_volume_cm3",
    "distance_km", "ship_deadline_days", "promised_days", "purchase_dow",
    "orders_last_7d", "same_state", "seller_prev_orders", "seller_prev_late_rate",
]
CATEGORY = ["customer_state", "seller_state", "product"]

X = df[NUMBER + CATEGORY].copy()
for col in CATEGORY:
    X[col] = X[col].astype("category")  # tells the model these are labels, not numbers
y = df["is_late"].to_numpy()


# ---------- 2. Helpers ----------
def make_model():
    # Gradient boosting: builds many small decision trees, each one fixing the
    # mistakes of the previous ones. It handles labels and missing values itself.
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        max_iter=200,
        learning_rate=0.05,
        random_state=42,
    )


all_metrics = []   # filled in by run(), saved to results/metrics.csv at the end


def evaluate(y_true, p):
    """p = the model's probability that each order is late."""
    base_rate = y_true.mean()
    k = int(len(p) * 0.10)                      # the riskiest 10% of orders
    flagged = np.argsort(-p)[:k]
    auc = roc_auc_score(y_true, p)
    ap = average_precision_score(y_true, p)
    caught = y_true[flagged].sum() / y_true.sum()
    precision = y_true[flagged].mean()
    print(f"  ROC AUC:            {auc:.3f}   (0.5 = guessing, 1.0 = perfect)")
    print(f"  Average precision:  {ap:.3f}   (guessing would score {base_rate:.3f})")
    print("  Flag the riskiest 10% of orders:")
    print(f"     catches {caught:.0%} of all late orders (guessing would catch 10%)")
    print(f"     {precision:.0%} of flagged orders really are late (guessing: {base_rate:.0%})")
    return {
        "roc_auc": auc,
        "avg_precision": ap,
        "late_rate_in_test": base_rate,
        "caught_in_top10pct": caught,
        "precision_in_top10pct": precision,
    }


def run(name, X_train, y_train, X_test, y_test):
    print(f"\n=== {name} ===")
    print(f"  Train: {len(y_train):,} orders ({y_train.mean():.1%} late)")
    print(f"  Test:  {len(y_test):,} orders ({y_test.mean():.1%} late)")
    model = make_model().fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict_proba(X_test)[:, 1])
    metrics.update(test=name, train_orders=len(y_train), test_orders=len(y_test))
    all_metrics.append(metrics)
    return model


# ---------- 3. Test A: random split ----------
# Hide a random 20% of orders, train on the rest. Easy, but optimistic: the model
# has seen orders from the same weeks it is tested on.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
run("TEST A: random split", X_tr, y_tr, X_te, y_te)

# ---------- 4. Test B: split by time ----------
# Train on the past, test on the future. This is how a model would really be used,
# so it is the more honest score.
CUTOFF = pd.Timestamp("2018-04-01")
past = (df["order_purchase_timestamp"] < CUTOFF).to_numpy()
model_b = run(
    f"TEST B: train before {CUTOFF.date()}, test after",
    X[past], y[past], X[~past], y[~past],
)

# ---------- 5. Which features matter? ----------
# Shuffle one column at a time and see how much the score drops.
# A big drop means the model leans on that column.
X_test_b, y_test_b = X[~past], y[~past]
rows = np.random.default_rng(0).choice(len(y_test_b), size=min(20000, len(y_test_b)), replace=False)
imp = permutation_importance(
    model_b, X_test_b.iloc[rows], y_test_b[rows],
    scoring="average_precision", n_repeats=3, random_state=0,
)
ranking = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
print("\n=== FEATURE IMPORTANCE (Test B model; bigger = matters more) ===")
print(ranking.round(4).to_string())

# ---------- 6. Save the results ----------
# Small files that make_charts.py reads, and that you can keep on GitHub.
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
pd.DataFrame(all_metrics).round(4).to_csv(RESULTS / "metrics.csv", index=False)
ranking.rename("importance").round(5).to_csv(RESULTS / "feature_importance.csv", index_label="feature")
print(f"\nSaved results to {RESULTS}")
