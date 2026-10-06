"""
build_deliveries.py  (version 2)
Joins the Olist CSV files into one flat file: deliveries.csv
One row = one delivered order.

New in version 2: distance between seller and customer, the seller's handover
deadline, and product volume.

Put the unzipped Olist CSVs in a folder called "data" next to this script.
Run:  python build_deliveries.py   (from any folder, it finds "data" by itself)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Find the "data" folder relative to THIS script, not relative to wherever
# the terminal happens to be. This avoids "No such file or directory" errors.
BASE = Path(__file__).resolve().parent
DATA = str(BASE / "data") + "/"

# ---------- 1. Load the files ----------
# parse_dates turns the date text into real dates so we can subtract them.
orders = pd.read_csv(
    DATA + "olist_orders_dataset.csv",
    parse_dates=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
)
items = pd.read_csv(DATA + "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
customers = pd.read_csv(DATA + "olist_customers_dataset.csv")
sellers = pd.read_csv(DATA + "olist_sellers_dataset.csv")
products = pd.read_csv(DATA + "olist_products_dataset.csv")
translation = pd.read_csv(DATA + "product_category_name_translation.csv")
geo = pd.read_csv(
    DATA + "olist_geolocation_dataset.csv",
    usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
)

# ---------- 2. Keep only orders we can learn from ----------
# An order needs a delivered status AND a delivery date to have a known outcome.
orders = orders[orders["order_status"] == "delivered"]
orders = orders.dropna(subset=["order_delivered_customer_date"])

# ---------- 3. One row per order from the items table ----------
# An order can contain several items. To keep one row per order, we take the
# first item (its seller, product and handover deadline) and add up price and freight.
items = items.sort_values(["order_id", "order_item_id"])
first_item = items.drop_duplicates("order_id", keep="first")[
    ["order_id", "product_id", "seller_id", "shipping_limit_date"]
]
totals = items.groupby("order_id").agg(
    price=("price", "sum"),
    freight_value=("freight_value", "sum"),
    n_items=("order_item_id", "count"),
).reset_index()

# ---------- 4. One location per zip code prefix ----------
# The geolocation file has many rows per zip prefix, and some coordinates are
# clearly wrong (outside Brazil). Drop those, then average what is left.
geo = geo[
    geo["geolocation_lat"].between(-34, 6) & geo["geolocation_lng"].between(-74, -34)
]
geo = (
    geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
    .mean()
    .reset_index()
)
customer_geo = geo.rename(columns={
    "geolocation_zip_code_prefix": "customer_zip_code_prefix",
    "geolocation_lat": "customer_lat",
    "geolocation_lng": "customer_lng",
})
seller_geo = geo.rename(columns={
    "geolocation_zip_code_prefix": "seller_zip_code_prefix",
    "geolocation_lat": "seller_lat",
    "geolocation_lng": "seller_lng",
})

# ---------- 5. Join everything (left joins keep all delivered orders) ----------
df = (
    orders
    .merge(first_item, on="order_id", how="left")
    .merge(totals, on="order_id", how="left")
    .merge(
        customers[["customer_id", "customer_state", "customer_zip_code_prefix"]],
        on="customer_id", how="left",
    )
    .merge(
        sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]],
        on="seller_id", how="left",
    )
    .merge(
        products[[
            "product_id", "product_category_name", "product_weight_g",
            "product_length_cm", "product_height_cm", "product_width_cm",
        ]],
        on="product_id", how="left",
    )
    .merge(translation, on="product_category_name", how="left")
    .merge(customer_geo, on="customer_zip_code_prefix", how="left")
    .merge(seller_geo, on="seller_zip_code_prefix", how="left")
)

# ---------- 6. Build the columns ----------
df["promised_date"] = df["order_estimated_delivery_date"]
df["actual_delivery_date"] = df["order_delivered_customer_date"]
df["route"] = df["seller_state"] + "->" + df["customer_state"]
df["supplier"] = df["seller_id"]
df["product"] = df["product_category_name_english"].fillna("unknown")


def haversine_km(lat1, lng1, lat2, lng2):
    """Straight-line distance over the Earth's surface, in km."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# Empty when a zip prefix has no coordinates. The model handles empty values.
df["distance_km"] = haversine_km(
    df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"]
)

# Days the seller has, from the moment of purchase, to hand the parcel to the
# carrier. Known at purchase time, so it is safe to use as a feature.
df["ship_deadline_days"] = (
    (df["shipping_limit_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
)

df["product_volume_cm3"] = (
    df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
)

# Positive delay = arrived after the promised date (late).
df["delay_days"] = (df["actual_delivery_date"] - df["promised_date"]).dt.days
df["is_late"] = (df["delay_days"] > 0).astype(int)

# ---------- 7. Save ----------
out = df[
    [
        "order_id", "order_purchase_timestamp", "promised_date",
        "actual_delivery_date", "route", "supplier", "product",
        "price", "freight_value", "n_items", "product_weight_g",
        "product_volume_cm3", "distance_km", "ship_deadline_days",
        "delay_days", "is_late",
    ]
].dropna(subset=["route"])

out.to_csv(BASE / "deliveries.csv", index=False)
print(f"Saved deliveries.csv with {len(out)} rows")
print(f"Share of late deliveries: {out['is_late'].mean():.1%}")
print(f"Orders with a distance:   {out['distance_km'].notna().mean():.1%}")
print(f"Median distance:          {out['distance_km'].median():.0f} km")
