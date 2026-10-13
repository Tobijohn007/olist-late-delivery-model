"""
predictor.py
The scoring logic behind the web app. No screen code here, so it is easy to test.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "results" / "model.joblib"

STATES = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapa", "BA": "Bahia",
    "CE": "Ceara", "DF": "Distrito Federal", "ES": "Espirito Santo", "GO": "Goias",
    "MA": "Maranhao", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso",
    "PA": "Para", "PB": "Paraiba", "PE": "Pernambuco", "PI": "Piaui", "PR": "Parana",
    "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RO": "Rondonia", "RR": "Roraima",
    "RS": "Rio Grande do Sul", "SC": "Santa Catarina", "SE": "Sergipe", "SP": "Sao Paulo",
    "TO": "Tocantins",
}


def load_bundle():
    return joblib.load(MODEL_PATH)


def estimate_distance(bundle, seller_state, customer_state):
    """Typical seller-to-customer distance for this pair of states, in km."""
    km = bundle["pair_distance"].get(f"{seller_state}->{customer_state}")
    if km is None:
        return float(bundle["medians"]["distance_km"]), False
    return float(km), True


def score_order(bundle, order):
    """
    order: dict with customer_state, seller_state, product, price, freight_value,
    n_items, product_weight_g, product_volume_cm3, promised_days, ship_deadline_days,
    purchase_dow, orders_last_7d, seller_prev_orders, seller_prev_late_rate (None if unknown).
    """
    distance_km, distance_known = estimate_distance(
        bundle, order["seller_state"], order["customer_state"]
    )
    row = {
        "price": order["price"],
        "freight_value": order["freight_value"],
        "n_items": order["n_items"],
        "product_weight_g": order["product_weight_g"],
        "product_volume_cm3": order["product_volume_cm3"],
        "distance_km": distance_km,
        "ship_deadline_days": order["ship_deadline_days"],
        "promised_days": order["promised_days"],
        "purchase_dow": order["purchase_dow"],
        "orders_last_7d": order["orders_last_7d"],
        "same_state": int(order["seller_state"] == order["customer_state"]),
        "seller_prev_orders": order["seller_prev_orders"],
        "seller_prev_late_rate": (
            np.nan if order["seller_prev_late_rate"] is None else order["seller_prev_late_rate"]
        ),
    }
    frame = pd.DataFrame([row])
    for col in ("customer_state", "seller_state", "product"):
        frame[col] = pd.Categorical([order[col]], categories=bundle["categories"][col])
    frame = frame[bundle["features"]]

    score = float(bundle["model"].predict_proba(frame)[0, 1])
    grid = np.linspace(0, 100, 101)
    percentile = float(np.interp(score, bundle["score_percentiles"], grid))
    return {
        "score": score,
        "percentile": percentile,                       # 0-100: riskier than this % of test orders
        "flagged": score >= bundle["top10_cutoff"],     # in the riskiest 10%
        "distance_km": distance_km,
        "distance_known": distance_known,
    }
