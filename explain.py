"""
explain.py
Works out WHY an order got its score, and writes down the facts the chatbot may use.

How the "why" is found: take the order, change ONE thing at a time to a typical
value (for example, give it the typical promised days), and see how much the
model's score drops. A big drop means that thing was pushing the risk up.
"""

import predictor

# Facts from the project's data exploration. The chatbot may use these and nothing else.
PROJECT_FACTS = """\
- Overall, 6.8% of the 96,470 delivered Olist orders arrived late.
- Late rate by destination state: AL 21.4%, MA 17.4%, PI 13.9%, CE 13.8%, BA 12.2%, RJ 12.1%.
  Sao Paulo (SP) is around 4 to 5%.
- Late rate by days promised at purchase: 0-10 days 9.1%, 11-20 days 6.7%, 21-30 days 7.4%, 31+ days 4.7%.
- Lateness came in bursts (Nov 2017, Feb to Mar 2018). The data does not say why.
- In the model, days promised and destination state mattered most; the seller handover deadline
  was a small third. Distance and seller history added almost nothing, probably because state
  already captures distance (not proven).
- Limits: only delivered orders, one marketplace, straight-line distance, and the model cannot
  see events after the order (carrier problems, weather, strikes).
"""

FACTORS = [
    ("Days promised", ["promised_days"]),
    ("Destination state", ["customer_state"]),
    ("Seller's state", ["seller_state"]),
    ("Seller handover deadline", ["ship_deadline_days"]),
    ("Product category", ["product"]),
    ("Price and freight", ["price", "freight_value"]),
    ("Weight, size, items", ["product_weight_g", "product_volume_cm3", "n_items"]),
    ("Day of the week", ["purchase_dow"]),
    ("Marketplace busyness", ["orders_last_7d"]),
    ("Seller track record", ["seller_prev_orders", "seller_prev_late_rate"]),
]

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def typical_order(bundle):
    """A 'typical' value for every input, used as the comparison point."""
    cats, med = bundle["categories"], bundle["medians"]

    def pick(col, preferred):
        return preferred if preferred in cats[col] else cats[col][0]

    return {
        "customer_state": pick("customer_state", "SP"),
        "seller_state": pick("seller_state", "SP"),
        "product": bundle["product_by_popularity"][0],
        "price": med["price"],
        "freight_value": med["freight_value"],
        "n_items": 1.0,
        "product_weight_g": med["product_weight_g"],
        "product_volume_cm3": med["product_volume_cm3"],
        "promised_days": med["promised_days"],
        "ship_deadline_days": med["ship_deadline_days"],
        "purchase_dow": int(round(med["purchase_dow"])),
        "orders_last_7d": med["orders_last_7d"],
        "seller_prev_orders": med.get("seller_prev_orders", 0.0),
        "seller_prev_late_rate": None,
    }


def _fmt(key, value):
    if value is None:
        return "unknown"
    if key == "purchase_dow":
        return WEEKDAYS[int(value)]
    if isinstance(value, str):
        return value
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}".rstrip("0").rstrip(".")


def drivers(bundle, order, base_score):
    """One row per factor, biggest effect first. delta is in score points (0-100)."""
    typical = typical_order(bundle)
    rows = []
    for label, keys in FACTORS:
        changed = dict(order)
        for k in keys:
            changed[k] = typical[k]
        alt = predictor.score_order(bundle, changed)["score"]
        rows.append({
            "factor": label,
            "this_order": ", ".join(_fmt(k, order[k]) for k in keys),
            "typical": ", ".join(_fmt(k, typical[k]) for k in keys),
            "effect_points": round((base_score - alt) * 100, 1),
            "score_if_typical": alt,
        })
    rows.sort(key=lambda r: abs(r["effect_points"]), reverse=True)
    return rows


def build_context(bundle, order, result, rows):
    """The only information the chatbot is given about this order and the model."""
    lines = [
        "THE ORDER ON SCREEN",
        f"- Destination state {order['customer_state']}, seller state {order['seller_state']}, "
        f"product category {order['product']}",
        f"- Days promised {order['promised_days']:.0f}, seller handover deadline "
        f"{order['ship_deadline_days']:.0f} days, price {order['price']:.2f}, freight {order['freight_value']:.2f}",
        f"- Model score {result['score']:.1%}; risk ranking {result['percentile']:.0f} out of 100 "
        f"(higher = riskier than more of the test orders)",
        f"- In the riskiest 10%: {'yes' if result['flagged'] else 'no'}",
        "",
        "WHY (change one thing to a typical value and see the score drop; positive = that thing pushes risk up)",
    ]
    for r in rows[:6]:
        lines.append(
            f"- {r['factor']}: this order {r['this_order']}, typical {r['typical']}, "
            f"effect {r['effect_points']:+.1f} score points (score would be {r['score_if_typical']:.1%})"
        )
    lines += [
        "",
        "THE MODEL",
        f"- ROC AUC {bundle['roc_auc']:.3f} on future orders it never saw (0.5 is guessing)",
        f"- Flagging the riskiest 10% catches {bundle['caught_in_top10pct']:.1%} of late orders; "
        f"{bundle['precision_in_top10pct']:.1%} of flagged orders are really late, against "
        f"{bundle['test_late_rate']:.1%} for an average order",
        "- The score is for ranking orders, not an exact probability. Most flagged orders still arrive on time.",
        "",
        "PROJECT FACTS",
        PROJECT_FACTS,
    ]
    return "\n".join(lines)
