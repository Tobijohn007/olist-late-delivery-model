"""
app.py
A small web app: enter the details of an order and see how risky it is.

Run it:
    python -m pip install -r requirements.txt
    python train_model.py          (once, creates results/model.joblib)
    python -m streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import predictor

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="Late delivery risk", layout="wide")


@st.cache_resource
def get_bundle():
    return predictor.load_bundle()


st.title("Will this order arrive late?")
st.write(
    "Enter the details known when an order is placed. The model compares the order "
    "with the future orders it was tested on and shows how risky it looks."
)

if not predictor.MODEL_PATH.exists():
    st.error(
        "The saved model was not found. Run `python train_model.py` first. "
        "It creates results/model.joblib."
    )
    st.stop()

bundle = get_bundle()
med = bundle["medians"]
states = sorted(bundle["categories"]["customer_state"])
seller_states = sorted(bundle["categories"]["seller_state"])
products = bundle["product_by_popularity"]


def state_label(code):
    return f"{code}  -  {predictor.STATES.get(code, code)}"


tab_predict, tab_results, tab_about = st.tabs(["Predict an order", "Model results", "About"])

# ---------------------------------------------------------------- Predict
with tab_predict:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Order details")
        c1, c2 = st.columns(2)
        customer_state = c1.selectbox(
            "Destination state", states,
            index=states.index("RJ") if "RJ" in states else 0, format_func=state_label,
        )
        seller_state = c2.selectbox(
            "Seller's state", seller_states,
            index=seller_states.index("SP") if "SP" in seller_states else 0,
            format_func=state_label,
        )
        product = st.selectbox("Product category", products)

        c3, c4 = st.columns(2)
        promised_days = c3.slider(
            "Days promised to the customer", 1, 60, int(round(med["promised_days"]))
        )
        ship_deadline_days = c4.slider(
            "Seller's handover deadline (days)", 1, 30, int(round(med["ship_deadline_days"]))
        )

        c5, c6 = st.columns(2)
        price = c5.number_input("Price", min_value=0.0, value=round(med["price"], 2), step=5.0)
        freight_value = c6.number_input(
            "Freight", min_value=0.0, value=round(med["freight_value"], 2), step=1.0
        )
        c7, c8 = st.columns(2)
        n_items = c7.number_input("Items in the order", min_value=1, value=1, step=1)
        product_weight_g = c8.number_input(
            "Weight (grams)", min_value=0.0, value=float(round(med["product_weight_g"])), step=100.0
        )

        with st.expander("More details (optional)"):
            weekday = st.selectbox(
                "Day the order is placed",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=int(med["purchase_dow"]),
            )
            product_volume_cm3 = st.number_input(
                "Volume (cm3)", min_value=0.0, value=float(round(med["product_volume_cm3"])),
                step=500.0,
            )
            orders_last_7d = st.number_input(
                "Orders placed in the last 7 days (marketplace busyness)",
                min_value=0.0, value=float(round(med["orders_last_7d"])), step=50.0,
            )
            seller_prev_orders = st.number_input(
                "Seller's past delivered orders", min_value=0.0,
                value=float(round(med.get("seller_prev_orders", 0))), step=10.0,
            )
            new_seller = st.checkbox("Seller has no track record yet", value=True)
            seller_prev_late_rate = None
            if not new_seller:
                seller_prev_late_rate = st.slider(
                    "Seller's past late rate", 0.0, 1.0, 0.07, step=0.01
                )

    order = {
        "customer_state": customer_state,
        "seller_state": seller_state,
        "product": product,
        "price": float(price),
        "freight_value": float(freight_value),
        "n_items": float(n_items),
        "product_weight_g": float(product_weight_g),
        "product_volume_cm3": float(product_volume_cm3),
        "promised_days": float(promised_days),
        "ship_deadline_days": float(ship_deadline_days),
        "purchase_dow": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                         "Saturday", "Sunday"].index(weekday),
        "orders_last_7d": float(orders_last_7d),
        "seller_prev_orders": float(seller_prev_orders),
        "seller_prev_late_rate": seller_prev_late_rate,
    }
    result = predictor.score_order(bundle, order)

    with right:
        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Risk ranking", f"{result['percentile']:.0f} / 100")
        m2.metric("Model score", f"{result['score']:.1%}")

        if result["flagged"]:
            st.warning(
                f"Flagged: this order is in the riskiest 10%. In testing, orders flagged "
                f"like this were late {bundle['precision_in_top10pct']:.1%} of the time, "
                f"against {bundle['test_late_rate']:.1%} for an average order."
            )
        else:
            st.success(
                f"Not flagged: this order looks riskier than about {result['percentile']:.0f}% "
                "of the orders in the test, but it is outside the riskiest 10%."
            )

        st.progress(min(max(result["percentile"] / 100.0, 0.0), 1.0))
        st.caption(
            f"Distance used: about {result['distance_km']:,.0f} km "
            + ("(typical for this pair of states)" if result["distance_known"]
               else "(typical across all routes; this pair of states was rare in the data)")
        )
        st.info(
            "Read the ranking more than the score. The model is best at putting orders in "
            "order of risk. It is not an exact probability, and most flagged orders still "
            "arrive on time."
        )

# ---------------------------------------------------------------- Results
with tab_results:
    st.subheader("How well does the model work?")
    k1, k2, k3 = st.columns(3)
    k1.metric("ROC AUC (honest test)", f"{bundle['roc_auc']:.3f}")
    k2.metric("Late orders caught in riskiest 10%", f"{bundle['caught_in_top10pct']:.1%}")
    k3.metric("Flagged orders that are really late", f"{bundle['precision_in_top10pct']:.1%}")
    st.write(
        "These numbers come from orders placed after 1 April 2018, which the model never saw "
        "while learning. Picking 10% of orders at random would catch 10% of late orders."
    )
    metrics_file = BASE / "results" / "metrics.csv"
    if metrics_file.exists():
        st.dataframe(pd.read_csv(metrics_file))
    for chart in ["late_by_state.png", "late_by_promise.png", "feature_importance.png"]:
        path = BASE / "charts" / chart
        if path.exists():
            st.image(str(path))

# ---------------------------------------------------------------- About
with tab_about:
    st.subheader("About this app")
    st.write(
        "This app scores orders from the public Olist Brazilian e-commerce dataset. "
        "The model is gradient boosting, trained on orders before 1 April 2018 and tested on "
        "orders after. It uses only facts known when an order is placed."
    )
    st.write(
        "Limits: the signal is real but modest. Most lateness comes from events after the "
        "order (carrier problems, weather, strikes), which this data cannot see. Data: "
        "Olist Brazilian E-Commerce Public Dataset on Kaggle."
    )
