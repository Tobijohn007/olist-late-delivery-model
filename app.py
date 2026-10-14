"""
app.py  (version 3)
The same late-delivery model and chatbot as before, now with a game layer:
a risk gauge, a Risk Challenge, a "Spot the late order" game, a state league,
XP, levels and badges.

Run it:
    python -m pip install -r requirements.txt
    python train_model.py          (once, creates results/model.joblib and the game sample)
    python -m streamlit run app.py
"""

import html
import math
import random
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

import chat
import explain
import predictor

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="Late delivery risk", page_icon="🚚", layout="wide")

# ------------------------------------------------------------------ Look and feel
# Semi-transparent colours, so the cards look right in both light and dark themes.
st.markdown(
    """
<style>
.hero {padding: 1.1rem 1.4rem; border-radius: 16px; margin-bottom: 0.8rem;
  background: linear-gradient(120deg, rgba(99,102,241,.25), rgba(16,185,129,.20));
  border: 1px solid rgba(128,128,128,.28);}
.hero h1 {margin: 0; font-size: 2rem; line-height: 1.2; padding: 0;}
.hero p {margin: .35rem 0 0 0; opacity: .85;}
.card {padding: .85rem 1.1rem; border-radius: 14px; margin-bottom: .6rem;
  border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.08);}
.chip {display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .85rem;
  margin: .2rem .3rem .1rem 0; border: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.12);}
.chip.gold {background: rgba(240,162,2,.22); border-color: rgba(240,162,2,.65);}
.chip.locked {opacity: .5;}
.xpbar {height: 10px; border-radius: 999px; background: rgba(128,128,128,.25); overflow: hidden; margin: .4rem 0 .2rem 0;}
.xpfill {height: 100%; border-radius: 999px; background: linear-gradient(90deg,#6366f1,#10b981);}
.verdict {font-size: 1.45rem; font-weight: 700; text-align: center; margin: 0 0 .5rem 0;}
.bar-row {display: flex; align-items: center; gap: .6rem; margin: .3rem 0; font-size: .92rem;}
.bar-label {width: 10.5rem; flex: none;}
.bar-track {flex: 1; height: 14px; border-radius: 999px; background: rgba(128,128,128,.2); overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-val {width: 3.8rem; text-align: right; flex: none; font-variant-numeric: tabular-nums;}
</style>
""",
    unsafe_allow_html=True,
)

GREEN, AMBER, RED = "#2ca25f", "#f0a202", "#e5484d"

# ------------------------------------------------------------------ Game rules
LEVELS = [(0, "Rookie Planner"), (30, "Route Scout"), (70, "Risk Analyst"), (120, "Supply Chain Pro")]

BADGES = {
    "red_flag": ("🚩 Red Flag Spotter", "Found an order the model flags as high risk"),
    "challenger": ("🎯 Challenger", "Cleared a Risk Challenge level"),
    "master": ("🏆 Challenge Master", "Earned 3 stars on every level"),
    "sharp_eye": ("👁️ Sharp Eye", "Spotted the late order 3 times"),
    "curious": ("💬 Curious Mind", "Asked the model a question"),
}

CHALLENGES = [
    {
        "title": "Level 1: Northeast rush",
        "story": "A customer in Bahia orders from a seller in Sao Paulo, with a tight promise. Make it safer.",
        "customer_state": "BA", "seller_state": "SP", "promised_days": 9, "ship_deadline_days": 2,
    },
    {
        "title": "Level 2: Far from home",
        "story": "An order is going to Alagoas, the state with the highest late rate. Can you tame it?",
        "customer_state": "AL", "seller_state": "SP", "promised_days": 7, "ship_deadline_days": 2,
    },
    {
        "title": "Level 3: Rio in a hurry",
        "story": "A Rio de Janeiro customer was promised delivery very fast, and the seller has little time to hand over.",
        "customer_state": "RJ", "seller_state": "MG", "promised_days": 5, "ship_deadline_days": 1,
    },
]

PROMISE_GRID = list(range(2, 61, 4))
HANDOVER_GRID = [1, 3, 7, 14, 30]


def init_state():
    ss = st.session_state
    ss.setdefault("xp", 0)
    ss.setdefault("badges", [])
    ss.setdefault("stars", {})
    ss.setdefault("bm_round", 1)
    ss.setdefault("bm_ids", None)
    ss.setdefault("bm_result", None)
    ss.setdefault("bm_you", 0)
    ss.setdefault("bm_model", 0)
    ss.setdefault("bm_rounds", 0)


def toast(message):
    if hasattr(st, "toast"):
        st.toast(message)


def award(key):
    if key not in st.session_state["badges"]:
        st.session_state["badges"].append(key)
        toast(f"Badge unlocked: {BADGES[key][0]}")


def add_xp(points):
    st.session_state["xp"] += points
    toast(f"+{points} XP")


def level_info(xp):
    idx = max(i for i, (need, _) in enumerate(LEVELS) if xp >= need)
    name = LEVELS[idx][1]
    if idx + 1 < len(LEVELS):
        lo, hi = LEVELS[idx][0], LEVELS[idx + 1][0]
        return idx + 1, name, (xp - lo) / (hi - lo), hi - xp
    return idx + 1, name, 1.0, 0


def render_hud(container):
    ss = st.session_state
    level, name, fraction, to_next = level_info(ss["xp"])
    chips = "".join(f'<span class="chip gold">{BADGES[k][0]}</span>' for k in ss["badges"])
    if not chips:
        chips = '<span class="chip locked">No badges yet: try the game tabs</span>'
    next_text = f"{to_next} XP to the next level" if to_next else "Top level reached"
    with container:
        st.markdown(
            f'<div class="card"><b>Level {level}: {name}</b> &nbsp;·&nbsp; {ss["xp"]} XP'
            f'<div class="xpbar"><div class="xpfill" style="width:{fraction * 100:.0f}%"></div></div>'
            f"<small>{next_text}</small><div>{chips}</div></div>",
            unsafe_allow_html=True,
        )


def gauge_svg(pct):
    """A half-circle dial: green (low), amber (watch), red (riskiest 10%)."""
    cx, cy, r = 150, 150, 110

    def point(p, radius=r):
        angle = math.pi * (1 - p / 100.0)
        return cx + radius * math.cos(angle), cy - radius * math.sin(angle)

    def arc(p0, p1, color):
        x0, y0 = point(p0)
        x1, y1 = point(p1)
        return (f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}" '
                f'stroke="{color}" stroke-width="22" fill="none"/>')

    nx, ny = point(min(max(pct, 0), 100), 88)
    return (
        '<svg viewBox="0 0 300 190" width="100%" role="img" aria-label="risk gauge" '
        'style="max-width:340px;display:block;margin:0 auto">'
        + arc(0, 60, GREEN) + arc(60, 90, AMBER) + arc(90, 100, RED)
        + f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="currentColor" '
          'stroke-width="4" stroke-linecap="round"/>'
        + f'<circle cx="{cx}" cy="{cy}" r="8" fill="currentColor"/>'
        + f'<text x="{cx}" y="{cy + 30}" text-anchor="middle" font-size="24" font-weight="700" '
          f'fill="currentColor">{pct:.0f} / 100</text>'
        + "</svg>"
    )


def verdict(result):
    if result["flagged"]:
        return "🚨 High risk: in the riskiest 10%", RED
    if result["percentile"] >= 60:
        return "⚠️ Watch this one", AMBER
    return "✅ Low risk", GREEN


def show_gauge(result):
    text, color = verdict(result)
    st.markdown(gauge_svg(result["percentile"]), unsafe_allow_html=True)
    st.markdown(f'<div class="verdict" style="color:{color}">{text}</div>', unsafe_allow_html=True)


def bar_rows(rows, color):
    """rows: list of (label, share of the longest bar from 0 to 1, text)."""
    out = []
    for label, share, text in rows:
        out.append(
            f'<div class="bar-row"><div class="bar-label">{html.escape(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{max(share, 0.02) * 100:.0f}%;'
            f'background:{color}"></div></div><div class="bar-val">{html.escape(text)}</div></div>'
        )
    return "".join(out)


# ------------------------------------------------------------------ Setup
init_state()

st.markdown(
    '<div class="hero"><h1>🚚 Will this order arrive late?</h1>'
    "<p>Enter the details known when an order is placed. The model compares the order with the "
    "future orders it was tested on and shows how risky it looks. Then play with it: earn XP, "
    "unlock badges and learn what really drives late deliveries.</p></div>",
    unsafe_allow_html=True,
)
hud = st.container()

if not predictor.MODEL_PATH.exists():
    st.error(
        "The saved model was not found. Run `python train_model.py` first. "
        "It creates results/model.joblib."
    )
    st.stop()


@st.cache_resource
def get_bundle():
    return predictor.load_bundle()


bundle = get_bundle()
med = bundle["medians"]
states = sorted(bundle["categories"]["customer_state"])
seller_states = sorted(bundle["categories"]["seller_state"])
products = bundle["product_by_popularity"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def state_label(code):
    return f"{code}  -  {predictor.STATES.get(code, code)}"


def pick_state(code, options):
    return code if code in options else options[0]


PRESETS = {
    "typical": {
        "in_customer_state": pick_state("RJ", states),
        "in_seller_state": pick_state("SP", seller_states),
        "in_product": products[0],
        "in_promised_days": int(round(med["promised_days"])),
        "in_ship_deadline_days": int(round(med["ship_deadline_days"])),
        "in_price": round(float(med["price"]), 2),
        "in_freight": round(float(med["freight_value"]), 2),
        "in_n_items": 1,
        "in_weight": float(round(med["product_weight_g"])),
    },
}
PRESETS["risky"] = dict(PRESETS["typical"], in_customer_state=pick_state("AL", states),
                        in_promised_days=8, in_ship_deadline_days=3)
PRESETS["safe"] = dict(PRESETS["typical"], in_customer_state=pick_state("SP", states),
                       in_promised_days=30)


def apply_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


if "in_customer_state" not in st.session_state:
    apply_preset("typical")

(tab_predict, tab_challenge, tab_game, tab_league,
 tab_chat, tab_results, tab_about) = st.tabs([
    "Predict an order", "Risk Challenge", "Spot the late order", "State league",
    "Ask the model", "Model results", "About",
])

# ------------------------------------------------------------------ Predict
with tab_predict:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Order details")
        p1, p2, p3 = st.columns(3)
        p1.button("🎲 Typical order", on_click=apply_preset, args=("typical",))
        p2.button("🔥 Risky order", on_click=apply_preset, args=("risky",))
        p3.button("🧊 Safe order", on_click=apply_preset, args=("safe",))

        c1, c2 = st.columns(2)
        customer_state = c1.selectbox("Destination state", states, key="in_customer_state",
                                      format_func=state_label)
        seller_state = c2.selectbox("Seller's state", seller_states, key="in_seller_state",
                                    format_func=state_label)
        product = st.selectbox("Product category", products, key="in_product")

        c3, c4 = st.columns(2)
        promised_days = c3.slider("Days promised to the customer", 1, 60, key="in_promised_days")
        ship_deadline_days = c4.slider("Seller's handover deadline (days)", 1, 30,
                                       key="in_ship_deadline_days")

        c5, c6 = st.columns(2)
        price = c5.number_input("Price", min_value=0.0, step=5.0, key="in_price")
        freight_value = c6.number_input("Freight", min_value=0.0, step=1.0, key="in_freight")
        c7, c8 = st.columns(2)
        n_items = c7.number_input("Items in the order", min_value=1, step=1, key="in_n_items")
        product_weight_g = c8.number_input("Weight (grams)", min_value=0.0, step=100.0,
                                           key="in_weight")

        with st.expander("More details (optional)"):
            weekday = st.selectbox("Day the order is placed", WEEKDAYS, index=int(med["purchase_dow"]))
            product_volume_cm3 = st.number_input(
                "Volume (cm3)", min_value=0.0, value=float(round(med["product_volume_cm3"])), step=500.0,
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
                seller_prev_late_rate = st.slider("Seller's past late rate", 0.0, 1.0, 0.07, step=0.01)

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
        "purchase_dow": WEEKDAYS.index(weekday),
        "orders_last_7d": float(orders_last_7d),
        "seller_prev_orders": float(seller_prev_orders),
        "seller_prev_late_rate": seller_prev_late_rate,
    }
    result = predictor.score_order(bundle, order)
    if result["flagged"]:
        award("red_flag")

    with right:
        st.subheader("Result")
        show_gauge(result)
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

        st.subheader("Why this score?")
        why_rows = explain.drivers(bundle, order, result["score"])
        st.dataframe(pd.DataFrame([
            {
                "Factor": r["factor"],
                "This order": r["this_order"],
                "Typical order": r["typical"],
                "Effect on score (points)": r["effect_points"],
            }
            for r in why_rows[:6]
        ]), hide_index=True)
        st.caption(
            "Each row changes one thing to a typical value and shows how much the score drops. "
            "A positive number means that factor pushes this order's risk up."
        )


# ------------------------------------------------------------------ Risk Challenge
@st.cache_data(show_spinner=False)
def challenge_numbers(_bundle, level):
    """Start risk, and the lowest risk reachable with the two levers (checked on a grid)."""
    spec = CHALLENGES[level]
    base = explain.typical_order(_bundle)
    base["customer_state"] = pick_state(spec["customer_state"], _bundle["categories"]["customer_state"])
    base["seller_state"] = pick_state(spec["seller_state"], _bundle["categories"]["seller_state"])
    start_order = dict(base, promised_days=float(spec["promised_days"]),
                       ship_deadline_days=float(spec["ship_deadline_days"]))
    start = predictor.score_order(_bundle, start_order)["percentile"]
    best = start
    for days in PROMISE_GRID:
        for handover in HANDOVER_GRID:
            trial = dict(base, promised_days=float(days), ship_deadline_days=float(handover))
            best = min(best, predictor.score_order(_bundle, trial)["percentile"])
    return base, start, best


def stars_for(progress):
    return 3 if progress >= 0.95 else 2 if progress >= 0.7 else 1 if progress >= 0.4 else 0


with tab_challenge:
    st.subheader("🎯 Risk Challenge")
    st.write(
        "Each level gives you a risky order. You control two things: the days promised to the "
        "customer and the seller's handover deadline. Bring the risk ranking down to earn stars and XP."
    )
    level_names = [c["title"] for c in CHALLENGES]
    chosen = st.radio("Pick a level", level_names, horizontal=True)
    lvl = level_names.index(chosen)
    spec = CHALLENGES[lvl]
    base_order, start_pct, best_pct = challenge_numbers(bundle, lvl)

    st.markdown(
        f'<div class="card"><b>{html.escape(spec["title"])}</b><br>{html.escape(spec["story"])}<br>'
        f'<small>Destination {html.escape(base_order["customer_state"])}, '
        f'seller in {html.escape(base_order["seller_state"])}. '
        f"Starting risk ranking: {start_pct:.0f} / 100.</small></div>",
        unsafe_allow_html=True,
    )

    cc1, cc2 = st.columns(2)
    ch_promise = cc1.slider("Days promised to the customer", 1, 60, spec["promised_days"], key=f"ch_pd_{lvl}")
    ch_handover = cc2.slider("Seller's handover deadline (days)", 1, 30, spec["ship_deadline_days"],
                             key=f"ch_hd_{lvl}")
    now = predictor.score_order(
        bundle, dict(base_order, promised_days=float(ch_promise), ship_deadline_days=float(ch_handover))
    )
    room = max(start_pct - best_pct, 1.0)
    progress = min(max((start_pct - now["percentile"]) / room, 0.0), 1.0)
    now_stars = stars_for(progress)

    g1, g2 = st.columns(2)
    with g1:
        st.caption("Now")
        show_gauge(now)
    with g2:
        st.caption("Your progress")
        st.markdown(f'<div class="verdict">{"⭐" * now_stars}{"☆" * (3 - now_stars)}</div>',
                    unsafe_allow_html=True)
        st.progress(progress)
        st.caption(f"Risk ranking moved from {start_pct:.0f} to {now['percentile']:.0f}. "
                   f"The lowest I found with these two levers is about {best_pct:.0f}.")

    if st.button("🔒 Lock in my score", key=f"lock_{lvl}"):
        best_so_far = st.session_state["stars"].get(lvl, 0)
        if now_stars == 0:
            st.warning("Not enough yet. Move the sliders until you get at least one star.")
        elif now_stars <= best_so_far:
            st.info(f"You already have {best_so_far} star(s) here. Go for more stars to earn extra XP.")
        else:
            add_xp(10 * (now_stars - best_so_far))
            st.session_state["stars"][lvl] = now_stars
            award("challenger")
            if all(st.session_state["stars"].get(i, 0) == 3 for i in range(len(CHALLENGES))):
                award("master")
            if now_stars == 3:
                st.balloons()
            st.success(f"Locked in: {now_stars} star(s)!")

    st.markdown("**Your stars:** " + "  ·  ".join(
        f"Level {i + 1}: {'⭐' * st.session_state['stars'].get(i, 0) or '–'}" for i in range(len(CHALLENGES))
    ))
    st.caption(
        "The lesson, and the trade-off: promising more days lowers the chance of being late, but a "
        "very long promise can lose the sale. Late risk is only one side of the decision."
    )


# ------------------------------------------------------------------ Spot the late order
@st.cache_data(show_spinner=False)
def load_game_orders():
    path = BASE / "results" / "challenge_orders.csv"
    return pd.read_csv(path) if path.exists() else None


def row_to_order(row):
    rate = row["seller_prev_late_rate"]
    return {
        "customer_state": str(row["customer_state"]),
        "seller_state": str(row["seller_state"]),
        "product": str(row["product"]),
        "price": float(row["price"]),
        "freight_value": float(row["freight_value"]),
        "n_items": float(row["n_items"]),
        "product_weight_g": float(row["product_weight_g"]),
        "product_volume_cm3": float(row["product_volume_cm3"]),
        "promised_days": float(row["promised_days"]),
        "ship_deadline_days": float(row["ship_deadline_days"]),
        "purchase_dow": int(row["purchase_dow"]),
        "orders_last_7d": float(row["orders_last_7d"]),
        "seller_prev_orders": float(row["seller_prev_orders"]),
        "seller_prev_late_rate": None if pd.isna(rate) else float(rate),
    }


def next_round():
    st.session_state["bm_round"] += 1
    st.session_state["bm_ids"] = None
    st.session_state["bm_result"] = None


with tab_game:
    st.subheader("🕵️ Spot the late order")
    game = load_game_orders()
    if game is None:
        st.info(
            "This game needs a small sample of real test orders. Run `python train_model.py` again, "
            "then upload `results/challenge_orders.csv` to GitHub next to model.joblib. "
            "The other tabs work without it."
        )
    else:
        st.write(
            "Five real orders from the test period: **exactly one** was delivered late. "
            "Can you spot it before the model does? (In real life only about 1 order in 22 is late. "
            "Here it is 1 in 5, so the game is playable.)"
        )
        ss = st.session_state
        if ss["bm_ids"] is None:
            late_ids = random.sample(list(game.index[game["is_late"] == 1]), 1)
            ok_ids = random.sample(list(game.index[game["is_late"] == 0]), 4)
            ids = late_ids + ok_ids
            random.shuffle(ids)
            ss["bm_ids"] = ids
            ss["bm_scores"] = [
                predictor.score_order(bundle, row_to_order(game.loc[i]))["score"] for i in ids
            ]
        ids, scores = ss["bm_ids"], ss["bm_scores"]
        letters = "ABCDE"

        def describe(pos):
            r = game.loc[ids[pos]]
            return (
                f"Order {letters[pos]}:  {predictor.STATES.get(r['customer_state'], r['customer_state'])}"
                f" (from {r['seller_state']}), {r['product']}, promised {r['promised_days']:.0f} days, "
                f"handover deadline {r['ship_deadline_days']:.0f} days, freight {r['freight_value']:.2f}"
            )

        revealed = ss["bm_result"] is not None
        pick = st.radio("Which order was late?", list(range(5)), format_func=describe,
                        key=f"bm_pick_{ss['bm_round']}", disabled=revealed)

        if not revealed and st.button("🔒 Lock in my pick"):
            truth = [bool(game.loc[i, "is_late"]) for i in ids].index(True)
            model_pick = max(range(5), key=lambda k: scores[k])
            you_ok, model_ok = pick == truth, model_pick == truth
            ss["bm_rounds"] += 1
            ss["bm_you"] += int(you_ok)
            ss["bm_model"] += int(model_ok)
            ss["bm_result"] = {"truth": truth, "model_pick": model_pick, "you_ok": you_ok, "model_ok": model_ok}
            if you_ok:
                add_xp(10)
                if ss["bm_you"] >= 3:
                    award("sharp_eye")
            st.rerun()

        if ss["bm_result"]:
            res = ss["bm_result"]
            truth_letter, model_letter = letters[res["truth"]], letters[res["model_pick"]]
            if res["you_ok"]:
                st.success(f"You spotted it: order {truth_letter} was late. +10 XP")
            else:
                st.error(f"Not this time. The late one was order {truth_letter}.")
            st.write(f"The model picked order {model_letter}, so it was "
                     f"{'right' if res['model_ok'] else 'wrong'} too.")
            st.dataframe(pd.DataFrame({
                "Order": list(letters),
                "Model score": [f"{s:.1%}" for s in scores],
                "What happened": ["LATE" if i == res["truth"] else "on time" for i in range(5)],
            }), hide_index=True)
            st.button("Next round ➡️", on_click=next_round)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Rounds played", ss["bm_rounds"])
        s2.metric("You got right", ss["bm_you"])
        s3.metric("Model got right", ss["bm_model"])
        s4.metric("Pure guessing", "1 in 5")
        st.caption(
            "The model is a modest tool, so it will miss often, and you may beat it. That is the point: "
            "the late orders are mostly caused by things nobody can see when the order is placed."
        )


# ------------------------------------------------------------------ State league
@st.cache_data(show_spinner=False)
def state_league(_bundle):
    base = explain.typical_order(_bundle)
    rows = []
    for code in _bundle["categories"]["customer_state"]:
        rows.append((code, predictor.score_order(_bundle, dict(base, customer_state=code))["score"]))
    return sorted(rows, key=lambda r: r[1], reverse=True)


with tab_league:
    st.subheader("🏁 State league")
    st.write(
        "The model's score for the same typical order shipped from Sao Paulo, changing only the "
        "destination state. This is the model's view, not the observed late rate."
    )
    league = state_league(bundle)
    top = league[0][1] or 1.0
    hot, safe = league[:10], league[-10:][::-1]
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("**🔥 Riskiest destinations**")
        st.markdown(
            bar_rows([(f"{i + 1}. {predictor.STATES.get(c, c)}", s / top, f"{s:.1%}")
                      for i, (c, s) in enumerate(hot)], RED),
            unsafe_allow_html=True,
        )
    with lc2:
        st.markdown("**🧊 Safest destinations**")
        st.markdown(
            bar_rows([(f"{i + 1}. {predictor.STATES.get(c, c)}", s / top, f"{s:.1%}")
                      for i, (c, s) in enumerate(safe)], GREEN),
            unsafe_allow_html=True,
        )
    st.caption(
        "Bars are scaled to the riskiest state. A typical order is a median-priced, median-weight "
        "order with the median promise."
    )

# ------------------------------------------------------------------ Ask the model
MAX_QUESTIONS = 8
SUGGESTIONS = [
    "Why did this order get this score?",
    "What could lower this order's risk?",
    "How reliable is this model?",
]

with tab_chat:
    st.subheader("Ask the model")
    st.write(
        "Ask why the order on the Predict tab got its score, or how the model works. "
        "Answers use only the numbers shown in this app, so the assistant explains and does not guess."
    )
    api_key = chat.get_api_key()
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "chat_count" not in st.session_state:
        st.session_state["chat_count"] = 0

    pending = None
    suggestion_cols = st.columns(len(SUGGESTIONS))
    for col, text in zip(suggestion_cols, SUGGESTIONS):
        if col.button(text):
            pending = text
    typed = st.chat_input("Ask about this order or the model")
    question = typed or pending

    if question:
        if not api_key:
            st.info(
                "The chatbot is switched off because no GROQ_API_KEY is set. "
                "The 'Why this score?' table on the Predict tab still works without it."
            )
        elif st.session_state["chat_count"] >= MAX_QUESTIONS:
            st.info(f"This demo allows {MAX_QUESTIONS} questions per visit. Refresh the page to start again.")
        else:
            context = explain.build_context(bundle, order, result, why_rows)
            history = list(st.session_state["chat_messages"])
            try:
                with st.spinner("Thinking..."):
                    answer = chat.ask(question, history, context, api_key)
                st.session_state["chat_messages"].append({"role": "user", "content": question})
                st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
                st.session_state["chat_count"] += 1
                award("curious")
            except ImportError:
                traceback.print_exc()
                st.warning(
                    "The chat library is not installed. Add `groq` to requirements.txt and redeploy."
                )
            except Exception:
                # The full error goes to the terminal (or the Render logs) so it can be fixed.
                traceback.print_exc()
                st.warning(
                    "The chatbot could not answer right now (a wrong or expired key, a retired model, "
                    "too many requests, or a connection problem). Check the logs, or try again in a "
                    "minute. The 'Why this score?' table still works."
                )

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    st.caption(
        "AI explanations can still make mistakes. The facts they rely on are in the "
        "Predict and Model results tabs."
    )

# ------------------------------------------------------------------ Results
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

# ------------------------------------------------------------------ About
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
    st.write(
        "The game layer (XP, levels, badges, challenges) is for learning and demos. Your progress "
        "lasts for this visit only and resets when you refresh the page."
    )
    st.markdown("**Badges you can earn**")
    for _, (label, how) in BADGES.items():
        st.write(f"{label}: {how}")

# The status card sits at the top of the page but is drawn last, so it shows this run's XP and badges.
render_hud(hud)
