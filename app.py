"""
Lartisien Collection — ML Rate Intelligence App  (D2)
======================================================
Streamlit app: user selects hotel + month + room type →
  • classifies as Premium (>€3,000) or Standard
  • estimates the nightly EUR rate
"""

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lartisien Rate Intelligence",
    page_icon="🏨",
    layout="centered",
)

# ── Load models (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf = joblib.load("classifier.pkl")
    reg = joblib.load("regressor.pkl")
    return clf, reg

clf_pipe, reg_pipe = load_models()

# ── Domain constants ─────────────────────────────────────────────────────────
HOTELS = {
    "Amanzoe (Greece)":                    "amanzoe",
    "Le Sirenuse (Italy)":                 "le_sirenuse",
    "Cheval Blanc Randheli (Maldives)":    "cheval_blanc_randheli",
    "Bulgari Resort Bali (Indonesia)":     "bulgari_resort_bali",
    "The Brando (French Polynesia)":       "the_brando",
    "Singita Grumeti (Tanzania)":          "singita_grumeti",
    "Aman Tokyo (Japan)":                  "aman_tokyo",
    "Four Seasons Bora Bora":              "four_seasons_bora_bora",
    "Singita Ebony Lodge (South Africa)":  "singita_ebony_lodge",
    "Eden Rock St Barths":                 "eden_rock_st_barths",
}

ROOM_ORDER = [
    "Deluxe Room", "Superior Room", "Junior Suite",
    "Suite", "Grand Suite", "Presidential Suite",
]
ROOM_TIER  = {r: i + 1 for i, r in enumerate(ROOM_ORDER)}
MEAN_TIER  = 3.5   # pre-computed mean of tiers 1-6

# French school-holiday months (simplified annual calendar)
SCHOOL_HOL = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0,
              7: 1, 8: 1, 9: 0, 10: 0, 11: 0, 12: 1}

MONTH_NAMES = {
    1:"January", 2:"February", 3:"March", 4:"April",
    5:"May",     6:"June",     7:"July",  8:"August",
    9:"September",10:"October",11:"November",12:"December",
}

def days_to_xmas(month: int) -> int:
    """Approximate days to Christmas using the 15th of the given month."""
    d    = datetime.date(2024, month, 15)
    xmas = datetime.date(2024, 12, 25)
    return abs((xmas - d).days)

def build_features(hotel_id: str, month: int, room_type: str) -> pd.DataFrame:
    """Convert raw user inputs into the 10-column feature DataFrame the pipeline expects."""
    tier      = ROOM_TIER[room_type]
    tier_c    = tier - MEAN_TIER
    is_hol    = SCHOOL_HOL[month]
    is_wknd   = 0   # default weekday for monthly estimates
    dtx       = days_to_xmas(month)

    return pd.DataFrame([{
        "hotel_id":          hotel_id,
        "xmas_proximity":    1 / (dtx + 1),
        "month_sin":         np.sin(2 * np.pi * month / 12),
        "month_cos":         np.cos(2 * np.pi * month / 12),
        "is_school_holiday": is_hol,
        "is_weekend":        is_wknd,
        "room_tier":         tier,
        "room_tier_c_sq":    tier_c ** 2,
        "room_x_holiday":    tier * is_hol,
        "holiday_x_weekend": is_hol * is_wknd,
    }])

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🏨 Lartisien Collection")
st.subheader("ML Rate Intelligence — Predict nightly rates & tier classification")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    hotel_label = st.selectbox("🌍 Hotel", list(HOTELS.keys()))

with col2:
    month_label = st.selectbox("📅 Month", list(MONTH_NAMES.values()))
    month_num   = [k for k, v in MONTH_NAMES.items() if v == month_label][0]

with col3:
    room_type = st.selectbox("🛏 Room Type", ROOM_ORDER)

st.markdown("---")

if st.button("🔮  Predict", use_container_width=True, type="primary"):
    hotel_id = HOTELS[hotel_label]
    X        = build_features(hotel_id, month_num, room_type)

    # Classification
    pred_class = int(clf_pipe.predict(X)[0])
    pred_proba = float(clf_pipe.predict_proba(X)[0][1])

    # Regression
    pred_log  = float(reg_pipe.predict(X)[0])
    pred_eur  = float(np.expm1(pred_log))

    # Display
    st.markdown("### Results")
    rcol1, rcol2 = st.columns(2)

    with rcol1:
        if pred_class == 1:
            st.success("**💎 PREMIUM BOOKING**\n\nRate > €3,000/night")
        else:
            st.info("**🏷 STANDARD BOOKING**\n\nRate ≤ €3,000/night")
        st.metric("Premium probability", f"{pred_proba*100:.1f}%")

    with rcol2:
        st.metric("Estimated nightly rate", f"€ {pred_eur:,.0f}")
        st.caption(f"log_rate = {pred_log:.3f} → expm1 → EUR")

    # Context
    st.markdown("---")
    with st.expander("ℹ️  How this prediction was made"):
        st.markdown(f"""
**Input features computed from your selection:**

| Feature | Value |
|---------|-------|
| hotel_id | `{hotel_id}` |
| month | {month_num} ({month_label}) |
| room_tier | {ROOM_TIER[room_type]} ({room_type}) |
| xmas_proximity | {1/(days_to_xmas(month_num)+1):.4f} |
| month_sin | {np.sin(2*np.pi*month_num/12):.4f} |
| month_cos | {np.cos(2*np.pi*month_num/12):.4f} |
| is_school_holiday | {SCHOOL_HOL[month_num]} |
| room_x_holiday | {ROOM_TIER[room_type] * SCHOOL_HOL[month_num]} |

**Models:** GradientBoosting pipelines (preprocessor + model in one object).
**Test accuracy (classification):** F1-macro = 0.862 · ROC-AUC = 0.941
**Test accuracy (regression):** R² = 0.930 · MAE = €1,019/night
**72% of predictions are within ±20% of the actual rate.**
        """)

    st.caption("⚠️  Valid only for the 10 Lartisien training hotels. "
               "Does not capture real-time promotions or last-minute pricing.")

# ── Sidebar — findings dashboard ─────────────────────────────────────────────
st.sidebar.title("📊 Model Findings")
st.sidebar.markdown("""
**Classification — is_premium**
- Best model: GradBoost (tuned)
- Test F1-macro: **0.862**
- Test ROC-AUC: **0.941**
- Beats baseline by +0.468 F1

**Regression — EUR/night**
- Best model: GradBoost (tuned)
- Test R²: **0.930**
- MAE: **€1,019/night**
- 72% within ±20% of actual

**Top pricing drivers**
1. 🛏 Room category (30%)
2. 🏨 Hotel brand (≈25%)
3. 📅 Seasonality (≈25%)

**Dataset**
- 10 Lartisien hotels
- 600 rows · 10 features
- 480 train / 120 test
""")
st.sidebar.markdown("---")
st.sidebar.caption("Built with scikit-learn GradientBoosting · Streamlit · MADA 2025")
