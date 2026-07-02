"""
Lartisien Collection — Rate Intelligence API
FastAPI backend serving GradientBoosting predictions from classifier.pkl / regressor.pkl
"""

import datetime
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load models once at startup ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
clf_pipe = joblib.load(ROOT / "classifier.pkl")
reg_pipe = joblib.load(ROOT / "regressor.pkl")

# ── Constants (mirror app.py) ─────────────────────────────────────────────────
ROOM_TIER = {
    "Deluxe Room": 1, "Superior Room": 2, "Junior Suite": 3,
    "Suite": 4,       "Grand Suite": 5,   "Presidential Suite": 6,
}
MEAN_TIER = 3.5
SCHOOL_HOL = {1:0, 2:1, 3:0, 4:1, 5:0, 6:0, 7:1, 8:1, 9:0, 10:0, 11:0, 12:1}


def days_to_xmas(month: int) -> int:
    d    = datetime.date(2024, month, 15)
    xmas = datetime.date(2024, 12, 25)
    return abs((xmas - d).days)


def build_features(hotel_id: str, month: int, room_type: str) -> pd.DataFrame:
    tier    = ROOM_TIER[room_type]
    tier_c  = tier - MEAN_TIER
    is_hol  = SCHOOL_HOL[month]
    dtx     = days_to_xmas(month)
    return pd.DataFrame([{
        "hotel_id":          hotel_id,
        "xmas_proximity":    1 / (dtx + 1),
        "month_sin":         math.sin(2 * math.pi * month / 12),
        "month_cos":         math.cos(2 * math.pi * month / 12),
        "is_school_holiday": is_hol,
        "is_weekend":        0,
        "room_tier":         tier,
        "room_tier_c_sq":    tier_c ** 2,
        "room_x_holiday":    tier * is_hol,
        "holiday_x_weekend": 0,
    }])


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lartisien Rate Intelligence API",
    description="GradientBoosting predictions for 10 Lartisien luxury hotels",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    hotel_id: str
    month: int
    room_type: str


class PredictResponse(BaseModel):
    is_premium: int
    premium_probability: float
    estimated_rate_eur: int
    label: str
    confidence: str


@app.get("/")
def health():
    return {"status": "ok", "model": "GradientBoosting · scikit-learn"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = build_features(req.hotel_id, req.month, req.room_type)

    pred_class = int(clf_pipe.predict(X)[0])
    pred_proba = float(clf_pipe.predict_proba(X)[0][1])
    pred_eur   = float(np.expm1(reg_pipe.predict(X)[0]))

    pct = pred_proba * 100
    confidence = "High" if pct > 80 or pct < 20 else "Medium" if pct > 65 or pct < 35 else "Low"

    return PredictResponse(
        is_premium=pred_class,
        premium_probability=round(pred_proba, 4),
        estimated_rate_eur=round(pred_eur),
        label="Premium (>€3,000)" if pred_class == 1 else "Standard (≤€3,000)",
        confidence=confidence,
    )
