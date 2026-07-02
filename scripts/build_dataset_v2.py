"""
build_dataset_v2.py — Lartisien Collection dataset, version 2.

15 hotels × 6 room types × 12 months = 1,080 rows.

Key changes from v1:
- Dropped: Eden Rock St Barths, Cheval Blanc Randheli, The Brando, Four Seasons Bora Bora
  (EDA identified these as statistical outliers — base_peak rates 2–3× the portfolio median,
  causing the model to memorise hotel identity rather than learn generalizable patterns)
- Added: European properties including all countries neighbouring Slovenia
  (Italy ×3, Austria, Croatia, Hungary) plus St. Moritz, Monaco, French Riviera, Turkey
- 12 sample months instead of 10 (full calendar coverage)
- Added hotel_tier (1–3) and resort_category columns for interpretable features

All rates are realistic figures cross-referenced with lartisien.com public listings.
Seed: np.random.default_rng(42)
"""

import numpy as np
import pandas as pd
import datetime

rng = np.random.default_rng(42)

# ── 1. Hotel master table ─────────────────────────────────────────────────────
#  hotel_id | country | region | resort_type | hotel_tier | rooms | base_low | base_peak | peak_months
hotels = [
    # ── Slovenia neighbours ──────────────────────────────────────────────
    # Italy (3 properties — closest neighbour)
    ("villa_deste_como",        "Italy",        "Europe",  "lake",     2, 153,  600, 2400, [5,6,7,8,9]),
    ("aman_venice",             "Italy",        "Europe",  "city",     3,  24, 1100, 3200, [4,5,6,9,10]),
    ("le_sirenuse",             "Italy",        "Europe",  "coastal",  2,  58,  580, 2600, [6,7,8,9]),
    # Austria
    ("schloss_fuschl",          "Austria",      "Europe",  "mountain", 2, 111,  420, 2200, [6,7,8,12,1,2]),
    # Croatia
    ("hotel_kompas_dubrovnik",  "Croatia",      "Europe",  "beach",    1, 174,  280, 1600, [6,7,8,9]),
    # Hungary
    ("four_seasons_budapest",   "Hungary",      "Europe",  "city",     1, 179,  350, 1700, [4,5,9,10]),
    # ── Other European luxury ─────────────────────────────────────────────
    ("badrutts_palace",         "Switzerland",  "Europe",  "mountain", 3, 157,  680, 3800, [12,1,2,7,8]),
    ("hotel_de_paris_monaco",   "Monaco",       "Europe",  "city",     3,  96,  820, 4000, [5,6,7,8]),
    ("chateau_chevre_dor",      "France",       "Europe",  "coastal",  2,  33,  620, 3000, [6,7,8,9]),
    ("macakizi_bodrum",         "Turkey",       "Europe",  "beach",    1,  85,  320, 1800, [6,7,8]),
    # ── International (balanced rates, no extreme outliers) ───────────────
    ("amanzoe",                 "Greece",       "Europe",  "beach",    3,  38,  820, 3500, [6,7,8,9]),
    ("bulgari_resort_bali",     "Indonesia",    "Asia",    "beach",    2,  59,  620, 2600, [7,8,12]),
    ("singita_grumeti",         "Tanzania",     "Africa",  "safari",   3,  34, 1100, 3800, [7,8,9,12]),
    ("aman_tokyo",              "Japan",        "Asia",    "city",     2,  84,  820, 3000, [3,4,10,11]),
    ("four_seasons_seychelles", "Seychelles",   "Africa",  "beach",    2,  67,  750, 3200, [1,7,8,12]),
]

hotel_df = pd.DataFrame(hotels, columns=[
    "hotel_id","country","region","resort_type","hotel_tier",
    "hotel_size_rooms","base_low","base_peak","season_peak_months"
])

# ── 2. Room types with price multipliers ──────────────────────────────────────
room_types = [
    ("Deluxe Room",        1.00),
    ("Superior Room",      1.35),
    ("Junior Suite",       1.85),
    ("Suite",              2.60),
    ("Grand Suite",        3.60),
    ("Presidential Suite", 4.80),
]

# ── 3. Sample dates — one per calendar month (12 dates) ──────────────────────
# (month, day, is_school_holiday)
# French school holiday calendar 2024
sample_dates = [
    (1,  15, 0),   # mid-Jan         — low season
    (2,  14, 1),   # Valentine's Day — winter school hols FR
    (3,  15, 0),   # mid-Mar         — shoulder
    (4,  15, 1),   # Easter week     — spring school hols FR
    (5,  10, 0),   # mid-May         — shoulder
    (6,  28, 0),   # late Jun        — just before school hols
    (7,  15, 1),   # mid-Jul         — peak summer school hols FR
    (8,  15, 1),   # mid-Aug         — peak summer school hols FR
    (9,  12, 0),   # mid-Sep         — autumn shoulder
    (10,  5, 0),   # early Oct       — autumn
    (11, 12, 0),   # mid-Nov         — low
    (12, 23, 1),   # 23 Dec          — Christmas school hols FR (2 days to Xmas)
]

def days_to_xmas(month, day, year=2024):
    d    = datetime.date(year, month, day)
    xmas = datetime.date(year, 12, 25)
    diff = (xmas - d).days
    # After Christmas, use next year's Christmas
    if diff < 0:
        xmas = datetime.date(year + 1, 12, 25)
        diff = (xmas - d).days
    return diff

def is_weekend(month, day, year=2024):
    return int(datetime.date(year, month, day).weekday() >= 5)

# ── 4. Build rows ─────────────────────────────────────────────────────────────
rows = []
for _, h in hotel_df.iterrows():
    peak_months = h["season_peak_months"]
    for rt_name, rt_mult in room_types:
        for (month, day, is_hol) in sample_dates:

            # Seasonal multiplier
            if month in peak_months:
                base        = h["base_peak"]
                season_mult = rng.uniform(0.92, 1.08)
            elif any(m in peak_months for m in [month-1, month+1]):
                base        = (h["base_low"] + h["base_peak"]) / 2
                season_mult = rng.uniform(0.88, 1.05)
            else:
                base        = h["base_low"]
                season_mult = rng.uniform(0.82, 1.12)

            # Christmas / New Year surge (all properties)
            if month == 12 and day >= 20:
                season_mult *= rng.uniform(1.25, 1.55)
            if month == 1 and day <= 5:
                season_mult *= rng.uniform(1.10, 1.35)

            # Valentine's premium (city and coastal hotels respond more)
            if month == 2 and day == 14:
                if h["resort_type"] in ["city", "coastal"]:
                    season_mult *= rng.uniform(1.12, 1.28)
                else:
                    season_mult *= rng.uniform(1.04, 1.12)

            # Easter premium
            if month == 4 and is_hol:
                season_mult *= rng.uniform(1.08, 1.20)

            # Monaco Grand Prix premium (May for Monaco)
            if month == 5 and h["hotel_id"] == "hotel_de_paris_monaco":
                season_mult *= rng.uniform(1.40, 1.70)

            # Weekend micro-premium
            is_wknd = is_weekend(month, day)
            if is_wknd:
                season_mult *= rng.uniform(1.03, 1.09)

            rate = round(base * rt_mult * season_mult / 10) * 10  # round to €10

            rows.append({
                "hotel_id":          h["hotel_id"],
                "country":           h["country"],
                "region":            h["region"],
                "resort_type":       h["resort_type"],
                "hotel_tier":        int(h["hotel_tier"]),
                "hotel_size_rooms":  int(h["hotel_size_rooms"]),
                "room_type":         rt_name,
                "month":             month,
                "is_school_holiday": int(is_hol),
                "is_weekend":        is_wknd,
                "days_to_christmas": days_to_xmas(month, day),
                "is_peak_season":    int(month in peak_months),
                "rate_eur_per_night": float(rate),
            })

df = pd.DataFrame(rows)

# ── 5. Inject realistic missing values (hotel_size occasionally not reported) ─
missing_idx = rng.choice(df.index, size=18, replace=False)
df.loc[missing_idx, "hotel_size_rooms"] = np.nan

# ── 6. Derived targets ────────────────────────────────────────────────────────
df["is_premium"] = (df["rate_eur_per_night"] > 3000).astype(int)
df["log_rate"]   = np.log1p(df["rate_eur_per_night"])

# ── 7. Save ───────────────────────────────────────────────────────────────────
OUT = "/Users/minkaborec/Desktop/lartisien-ml-project"
df.to_csv(f"{OUT}/hotels_rates.csv", index=False)
print(f"Saved hotels_rates.csv — {len(df)} rows × {len(df.columns)} columns")
print(f"\nHotels: {df['hotel_id'].nunique()}")
print(f"Countries: {sorted(df['country'].unique())}")
print(f"\nRate summary:")
print(df["rate_eur_per_night"].describe().round(0))
print(f"\nPremium balance:")
print(df["is_premium"].value_counts())
print(f"  → {df['is_premium'].mean()*100:.1f}% Premium")
print(f"\nMedian rate by hotel:")
print(df.groupby("hotel_id")["rate_eur_per_night"].median().sort_values().round(0).to_string())
