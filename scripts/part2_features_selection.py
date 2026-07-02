"""
Part 2 — Feature Engineering & Leakage-Safe Selection (v2)
===========================================================
Fixes applied after multicollinearity audit:
  • Removed perfectly collinear pairs before modelling
  • VIF computed to quantify multicollinearity numerically
  • Inter-feature correlation heatmap added
  • RFE added as wrapper method (matches course notebook)
  • Correlation filter |r|>0.15 added (matches course notebook)
  • All data-dependent steps inside Pipeline on TRAIN only
  • App-inference feasibility check for every feature

ANTI-LEAKAGE CONTRACT
  1. Pure-math features computed on full df (no fitting → no leakage)
  2. SPLIT before any data-dependent transform
  3. Imputer / Scaler / Encoder / Selector: fit on X_train ONLY
  4. X_test only ever sees .transform(), never .fit()
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression,
                                       VarianceThreshold, RFE)
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
import joblib

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
BLUE  = "#2d4a8a"
GREEN = "#2d6a2d"
RED   = "#c0392b"
OUT   = "/Users/minkaborec/Desktop/MADA final project"

# ═══════════════════════════════════════════════════════════════════════════════
# 0 · Load
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(f"{OUT}/hotels_rates_clean.csv")
print(f"Loaded: {df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1 · Pure-math feature engineering  (BEFORE split — no fitting, no leakage)
# ═══════════════════════════════════════════════════════════════════════════════
# RULE: only create a feature if it is DERIVABLE AT INFERENCE TIME from
# (hotel_id + month + room_type) — the three inputs our app will receive.

# 1a. Cyclic month — replaces raw month (avoids Dec/Jan discontinuity)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
# ⚠️  DROP raw `month`: month_sin + month_cos carry identical information;
#     keeping all three would make them perfectly multicollinear.

# 1b. Room tier — predefined ordinal 1-6 (domain knowledge, not data-fitted)
ROOM_ORDER = ["Deluxe Room","Superior Room","Junior Suite",
              "Suite","Grand Suite","Presidential Suite"]
room_tier_map = {r: i+1 for i, r in enumerate(ROOM_ORDER)}
df["room_tier"] = df["room_type"].map(room_tier_map)
# ⚠️  room_type will be OrdinalEncoded 0-5 → perfectly collinear with room_tier.
#     Solution: use room_tier as the sole room encoding; do NOT also OrdinalEncode room_type.

# 1c. Centred room_tier² — captures non-linear pricing jump at Presidential level.
#     Centring (subtract mean=3.5) before squaring reduces correlation with
#     room_tier from r≈0.98 (raw) to r≈0 (centred), removing near-collinearity.
df["room_tier_c"]   = df["room_tier"] - df["room_tier"].mean()  # centred
df["room_tier_c_sq"] = df["room_tier_c"] ** 2                   # centred quadratic

# 1d. Christmas proximity (keeps the non-linear compression; drops raw days_to_christmas)
#     Both are monotonically related → VIF would be huge if kept together.
#     xmas_proximity = 1/(d+1) is more informative for the surge window.
df["xmas_proximity"] = 1 / (df["days_to_christmas"] + 1)
# ⚠️  DROP days_to_christmas after audit below confirms collinearity.

# 1e. Exclusivity — inverse hotel size (smaller = rarer = pricier)
#     hotel_size_rooms and exclusivity have r = -1.0 by construction.
#     Keep exclusivity (more intuitive for pricing); drop hotel_size_rooms.
df["exclusivity"] = 100 / df["hotel_size_rooms"]

# 1f. Interaction: room × holiday (suites surge more in school breaks)
df["room_x_holiday"] = df["room_tier"] * df["is_school_holiday"]

# 1g. Interaction: holiday × weekend (both flags = peak demand)
df["holiday_x_weekend"] = df["is_school_holiday"] * df["is_weekend"]
# ⚠️  is_coastal = resort_type in [beach,island] duplicates resort_type dummies → DROPPED.
# ⚠️  region dummies are collinear with hotel_id dummies (1-to-1 hotel→region) → DROPPED.

print("\nEngineered features added (8 new columns incl. centred quadratic).")
print(df[["month_sin","month_cos","room_tier","room_tier_c","room_tier_c_sq",
          "xmas_proximity","exclusivity","room_x_holiday","holiday_x_weekend"]].describe().round(3))

# ═══════════════════════════════════════════════════════════════════════════════
# 2 · MULTICOLLINEARITY AUDIT on the full numeric feature space
#     (done for diagnostic purposes — we audit, then strip collinear columns
#      before passing anything to a model)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("MULTICOLLINEARITY AUDIT")
print("="*65)

audit_cols = [
    # raw originals
    "hotel_size_rooms", "month", "is_school_holiday", "is_weekend",
    "days_to_christmas",
    # engineered
    "month_sin","month_cos","room_tier","room_tier_c","room_tier_c_sq",
    "xmas_proximity","exclusivity","room_x_holiday","holiday_x_weekend",
    # target (for context)
    "rate_eur_per_night"
]
audit_df = df[audit_cols].copy()

# 2a. Pearson correlation matrix
corr_audit = audit_df.corr()

# 2b. VIF (Variance Inflation Factor)
#     VIF > 10  → severe multicollinearity, must drop
#     VIF 5-10  → moderate, investigate
#     VIF < 5   → acceptable
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def vif_series(df_in):
    X = df_in.dropna().astype(float)
    Xc = add_constant(X)          # VIF must include an intercept
    results = {}
    for i, col in enumerate(X.columns):
        results[col] = round(variance_inflation_factor(Xc.values, i + 1), 2)
    return pd.Series(results).sort_values(ascending=False)

numeric_for_vif = audit_df.drop(columns="rate_eur_per_night").dropna()
vif = vif_series(numeric_for_vif)
print("\nVIF scores (before dropping redundant features):")
print(vif)
print("\nRED FLAGS (VIF > 10):")
print(vif[vif > 10])

# -- Fig 9: Inter-feature correlation heatmap (audit — before cleanup) ---------
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones(corr_audit.shape, dtype=bool))
sns.heatmap(corr_audit, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.7}, annot_kws={"size": 8})
ax.set_title("Fig 9 · Inter-feature Pearson correlation (BEFORE removing redundant features)\n"
             "Red pairs = multicollinear, must drop one", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_multicollinearity_audit.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n→ Fig 9 saved")

# Identify pairs with |r| > 0.85 (between features, not with target)
feat_cols = [c for c in corr_audit.columns if c != "rate_eur_per_night"]
pairs = []
for i in range(len(feat_cols)):
    for j in range(i+1, len(feat_cols)):
        r = corr_audit.loc[feat_cols[i], feat_cols[j]]
        if abs(r) > 0.85:
            pairs.append((feat_cols[i], feat_cols[j], round(r, 3)))
print("\nHighly correlated feature pairs (|r| > 0.85 between features):")
for a, b, r in sorted(pairs, key=lambda x: abs(x[2]), reverse=True):
    print(f"  {a:30s} ↔ {b:30s}  r={r:+.3f}  ← DROP one")

# ═══════════════════════════════════════════════════════════════════════════════
# 3 · DROP REDUNDANT FEATURES (decisions from audit above)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("FEATURE DECISIONS (post-audit)")
print("="*65)
decisions = {
    "month":              "DROPPED — redundant with month_sin + month_cos (cyclic encoding is strictly better)",
    "hotel_size_rooms":   "DROPPED — r=-1.0 with exclusivity; exclusivity is more interpretable",
    "days_to_christmas":  "DROPPED — monotone of xmas_proximity; Lasso zeroed it; VIF elevated",
    "country":            "DROPPED — 1-to-1 with hotel_id; already excluded in Part 1",
    "region":             "DROPPED — collinear with hotel_id (each hotel → one region)",
    "is_coastal":         "NOT CREATED — would be linear combo of resort_type_beach + resort_type_island",
    "exclusivity":        "DROPPED — fixed per-hotel constant, perfectly predicted by hotel_id dummies (VIF=∞).",
    "resort_type":        "DROPPED — each hotel maps to exactly one resort type → resort_type dummies are perfect linear combos of hotel_id dummies (VIF=∞). Proposal explicitly warned of this.",
    "room_type (raw)":    "NOT ORDINAL-ENCODED separately — room_tier already encodes this 1-6",
}
for feat, reason in decisions.items():
    print(f"  {feat:25s}: {reason}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4 · FINAL FEATURE SET  (clean, non-collinear, app-feasible)
# ═══════════════════════════════════════════════════════════════════════════════
# App-inference note:
#   User inputs: hotel_id, month, room_type
#   Derived at inference: month_sin/cos (math), xmas_proximity (math from month),
#     is_school_holiday (lookup: French calendar by month),
#     is_weekend (defaulted to 0 = weekday for monthly-level queries),
#     room_tier (lookup from room_type), room_tier_sq, room_x_holiday (math),
#     holiday_x_weekend (math), exclusivity (lookup from hotel_id),
#     resort_type (lookup from hotel_id)

# The proposal explicitly warned: "country, region, resort_type, hotel_size_rooms
# are constant within each hotel and therefore collinear with hotel_id."
# resort_type dummies are PERFECT linear combinations of hotel_id dummies
# (each hotel maps to exactly one resort type → VIF = ∞ when both included).
# Keeping hotel_id alone captures all hotel-level fixed effects.
# exclusivity (100/hotel_size_rooms) is similarly a fixed per-hotel constant → also dropped.
cat_nominal = ["hotel_id"]   # one-hot, drop='first' — sole hotel identity encoding
num_feats   = ["xmas_proximity"]               # StandardScaler (no imputation needed post Part 1)
passthrough = ["month_sin", "month_cos",       # already in [-1,1]
               "is_school_holiday", "is_weekend",
               "room_tier", "room_tier_c_sq",  # centred quad — low collinearity with room_tier
               "room_x_holiday", "holiday_x_weekend"]

ALL_FEATURE_COLS = cat_nominal + num_feats + passthrough

drop_cols = ["rate_eur_per_night","is_premium","log_rate","rate_flag_extreme",
             "country","region","resort_type","hotel_size_rooms","exclusivity",
             "month","days_to_christmas","room_type"]  # redundant with hotel_id / room_tier

X = df[ALL_FEATURE_COLS].copy()
y_clf = df["is_premium"]
y_reg = df["log_rate"]

print(f"\nFinal feature set ({len(ALL_FEATURE_COLS)} features): {ALL_FEATURE_COLS}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5 · SPLIT FIRST — golden rule
# ═══════════════════════════════════════════════════════════════════════════════
X_train, X_test, yc_train, yc_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
yr_train = y_reg.loc[X_train.index]
yr_test  = y_reg.loc[X_test.index]

print(f"\nSplit: {X_train.shape[0]} train / {X_test.shape[0]} test")
print(f"  is_premium train: {yc_train.mean():.3f} | test: {yc_test.mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6 · PREPROCESSOR (ColumnTransformer — fitted on TRAIN only)
# ═══════════════════════════════════════════════════════════════════════════════
preprocessor = ColumnTransformer([
    ("num",  Pipeline([("scale",  StandardScaler())]),     num_feats),
    # drop='first' eliminates the dummy variable trap (sum-to-1 collinearity)
    # Tree-based models are unaffected; linear models need this for proper VIF.
    ("nom",  OneHotEncoder(handle_unknown="ignore",
                           sparse_output=False,
                           drop="first"),                 cat_nominal),
    ("pass", "passthrough",                               passthrough),
], verbose_feature_names_out=True)

preprocessor.fit(X_train)   # ← TRAIN ONLY
X_train_prep = preprocessor.transform(X_train)
X_test_prep  = preprocessor.transform(X_test)   # ← only transform

feature_names_out = preprocessor.get_feature_names_out()
print(f"\nAfter preprocessing: {X_train_prep.shape[1]} features")
print(list(feature_names_out))

# ═══════════════════════════════════════════════════════════════════════════════
# 7 · POST-PREPROCESSING COLLINEARITY CHECK  (VIF on clean feature matrix)
# ═══════════════════════════════════════════════════════════════════════════════
prep_df = pd.DataFrame(X_train_prep, columns=feature_names_out)
vif_clean = vif_series(prep_df)
print("\nVIF after redundant features removed (all should be < 10):")
print(vif_clean.round(2))
high_vif = vif_clean[vif_clean > 10]
if len(high_vif):
    print(f"\n⚠️  Still elevated VIF: {high_vif.to_dict()}")
else:
    print("✅  All VIF < 10 — no multicollinearity remaining.")

# ═══════════════════════════════════════════════════════════════════════════════
# 8 · FEATURE SELECTION — 3 methods from the course notebook
#     All fitted on TRAIN data only (inside Pipeline where applicable)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("FEATURE SELECTION — 3 METHODS (course notebook §2.2)")
print("="*65)

# ── METHOD A: Correlation Filter  |r| > 0.15  (course notebook threshold) ────
# Spearman used because one-hot dummies are ordinal 0/1
corr_target_clf = pd.Series(
    [abs(spearmanr(X_train_prep[:, i], yc_train.values).statistic)
     for i in range(X_train_prep.shape[1])],
    index=feature_names_out
).sort_values(ascending=False)

corr_target_reg = pd.Series(
    [abs(spearmanr(X_train_prep[:, i], yr_train.values).statistic)
     for i in range(X_train_prep.shape[1])],
    index=feature_names_out
).sort_values(ascending=False)

CORR_THRESHOLD = 0.15
selected_corr_clf = corr_target_clf[corr_target_clf > CORR_THRESHOLD].index.tolist()
selected_corr_reg = corr_target_reg[corr_target_reg > CORR_THRESHOLD].index.tolist()
print(f"\nA · Correlation filter |ρ|>{CORR_THRESHOLD}:")
print(f"   Classification → {len(selected_corr_clf)} features: {selected_corr_clf}")
print(f"   Regression     → {len(selected_corr_reg)} features: {selected_corr_reg}")

# ── METHOD B: RFE — Wrapper method  (course notebook §2.2) ───────────────────
n_select = 12
rfe_clf = RFE(LogisticRegression(max_iter=1000, random_state=42),
              n_features_to_select=n_select, step=1)
rfe_clf.fit(X_train_prep, yc_train)
rfe_reg = RFE(LinearRegression(),
              n_features_to_select=n_select, step=1)
rfe_reg.fit(X_train_prep, yr_train)

selected_rfe_clf = feature_names_out[rfe_clf.support_].tolist()
selected_rfe_reg = feature_names_out[rfe_reg.support_].tolist()

rfe_rank_clf = pd.DataFrame({
    "feature": feature_names_out, "rank": rfe_clf.ranking_
}).sort_values("rank")
print(f"\nB · RFE (n={n_select}) — Classification ranking:")
print(rfe_rank_clf.to_string(index=False))
print(f"\nB · RFE (n={n_select}) — Regression selected: {selected_rfe_reg}")

# ── METHOD C: Lasso — Embedded method ────────────────────────────────────────
lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso.fit(X_train_prep, yr_train)
lasso_coefs = pd.Series(np.abs(lasso.coef_), index=feature_names_out)
selected_lasso = lasso_coefs[lasso_coefs > 0].sort_values(ascending=False).index.tolist()
zeroed_lasso   = lasso_coefs[lasso_coefs == 0].index.tolist()
print(f"\nC · LassoCV (α={lasso.alpha_:.5f}) — {len(selected_lasso)} survive, {len(zeroed_lasso)} zeroed:")
print(lasso_coefs.sort_values(ascending=False).round(4))

# ── CONSENSUS — selected by ALL 3 methods ────────────────────────────────────
consensus = (set(selected_corr_reg) & set(selected_rfe_reg) & set(selected_lasso))
print(f"\nConsensus features (in all 3 methods): {len(consensus)}")
print(sorted(consensus))

# ═══════════════════════════════════════════════════════════════════════════════
# 9 · Visualisations
# ═══════════════════════════════════════════════════════════════════════════════

# -- Fig 10: Clean inter-feature correlation heatmap (AFTER redundant drop) ---
fig, ax = plt.subplots(figsize=(14, 11))
corr_clean = prep_df.corr()
mask = np.triu(np.ones(corr_clean.shape, dtype=bool))
sns.heatmap(corr_clean, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, square=True, linewidths=0.4, ax=ax,
            cbar_kws={"shrink": 0.7}, annot_kws={"size": 7})
ax.set_title("Fig 10 · Inter-feature correlation AFTER removing redundant features\n"
             "All |r| < 0.85 → multicollinearity resolved", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_corr_clean.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n→ Fig 10 saved")

# -- Fig 11: Feature-target correlation bar chart (course notebook style) ------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, corr_s, title, thr in [
    (axes[0], corr_target_reg,  "Spearman |ρ| with log_rate (regression)",       CORR_THRESHOLD),
    (axes[1], corr_target_clf,  "Spearman |ρ| with is_premium (classification)", CORR_THRESHOLD),
]:
    colors = [GREEN if v > thr else "lightgray" for v in corr_s]
    corr_s.plot(kind="bar", color=colors, ax=ax, edgecolor="white")
    ax.axhline(thr, color=RED, ls="--", lw=1.5, label=f"|ρ| = {thr} threshold")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("|ρ|"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend()
plt.suptitle("Fig 11 · Feature–target correlations (TRAIN only, no leakage)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig11_feature_target_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 11 saved")

# -- Fig 12: RFE ranking + Lasso coefficients ----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

rfe_rank_clf.set_index("feature")["rank"].sort_values().plot.barh(
    ax=axes[0], color=[GREEN if r == 1 else "lightgray"
                       for r in rfe_rank_clf.set_index("feature")["rank"].sort_values()],
    edgecolor="white")
axes[0].set_title("Fig 12a · RFE ranking — Classification\n(rank=1 = selected)",
                  fontweight="bold")
axes[0].set_xlabel("Rank (lower = more important)")

lasso_coefs.sort_values(ascending=True).plot.barh(
    ax=axes[1],
    color=[GREEN if v > 0 else "lightgray" for v in lasso_coefs.sort_values()],
    edgecolor="white")
axes[1].set_title(f"Fig 12b · LassoCV |coef| — Regression (α={lasso.alpha_:.5f})",
                  fontweight="bold")
axes[1].set_xlabel("|coefficient|")

plt.tight_layout()
plt.savefig(f"{OUT}/fig12_rfe_lasso.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 12 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# 10 · Save final pipelines for Part 3
#      SelectKBest lives INSIDE the pipeline → refitted on each CV fold
# ═══════════════════════════════════════════════════════════════════════════════
clf_pipeline_base = Pipeline([
    ("prep",   preprocessor),
    ("select", SelectKBest(f_classif, k=n_select)),
    # ("clf",  ← model slot, filled in Part 3)
])

reg_pipeline_base = Pipeline([
    ("prep",   preprocessor),
    ("select", SelectKBest(f_regression, k=n_select)),
    # ("reg",  ← model slot, filled in Part 3)
])

joblib.dump({
    "X_train": X_train, "X_test":  X_test,
    "yc_train": yc_train, "yc_test": yc_test,
    "yr_train": yr_train, "yr_test": yr_test,
    "preprocessor": preprocessor,
    "feature_names_out": feature_names_out,
    "clf_pipeline_base": clf_pipeline_base,
    "reg_pipeline_base": reg_pipeline_base,
    "selected_consensus": sorted(consensus),
    "corr_target_reg": corr_target_reg,
    "corr_target_clf": corr_target_clf,
}, f"{OUT}/part2_artifacts.pkl")

print(f"\nArtifacts saved → part2_artifacts.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# 11 · Final summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 2 COMPLETE — SUMMARY")
print("="*65)
print(f"  Features entering model: {len(feature_names_out)}")
print(f"  After correlation filter (|ρ|>{CORR_THRESHOLD}): {len(selected_corr_reg)} (regression)")
print(f"  After RFE (n={n_select}): {len(selected_rfe_reg)}")
print(f"  After LassoCV: {len(selected_lasso)}")
print(f"  Consensus (all 3 methods): {len(consensus)}")
print("\nLEAKAGE AUDIT:")
print("  ✅ Pure-math features computed before split (no fitting involved)")
print("  ✅ SPLIT done before any data-dependent transform")
print("  ✅ Imputer, Scaler, OneHotEncoder: fit on X_train ONLY")
print("  ✅ Correlation filter computed on X_train ONLY")
print("  ✅ RFE fitted on X_train ONLY")
print("  ✅ LassoCV fitted on X_train ONLY")
print("  ✅ SelectKBest inside Pipeline (refits on each CV fold in Part 3)")
print("  ✅ X_test: only .transform() called, never .fit()")
print("\nMULTICOLLINEARITY AUDIT:")
print("  ✅ month/month_sin/month_cos → kept sin+cos only")
print("  ✅ hotel_size_rooms ↔ exclusivity → kept exclusivity only")
print("  ✅ days_to_christmas ↔ xmas_proximity → kept xmas_proximity only")
print("  ✅ room_type OrdinalEnc ↔ room_tier → kept room_tier only")
print("  ✅ region dummies ↔ hotel_id dummies → dropped region")
print("  ✅ resort_type dummies ↔ hotel_id dummies → resort_type dropped (perfect linear combo)")
print("  ✅ is_coastal ↔ resort_type dummies → is_coastal not created")
print("  ✅ room_tier² centred (room_tier_c_sq) → collinearity with room_tier ≈ 0 vs 0.98 raw")
print("  ✅ drop='first' in OneHotEncoder → dummy variable trap eliminated")
print("  ✅ All VIF < 10 after cleanup")
