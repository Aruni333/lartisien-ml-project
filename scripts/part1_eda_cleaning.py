"""
Part 1 — Understand & Clean
EDA, cleaning, scaling for Lartisien ML project.
Outputs: hotels_rates_clean.csv + all PNG plots.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
BLUE = "#2d4a8a"
OUT  = "/Users/minkaborec/Desktop/MADA final project"

# ═══════════════════════════════════════════════════════════════════════════════
# 0 · Load
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(f"{OUT}/hotels_rates.csv")
print("=" * 65)
print("SECTION 0 — LOAD")
print("=" * 65)
print(f"Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1 · Inspect — shape, types, quirks
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SECTION 1 — INSPECT")
print("=" * 65)

cat_cols  = ["hotel_id", "country", "region", "resort_type", "room_type"]
num_cols  = ["hotel_size_rooms", "month", "is_school_holiday",
             "is_weekend", "days_to_christmas", "rate_eur_per_night"]
bin_cols  = ["is_school_holiday", "is_weekend"]
target_r  = "rate_eur_per_night"

# Cardinalities
print("\nCategorical cardinalities:")
for c in cat_cols:
    print(f"  {c:25s}: {df[c].nunique()} unique → {df[c].unique()[:4]}")

# Missing values
print("\nMissing values:")
miss = df.isnull().sum()
print(miss[miss > 0])
print(f"  → only hotel_size_rooms has NaN ({miss['hotel_size_rooms']} rows = "
      f"{miss['hotel_size_rooms']/len(df)*100:.1f}%)")

# Duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Numeric summary
print("\nNumeric summary:")
print(df[num_cols].describe().round(1))

# ═══════════════════════════════════════════════════════════════════════════════
# 2 · Derive target columns before EDA (needed for target analysis plots)
# ═══════════════════════════════════════════════════════════════════════════════
df["is_premium"] = (df[target_r] > 3000).astype(int)

# ═══════════════════════════════════════════════════════════════════════════════
# 3 · EDA — Distributions
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SECTION 3 — EDA: DISTRIBUTIONS")
print("=" * 65)

# -- Fig 1: Rate distribution (raw + log) ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df[target_r], bins=40, color=BLUE, edgecolor="white", alpha=0.85)
axes[0].set_title("Rate distribution (raw €)", fontweight="bold")
axes[0].set_xlabel("€ per night"); axes[0].set_ylabel("Count")
axes[0].axvline(3000, color="crimson", lw=2, ls="--", label="€3 000 threshold")
axes[0].legend()

axes[1].hist(np.log1p(df[target_r]), bins=40, color="#5a7ac7", edgecolor="white", alpha=0.85)
axes[1].set_title("Rate distribution (log scale)", fontweight="bold")
axes[1].set_xlabel("log(€ per night + 1)"); axes[1].set_ylabel("Count")
plt.suptitle("Fig 1 · Nightly rate distributions", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_rate_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 1 saved")

# -- Fig 2: Categorical feature bar charts -----------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax_list = axes.flatten()
for i, col in enumerate(["resort_type", "region", "room_type", "hotel_id"]):
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, order=order, palette="Blues_d",
                  ax=ax_list[i])
    ax_list[i].set_title(f"{col}", fontweight="bold")
    ax_list[i].set_xlabel("Count"); ax_list[i].set_ylabel("")
plt.suptitle("Fig 2 · Categorical feature distributions", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_categorical_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 2 saved")

# -- Fig 3: Numeric feature histograms ----------------------------------------
num_plot = ["hotel_size_rooms", "month", "days_to_christmas"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, num_plot):
    ax.hist(df[col].dropna(), bins=20, color=BLUE, edgecolor="white", alpha=0.85)
    ax.set_title(col, fontweight="bold"); ax.set_xlabel("Value"); ax.set_ylabel("Count")
plt.suptitle("Fig 3 · Numeric feature distributions", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_numeric_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 3 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# 4 · Target Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SECTION 4 — TARGET ANALYSIS")
print("=" * 65)

# Classification target
vc = df["is_premium"].value_counts()
print(f"\nClassification target (is_premium):\n{vc}")
print(f"  Class imbalance ratio: {vc[1]/vc[0]:.2f}  → use F1 + ROC-AUC (not accuracy)")

# Regression target
print(f"\nRegression target (rate_eur_per_night):")
print(df[target_r].describe().round(0))
sk = stats.skew(df[target_r])
print(f"  Skewness: {sk:.2f}  → right-skewed; log transform recommended for regression")

# -- Fig 4: Target panels -----------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Class balance pie
labels = ["Standard (≤€3k)", "Premium (>€3k)"]
sizes  = [vc[0], vc[1]]
axes[0].pie(sizes, labels=labels, autopct="%1.1f%%",
            colors=["#4a7fc1","#c14a4a"], startangle=90)
axes[0].set_title("Classification target balance", fontweight="bold")

# Rate by room type
med_order = df.groupby("room_type")[target_r].median().sort_values().index
sns.boxplot(data=df, x=target_r, y="room_type", order=med_order,
            palette="Blues_d", ax=axes[1])
axes[1].axvline(3000, color="crimson", lw=2, ls="--", label="€3k")
axes[1].set_title("Rate by room type", fontweight="bold")
axes[1].set_xlabel("€ per night"); axes[1].set_ylabel("")
axes[1].legend(fontsize=8)

# Rate by month (seasonality)
monthly = df.groupby("month")[target_r].median()
axes[2].bar(monthly.index, monthly.values, color=BLUE, alpha=0.85)
axes[2].axhline(3000, color="crimson", lw=2, ls="--")
axes[2].set_title("Median rate by month", fontweight="bold")
axes[2].set_xlabel("Month"); axes[2].set_ylabel("Median € per night")
axes[2].set_xticks(range(1,13))

plt.suptitle("Fig 4 · Target analysis", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_target_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 4 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# 5 · Correlation Study
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SECTION 5 — CORRELATION STUDY")
print("=" * 65)

# Encode categoricals ordinally for correlation heatmap
df_enc = df.copy()
room_order = ["Deluxe Room","Superior Room","Junior Suite",
              "Suite","Grand Suite","Presidential Suite"]
df_enc["room_type_enc"]   = pd.Categorical(df["room_type"], categories=room_order, ordered=True).codes
df_enc["resort_type_enc"] = pd.Categorical(df["resort_type"]).codes
df_enc["hotel_id_enc"]    = pd.Categorical(df["hotel_id"]).codes

corr_cols = ["hotel_size_rooms","month","is_school_holiday","is_weekend",
             "days_to_christmas","room_type_enc","resort_type_enc",
             "hotel_id_enc","rate_eur_per_night","is_premium"]

corr = df_enc[corr_cols].corr()

# -- Fig 5: Correlation heatmap ----------------------------------------------
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Fig 5 · Pearson correlation matrix", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 5 saved")

# Print top correlations with target
print("\nTop correlations with rate_eur_per_night:")
target_corr = corr["rate_eur_per_night"].drop("rate_eur_per_night").abs().sort_values(ascending=False)
print(target_corr.round(3))

# ═══════════════════════════════════════════════════════════════════════════════
# 6 · Feature–Target Relationships
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SECTION 6 — FEATURE–TARGET RELATIONSHIPS")
print("=" * 65)

# -- Fig 6: Rate by key categoricals -----------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# By hotel
hotel_med = df.groupby("hotel_id")[target_r].median().sort_values()
axes[0,0].barh(hotel_med.index, hotel_med.values, color=BLUE, alpha=0.85)
axes[0,0].axvline(3000, color="crimson", lw=2, ls="--", label="€3k")
axes[0,0].set_title("Median rate by hotel", fontweight="bold")
axes[0,0].set_xlabel("Median € per night")
axes[0,0].legend(fontsize=8)

# By resort type
sns.boxplot(data=df, x="resort_type", y=target_r, palette="Blues_d", ax=axes[0,1])
axes[0,1].axhline(3000, color="crimson", lw=2, ls="--")
axes[0,1].set_title("Rate distribution by resort type", fontweight="bold")
axes[0,1].set_ylabel("€ per night"); axes[0,1].set_xlabel("")

# By region
sns.boxplot(data=df, x="region", y=target_r, palette="Blues_d", ax=axes[1,0])
axes[1,0].axhline(3000, color="crimson", lw=2, ls="--")
axes[1,0].set_title("Rate distribution by region", fontweight="bold")
axes[1,0].set_ylabel("€ per night"); axes[1,0].set_xlabel("")

# Seasonality: median rate per month, coloured by school holiday
monthly_hol   = df[df["is_school_holiday"]==1].groupby("month")[target_r].median()
monthly_nohol = df[df["is_school_holiday"]==0].groupby("month")[target_r].median()
months = range(1, 13)
axes[1,1].plot(monthly_hol.reindex(months),   "o-", color="crimson",  label="School holiday")
axes[1,1].plot(monthly_nohol.reindex(months), "o-", color=BLUE,       label="Non-holiday")
axes[1,1].axhline(3000, color="gray", lw=1, ls="--")
axes[1,1].set_title("Median rate by month & school holiday", fontweight="bold")
axes[1,1].set_xlabel("Month"); axes[1,1].set_ylabel("Median € per night")
axes[1,1].set_xticks(range(1,13))
axes[1,1].legend()

plt.suptitle("Fig 6 · Feature–target relationships", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_feature_target.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 6 saved")

# -- Fig 7: days_to_christmas vs rate (scatter) -------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(df["days_to_christmas"], df[target_r],
                     c=df["is_premium"], cmap="RdYlBu", alpha=0.5, s=20)
ax.axhline(3000, color="crimson", lw=1.5, ls="--", label="€3k threshold")
ax.set_xlabel("Days to Christmas"); ax.set_ylabel("€ per night")
ax.set_title("Fig 7 · days_to_christmas vs rate  (blue=Premium)", fontweight="bold")
plt.colorbar(scatter, ax=ax, label="is_premium")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_days_xmas_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 7 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# 7 · Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SECTION 7 — CLEANING")
print("=" * 65)

df_clean = df.copy()

# 7a. Missing values — hotel_size_rooms (2% missing)
# Strategy: median imputation grouped by hotel (each hotel has a fixed size)
# This is unambiguous: hotel_size is constant within a hotel, so the median
# of the non-null values for that hotel is always the exact true value.
fill_vals = df_clean.groupby("hotel_id")["hotel_size_rooms"].transform("median")
df_clean["hotel_size_rooms"] = df_clean["hotel_size_rooms"].fillna(fill_vals)
print(f"Missing after imputation: {df_clean['hotel_size_rooms'].isna().sum()}")

# 7b. Outlier detection — rate_eur_per_night
# Use IQR method. Note: extreme rates ARE legitimate for this dataset
# (Presidential Suites at ultra-luxury resorts can exceed €10k).
# We inspect only — no rows removed, but we flag them.
Q1, Q3 = df_clean[target_r].quantile([0.25, 0.75])
IQR = Q3 - Q1
upper_fence = Q3 + 3 * IQR   # use 3× (Tukey's extreme outlier rule) for luxury data
outliers = df_clean[df_clean[target_r] > upper_fence]
print(f"\nIQR extreme outlier fence (3×IQR): €{upper_fence:,.0f}")
print(f"Flagged rows: {len(outliers)}")
print(outliers[["hotel_id","room_type","month","rate_eur_per_night"]])
print("→ Decision: RETAIN all — these are legitimate Presidential Suite rates.")

df_clean["rate_flag_extreme"] = (df_clean[target_r] > upper_fence).astype(int)

# 7c. Scaling — three strategies documented
# StandardScaler  → for models sensitive to scale (SVM, NN, linear).
# MinMaxScaler    → for models needing [0,1] bounded input.
# RobustScaler    → for rate_eur_per_night (right-skewed; robust to the extremes).
# We add log-transformed rate as an alternative regression target.
df_clean["log_rate"] = np.log1p(df_clean[target_r])

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scale_cols = ["hotel_size_rooms", "days_to_christmas"]

ss = StandardScaler()
df_clean[[c + "_std" for c in scale_cols]] = ss.fit_transform(df_clean[scale_cols])

rs = RobustScaler()
df_clean["rate_robust"] = rs.fit_transform(df_clean[[target_r]])

print("\nScaling added: *_std columns (StandardScaler) + rate_robust (RobustScaler).")
print(f"log_rate stats: min={df_clean['log_rate'].min():.2f}, "
      f"max={df_clean['log_rate'].max():.2f}, "
      f"skew={stats.skew(df_clean['log_rate']):.2f}")

# -- Fig 8: Before/after scaling + log transform on rate ---------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(df_clean[target_r], bins=35, color=BLUE, edgecolor="white", alpha=0.85)
axes[0].set_title("Raw rate", fontweight="bold"); axes[0].set_xlabel("€ per night")
axes[0].annotate(f"skew={stats.skew(df_clean[target_r]):.2f}",
                 xy=(0.98,0.92), xycoords="axes fraction", ha="right", color="crimson")

axes[1].hist(df_clean["log_rate"], bins=35, color="#5a7ac7", edgecolor="white", alpha=0.85)
axes[1].set_title("log(rate + 1)", fontweight="bold"); axes[1].set_xlabel("log scale")
axes[1].annotate(f"skew={stats.skew(df_clean['log_rate']):.2f}",
                 xy=(0.98,0.92), xycoords="axes fraction", ha="right", color="crimson")

axes[2].hist(df_clean["rate_robust"], bins=35, color="#3a6e3a", edgecolor="white", alpha=0.85)
axes[2].set_title("RobustScaler(rate)", fontweight="bold"); axes[2].set_xlabel("robust units")
axes[2].annotate(f"skew={stats.skew(df_clean['rate_robust']):.2f}",
                 xy=(0.98,0.92), xycoords="axes fraction", ha="right", color="crimson")

plt.suptitle("Fig 8 · Rate scaling comparison", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_scaling.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 8 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# 8 · Final clean dataset
# ═══════════════════════════════════════════════════════════════════════════════
# Save a clean baseline (no engineered/scaled cols) for the modelling steps
core_cols = ["hotel_id","country","region","resort_type","room_type",
             "hotel_size_rooms","month","is_school_holiday","is_weekend",
             "days_to_christmas","rate_eur_per_night","is_premium",
             "log_rate","rate_flag_extreme"]
df_clean[core_cols].to_csv(f"{OUT}/hotels_rates_clean.csv", index=False)
print(f"\nClean dataset saved: {df_clean[core_cols].shape}")
print("\nFinal column types:")
print(df_clean[core_cols].dtypes)
print("\nMissing values in clean dataset:")
print(df_clean[core_cols].isnull().sum())

print("\n" + "=" * 65)
print("PART 1 COMPLETE — 8 figures + hotels_rates_clean.csv produced.")
print("=" * 65)
