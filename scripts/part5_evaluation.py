"""
Part E — Honest Leaderboard & Business Evaluation
==================================================
Rules from notebook §3.1 + §3.5 + §3.6:
  • Report TEST performance only — not train (train scores inflate all estimates)
  • Keep the Dummy baseline in every table — quantify real gain over chance
  • Call out overfitting explicitly when train >> test
  • Do not oversell — state confidence intervals from CV
  • Classification: accuracy, precision/recall/F1 per class, ROC-AUC, confusion matrix
  • Regression: R², MAE, RMSE (log scale + EUR scale), predicted-vs-actual plot
  • Feature importance: what does the model actually use?
  • Business interpretation: what can and cannot this model be trusted for?

All models evaluated are the FULL PIPELINES saved from Part D
(preprocessor + model in one object, accepting raw X_test directly).
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
BLUE   = "#2d4a8a"
GREEN  = "#2d6a2d"
RED    = "#c0392b"
GOLD   = "#d4a017"
ORANGE = "#e67e22"
GRAY   = "#7f8c8d"
OUT    = "/Users/minkaborec/Desktop/MADA final project"

# ═══════════════════════════════════════════════════════════════════════════════
# 0 · Load data + all saved pipelines
# ═══════════════════════════════════════════════════════════════════════════════
art   = joblib.load(f"{OUT}/part2_artifacts.pkl")
art4  = joblib.load(f"{OUT}/part4_leaderboards.pkl")
clf_pipe = joblib.load(f"{OUT}/classifier.pkl")   # GradBoost tuned
reg_pipe = joblib.load(f"{OUT}/regressor.pkl")    # GradBoost tuned

X_train  = art["X_train"];   X_test  = art["X_test"]
yc_train = art["yc_train"];  yc_test = art["yc_test"]
yr_train = art["yr_train"];  yr_test = art["yr_test"]

cat_nominal = ["hotel_id"]
num_feats   = ["xmas_proximity"]
passthrough = ["month_sin","month_cos","is_school_holiday","is_weekend",
               "room_tier","room_tier_c_sq","room_x_holiday","holiday_x_weekend"]

def make_pre():
    return ColumnTransformer([
        ("num",  Pipeline([("scale", StandardScaler())]),              num_feats),
        ("nom",  OneHotEncoder(handle_unknown="ignore",
                               sparse_output=False, drop="first"),    cat_nominal),
        ("pass", "passthrough",                                        passthrough),
    ], verbose_feature_names_out=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf  = KFold(n_splits=5,           shuffle=True, random_state=42)

# Rebuild all other pipelines (already fitted inside — just for predictions)
all_clf = {
    "Dummy (majority)":     Pipeline([("pre",make_pre()),("clf",DummyClassifier(strategy="most_frequent",random_state=42))]),
    "Logistic (C=1)":       Pipeline([("pre",make_pre()),("clf",LogisticRegression(C=1,max_iter=2000,class_weight="balanced",random_state=42))]),
    "Random Forest":         Pipeline([("pre",make_pre()),("clf",RandomForestClassifier(n_estimators=300,max_depth=6,class_weight="balanced",random_state=42,n_jobs=-1))]),
    "GradBoost (default)":  Pipeline([("pre",make_pre()),("clf",GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.08,subsample=0.8,random_state=42))]),
    "GradBoost (tuned) ★":  clf_pipe,
}
all_reg = {
    "Dummy (mean)":          Pipeline([("pre",make_pre()),("reg",DummyRegressor(strategy="mean"))]),
    "Ridge (α=1)":           Pipeline([("pre",make_pre()),("reg",Ridge(alpha=1.0))]),
    "Random Forest":          Pipeline([("pre",make_pre()),("reg",RandomForestRegressor(n_estimators=300,max_depth=8,random_state=42,n_jobs=-1))]),
    "GradBoost (default)":   Pipeline([("pre",make_pre()),("reg",GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.08,subsample=0.8,random_state=42))]),
    "GradBoost (tuned) ★":   reg_pipe,
}

# Fit all (tuned ones are already fitted — re-fitting on X_train is idempotent here)
for name, pipe in all_clf.items():
    pipe.fit(X_train, yc_train)
for name, pipe in all_reg.items():
    pipe.fit(X_train, yr_train)

# ═══════════════════════════════════════════════════════════════════════════════
# 1 · CLASSIFICATION LEADERBOARD — full metrics on test set
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CLASSIFICATION LEADERBOARD — Test Set (n=120, stratified hold-out)")
print(f"{'='*70}")

clf_rows = []
for name, pipe in all_clf.items():
    yp    = pipe.predict(X_test)
    yprob = pipe.predict_proba(X_test)[:, 1]
    cm    = confusion_matrix(yc_test, yp)
    tn, fp, fn, tp = cm.ravel()
    clf_rows.append({
        "Model":       name,
        "Accuracy":    round(accuracy_score(yc_test, yp), 4),
        "Precision-0": round(precision_score(yc_test, yp, pos_label=0, zero_division=0), 4),
        "Recall-0":    round(recall_score(yc_test, yp,    pos_label=0, zero_division=0), 4),
        "F1-0":        round(f1_score(yc_test, yp,        pos_label=0, zero_division=0), 4),
        "Precision-1": round(precision_score(yc_test, yp, pos_label=1, zero_division=0), 4),
        "Recall-1":    round(recall_score(yc_test, yp,    pos_label=1, zero_division=0), 4),
        "F1-1":        round(f1_score(yc_test, yp,        pos_label=1, zero_division=0), 4),
        "F1-macro":    round(f1_score(yc_test, yp, average="macro"), 4),
        "ROC-AUC":     round(roc_auc_score(yc_test, yprob), 4),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    })

clf_lb = pd.DataFrame(clf_rows)
print(clf_lb[["Model","Accuracy","F1-macro","ROC-AUC","F1-0","F1-1","TP","TN","FP","FN"]].to_string(index=False))

# Full classification report for the winner
print(f"\n{'='*70}")
print(f"WINNER CLASSIFICATION REPORT: GradBoost (tuned)")
print(f"{'='*70}")
yp_best = clf_pipe.predict(X_test)
print(classification_report(yc_test, yp_best, target_names=["Standard (0)","Premium (1)"]))

# CV confidence intervals for the winner
cv_f1  = cross_val_score(clf_pipe, X_train, yc_train, cv=skf, scoring="f1_macro")
cv_auc = cross_val_score(clf_pipe, X_train, yc_train, cv=skf, scoring="roc_auc")
print(f"CV F1-macro:  {cv_f1.mean():.3f} ± {cv_f1.std():.3f}  [95% CI: {cv_f1.mean()-1.96*cv_f1.std():.3f}, {cv_f1.mean()+1.96*cv_f1.std():.3f}]")
print(f"CV ROC-AUC:   {cv_auc.mean():.3f} ± {cv_auc.std():.3f}  [95% CI: {cv_auc.mean()-1.96*cv_auc.std():.3f}, {cv_auc.mean()+1.96*cv_auc.std():.3f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 2 · REGRESSION LEADERBOARD — full metrics on test set
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("REGRESSION LEADERBOARD — Test Set (n=120, held-out)")
print(f"{'='*70}")

reg_rows = []
for name, pipe in all_reg.items():
    yp  = pipe.predict(X_test)
    yp_eur  = np.expm1(yp)
    yt_eur  = np.expm1(yr_test.values)
    reg_rows.append({
        "Model":        name,
        "Test R²":      round(r2_score(yr_test, yp), 4),
        "Test RMSE\n(log)": round(mean_squared_error(yr_test, yp)**0.5, 4),
        "Test MAE\n(log)":  round(mean_absolute_error(yr_test, yp), 4),
        "Test RMSE\n(€)":   round(mean_squared_error(yt_eur, yp_eur)**0.5, 0),
        "Test MAE\n(€)":    round(mean_absolute_error(yt_eur, yp_eur), 0),
        "Within 20%":   f"{np.mean(np.abs((yp_eur - yt_eur)/yt_eur) <= 0.20)*100:.0f}%",
    })

reg_lb = pd.DataFrame(reg_rows)
print(reg_lb.to_string(index=False))

cv_r2  = cross_val_score(reg_pipe, X_train, yr_train, cv=kf, scoring="r2")
cv_rmse= -cross_val_score(reg_pipe, X_train, yr_train, cv=kf, scoring="neg_root_mean_squared_error")
print(f"\nWinner CV R²:   {cv_r2.mean():.3f} ± {cv_r2.std():.3f}  [95% CI: {cv_r2.mean()-1.96*cv_r2.std():.3f}, {cv_r2.mean()+1.96*cv_r2.std():.3f}]")
print(f"Winner CV RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} (log scale)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3 · FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
feature_names = list(clf_pipe[:-1].get_feature_names_out())

clf_importances = pd.Series(
    clf_pipe[-1].feature_importances_, index=feature_names
).sort_values(ascending=False)

reg_importances = pd.Series(
    reg_pipe[-1].feature_importances_, index=feature_names
).sort_values(ascending=False)

print(f"\n{'='*70}")
print("FEATURE IMPORTANCE — Classification (top 12)")
print(f"{'='*70}")
print(clf_importances.head(12).round(4))

print(f"\n{'='*70}")
print("FEATURE IMPORTANCE — Regression (top 12)")
print(f"{'='*70}")
print(reg_importances.head(12).round(4))

# ═══════════════════════════════════════════════════════════════════════════════
# 4 · OVERFITTING SUMMARY (train vs test, for the record)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("OVERFITTING AUDIT — Train vs CV vs Test (honest)")
print(f"{'='*70}")
print("Classification (F1-macro):")
for name, pipe in all_clf.items():
    train_f1 = f1_score(yc_train, pipe.predict(X_train), average="macro")
    test_f1  = f1_score(yc_test,  pipe.predict(X_test),  average="macro")
    gap = train_f1 - test_f1
    flag = " ⚠️  overfit" if gap > 0.15 else " ✅" if gap < 0.05 else ""
    print(f"  {name:30s}: train={train_f1:.3f}  test={test_f1:.3f}  gap={gap:+.3f}{flag}")
print("\nRegression (R²):")
for name, pipe in all_reg.items():
    train_r2 = r2_score(yr_train, pipe.predict(X_train))
    test_r2  = r2_score(yr_test,  pipe.predict(X_test))
    gap = train_r2 - test_r2
    flag = " ⚠️  overfit" if gap > 0.15 else " ✅" if gap < 0.05 else ""
    print(f"  {name:30s}: train={train_r2:.3f}  test={test_r2:.3f}  gap={gap:+.3f}{flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5 · FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

# -- Fig 18: Full classification leaderboard (all metrics) --------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models = clf_lb["Model"]
x = np.arange(len(models))

# F1-macro
def _cclf(m):
    if "tuned" in m: return GOLD
    if "default" in m.lower() or "Forest" in m: return GREEN
    if "Logistic" in m: return BLUE
    return GRAY
colors = [_cclf(m) for m in models]
axes[0].bar(x, clf_lb["F1-macro"], color=colors, edgecolor="white", alpha=0.9)
axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
axes[0].set_ylabel("F1-macro"); axes[0].set_ylim(0, 1.05)
axes[0].set_title("F1-macro (both classes weighted)", fontweight="bold")
axes[0].axhline(clf_lb.loc[clf_lb["Model"]=="Dummy (majority)","F1-macro"].values[0],
                color=GRAY, ls=":", lw=1.5, label="Dummy baseline")
axes[0].legend(fontsize=8)
for i, v in enumerate(clf_lb["F1-macro"]):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

# ROC-AUC
axes[1].bar(x, clf_lb["ROC-AUC"], color=colors, edgecolor="white", alpha=0.9)
axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
axes[1].set_ylabel("ROC-AUC"); axes[1].set_ylim(0, 1.05)
axes[1].set_title("ROC-AUC (threshold-free)", fontweight="bold")
axes[1].axhline(0.5, color=GRAY, ls=":", lw=1.5, label="Random AUC=0.5")
axes[1].legend(fontsize=8)
for i, v in enumerate(clf_lb["ROC-AUC"]):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

# Accuracy
axes[2].bar(x, clf_lb["Accuracy"], color=colors, edgecolor="white", alpha=0.9)
axes[2].set_xticks(x); axes[2].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
axes[2].set_ylabel("Accuracy"); axes[2].set_ylim(0, 1.05)
axes[2].set_title("Accuracy (note: majority-class bias)", fontweight="bold")
axes[2].axhline(0.65, color=GRAY, ls=":", lw=1.5, label="Majority-class ceiling (0.65)")
axes[2].legend(fontsize=8)
for i, v in enumerate(clf_lb["Accuracy"]):
    axes[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

plt.suptitle("Fig 18 · Classification Leaderboard — All Test Metrics\n"
             "Gold = winner (GradBoost tuned)", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig18_clf_full_leaderboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n→ Fig 18 saved")

# -- Fig 19: Confusion matrices (all classifiers) -----------------------------
fig, axes = plt.subplots(1, len(all_clf), figsize=(4*len(all_clf), 4))
for ax, (name, pipe) in zip(axes, all_clf.items()):
    yp = pipe.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        yc_test, yp, display_labels=["Std", "Prem"],
        cmap="Blues", ax=ax, colorbar=False)
    short = name.replace(" (majority)","").replace(" (default)","").replace(" (tuned) ★","★")
    ax.set_title(f"{short}\nF1={f1_score(yc_test,yp,average='macro'):.3f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Fig 19 · Confusion Matrices — All Classifiers on Test Set",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig19_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 19 saved")

# -- Fig 20: ROC curves all classifiers on one plot ---------------------------
fig, ax = plt.subplots(figsize=(8, 6))
model_colors = [GRAY, BLUE, GREEN, ORANGE, GOLD]
for (name, pipe), col in zip(all_clf.items(), model_colors):
    yprob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(yc_test, yprob)
    auc = roc_auc_score(yc_test, yprob)
    lw = 2.5 if "tuned" in name else 1.2
    ax.plot(fpr, tpr, color=col, lw=lw, label=f"{name}  (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Fig 20 · ROC Curves — All Classifiers on Test Set\n"
             "Higher and left = better discrimination", fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/fig20_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 20 saved")

# -- Fig 21: Regression leaderboard ------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
reg_models = reg_lb["Model"]
xr = np.arange(len(reg_models))
def _creg(m):
    if "tuned" in m: return GOLD
    if "default" in m.lower() or "Forest" in m: return GREEN
    if "Ridge" in m or "Linear" in m: return BLUE
    return GRAY
rcolors = [_creg(m) for m in reg_models]

for ax, col, label, ylbl in [
    (axes[0], "Test R²",       "R² (higher = better)",       "R²"),
    (axes[1], "Test RMSE\n(log)", "RMSE log_rate (lower = better)", "RMSE (log scale)"),
    (axes[2], "Test MAE\n(log)",  "MAE log_rate (lower = better)",  "MAE (log scale)"),
]:
    vals = reg_lb[col].clip(lower=0)
    axes_obj = [axes[0], axes[1], axes[2]][[0,1,2][list([axes[0],axes[1],axes[2]]).index(ax)]]
    ax.bar(xr, vals, color=rcolors, edgecolor="white", alpha=0.9)
    ax.set_xticks(xr); ax.set_xticklabels(reg_models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylbl); ax.set_title(label, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{reg_lb[col].iloc[i]:.3f}", ha="center", fontsize=8)

axes[0].axhline(0, color=GRAY, ls=":", lw=1.5)
plt.suptitle("Fig 21 · Regression Leaderboard — All Test Metrics\n"
             "Gold = winner (GradBoost tuned)", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig21_reg_full_leaderboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 21 saved")

# -- Fig 22: Predicted vs Actual — all regressors in one panel ----------------
fig, axes = plt.subplots(1, len(all_reg), figsize=(4*len(all_reg), 4))
mn, mx = yr_test.values.min(), yr_test.values.max()
for ax, (name, pipe) in zip(axes, all_reg.items()):
    yp = pipe.predict(X_test)
    r2 = r2_score(yr_test, yp)
    ax.scatter(yr_test.values, yp, alpha=0.5, s=20,
               color=GOLD if "tuned" in name else BLUE)
    ax.plot([mn,mx],[mn,mx], color=RED, ls="--", lw=1.2)
    short = name.replace(" (mean)","").replace(" (default)","").replace(" (tuned) ★","★")
    ax.set_title(f"{short}\nR²={r2:.3f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Actual log_rate"); ax.set_ylabel("Predicted")
plt.suptitle("Fig 22 · Predicted vs Actual log_rate — All Regressors on Test Set\n"
             "Dots on the dashed line = perfect prediction", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig22_pred_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 22 saved")

# -- Fig 23: Feature importance (side by side clf + reg) ----------------------
top_n = 12
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
clf_importances.head(top_n).sort_values().plot.barh(ax=axes[0], color=BLUE, edgecolor="white", alpha=0.85)
axes[0].set_title(f"Fig 23a · Feature Importance — Classification\n(GradBoost tuned, top {top_n})", fontweight="bold")
axes[0].set_xlabel("Importance score")

reg_importances.head(top_n).sort_values().plot.barh(ax=axes[1], color=GREEN, edgecolor="white", alpha=0.85)
axes[1].set_title(f"Fig 23b · Feature Importance — Regression\n(GradBoost tuned, top {top_n})", fontweight="bold")
axes[1].set_xlabel("Importance score")

plt.tight_layout()
plt.savefig(f"{OUT}/fig23_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Fig 23 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# 6 · BUSINESS ACCURACY — how many EUR are we off by?
# ═══════════════════════════════════════════════════════════════════════════════
yp_reg_eur = np.expm1(reg_pipe.predict(X_test))
yt_eur     = np.expm1(yr_test.values)
abs_err_eur = np.abs(yp_reg_eur - yt_eur)
pct_err     = np.abs((yp_reg_eur - yt_eur) / yt_eur)

print(f"\n{'='*70}")
print("BUSINESS ACCURACY — Rate Prediction in EUR (test set, n=120)")
print(f"{'='*70}")
print(f"  Mean Absolute Error:      €{abs_err_eur.mean():,.0f}/night")
print(f"  Median Absolute Error:    €{np.median(abs_err_eur):,.0f}/night")
print(f"  Max Absolute Error:       €{abs_err_eur.max():,.0f}/night")
print(f"  Mean % Error:             {pct_err.mean()*100:.1f}%")
print(f"  Within ±10% of actual:    {(pct_err<=0.10).mean()*100:.0f}% of predictions")
print(f"  Within ±20% of actual:    {(pct_err<=0.20).mean()*100:.0f}% of predictions")
print(f"  Within ±30% of actual:    {(pct_err<=0.30).mean()*100:.0f}% of predictions")

print(f"\n{'='*70}")
print("EXECUTIVE SUMMARY")
print(f"{'='*70}")
print(f"""
PROJECT: Lartisien Collection — Luxury Hotel Rate Intelligence App
DATASET: 600 rows × 10 features (10 hotels × 6 room types × 10 dates)

CLASSIFICATION (is_premium > €3,000/night):
  Winning model:  GradientBoostingClassifier (n_estimators=300, max_depth=4, lr=0.1)
  Test F1-macro:  {f1_score(yc_test, clf_pipe.predict(X_test), average='macro'):.3f}   (Dummy baseline: 0.394)
  Test ROC-AUC:   {roc_auc_score(yc_test, clf_pipe.predict_proba(X_test)[:,1]):.3f}   (Random: 0.500)
  Gain over baseline: +{f1_score(yc_test,clf_pipe.predict(X_test),average='macro')-0.394:.3f} F1 points
  Honest note: train F1 = 1.000 → model memorises training set (small dataset effect).
               CV and test scores are stable — generalisation is real but bounded by n=480 train.

REGRESSION (log_rate → EUR/night):
  Winning model:  GradientBoostingRegressor (n_estimators=300, max_depth=4, lr=0.1)
  Test R²:        {r2_score(yr_test, reg_pipe.predict(X_test)):.3f}   (Dummy baseline: -0.002)
  Test RMSE:      {mean_squared_error(yr_test, reg_pipe.predict(X_test))**0.5:.3f} log units ≈ €{abs_err_eur.mean():,.0f} mean absolute error
  Within ±20%:    {(pct_err<=0.20).mean()*100:.0f}% of test predictions
  Honest note: train R² = 0.976, test R² = 0.930 — meaningful overfit gap of 0.046.
               Acceptable for a 480-row training set; more data would close it.

TOP FEATURES (both tasks):
  room_tier — room category is the dominant pricing signal
  hotel_id  — individual hotel identity (brand, location premium)
  month_sin — seasonality (cyclic encoding)
  xmas_proximity — Christmas surge effect

WHAT THE MODEL CANNOT BE TRUSTED FOR:
  • Predicting rates for NEW hotels not in the 10 training properties
  • Year-over-year price trends (no temporal dimension in the dataset)
  • Real-time rate fluctuations (last-minute deals, promotions, events)
  • Currency-adjusted rates outside EUR
""")
