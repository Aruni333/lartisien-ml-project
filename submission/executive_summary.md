# Executive Summary

**Project:** Lartisien Collection — ML Rate Intelligence (v2)
**Author:** Minka Borec · MADA 2025
**Date:** July 2026

---

## The Question

Can we tell a Lartisien Voyager traveller when a property is entering a low-demand window — lower price, higher room availability, more space and privacy — for a given hotel, month, and room type? We answer this with two predictions: a Premium/Standard tier (a proxy for high- vs. low-demand periods) and the estimated nightly rate in EUR.

## Why This Matters

Off-season travel at a luxury property isn't just cheaper — it's a different experience: lower occupancy means more attentive staff, more privacy, and a more exclusive stay. A traveller who knows *when* a hotel enters its quiet season can get the same room for less, with more space around them.

## What We Did

We used pricing data for 24 Lartisien Collection properties across 6 room categories and 10 sample dates each — 1,440 observations after removing 2 statistical outlier hotels (median rates 2–3× the rest of the portfolio, via IQR fencing) that would have caused the model to memorise individual brands rather than learn general pricing logic. We engineered 7 interpretable features (room tier, hotel tier, cyclic seasonality, Christmas proximity, school holidays, resort type).

**Method safeguards:**
- **No leakage:** train/test split happens first (80/20, stratified); scaling and encoding are fitted inside a scikit-learn Pipeline on the training data only.
- **Independent features:** every feature was checked for collinearity (VIF); all score below the VIF=10 threshold except hotel tier (6.66), which was investigated and kept deliberately (see below).
- **Right metric for this data:** the classification target is imbalanced (~35/65), so **F1-macro** — not accuracy — is primary; for rates, **MAE** (€1,677) is the business-readable, same-unit figure, robust to a few very expensive Presidential Suite outliers.

## What We Found

- **Tier classification:** XGBoost — F1=0.805, AUC=0.905, beating a majority-class baseline by 45 points.
- **Rate prediction:** Gradient Boosting — R²=0.736, MAE=€1,677, RMSE=€2,741.
- **Top pricing drivers:** room category (44%), hotel tier (13%), Christmas proximity (11%), cyclic seasonality (~12%).

## How We Solved Overfitting

**First**, we measured it and removed it where we could: an unconstrained Decision Tree hits a perfect Train F1=1.000 but only 0.719 on test — a 0.14 gap, textbook memorisation. Capping GradBoost/XGBoost at a shallow `max_depth=3` (a deliberate regularisation choice, not an automatic search) brought that gap down to 0.06–0.07 — about a fifth the size — while still beating every simpler model on test performance.

**Then**, we recognised the remaining risk matters less than it looks: this model only has to price the 24 known Lartisien Collection hotels, never an unfamiliar brand — a narrower target than a general-purpose model. That said, "how exclusive is the hotel" is built from very few hotels per tier (10 / 8 / 6 hotels in tiers 1/2/3) and carries the highest VIF (6.66) in the feature set, so tier-based generalisation to a brand-new, unlisted property should be treated as directional, not certain.

## What We Recommend

**Use this for:** individual Lartisien Voyager members planning their own trip — finding the month a favourite property is likely to be quieter and better priced, so they can enjoy the same room for less, with more attentive service and fewer crowds.

**Don't use this for:** real-time pricing, hotels outside this 24-property portfolio, or firm quotes on Presidential Suites (highest-uncertainty tier).

## The App

Live at **lartisien-aruni999.vercel.app** — select a hotel, month, and room category for an instant tier + rate estimate (1,728 pre-computed predictions, no backend needed). In plain language: a €3,500/night estimate most likely means a real price between €2,300–€4,800, and the Premium/Standard tier call is right about 9 times in 10.
