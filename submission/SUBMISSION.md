# MADA 2025 — Final Project Submission
## Lartisien Collection: ML Rate Intelligence (v2 Rebuild)
**Student:** Minka Borec  
**Date:** July 2026

---

## Live Links

| Resource | URL |
|----------|-----|
| **Live App (React)** | https://lartisien-aruni999.vercel.app |
| **GitHub Repository** | https://github.com/Aruni333/lartisien-ml-project |

---

## Deliverables in this folder

| # | Deliverable | File | Description |
|---|------------|------|-------------|
| D1 | Analysis Report | `analysis.html` | Self-contained HTML — open in any browser, no install needed |
| D2 | Deployed Web App | Live link above | React + Vite on Vercel; Model Dashboard + Rate Predictor |
| D3 | AI Workflow Reflection | `ai_reflection.md` | Claude Code / MCP usage, verification steps, cost |
| D4 | Slide Deck | `presentation.html` | Reveal.js slides — open in browser, press → to advance |
| D5 | Executive Summary | `executive_summary.docx` | 1-page plain-language summary for non-technical reader |

---

## Submission Checklist — Self-Verification

### ☑ Dataset was approved and supports both tasks

- **Dataset v2:** 1,440 rows · 24 Lartisien-verified hotels · 6 room types · 10 sample dates
- Preliminary EDA removed 2 IQR outlier hotels (Cheval Blanc Randheli, The Brando) from 26 initial
- Both tasks use the same dataset: classification (is_premium = rate > €3,000) and regression (log_rate → EUR)
- Data file: `hotels_rates.csv` in repo root (used directly by `analysis.qmd`)
- The original Phase 1 proposal (10 hotels, 600 rows) grew to this 24-hotel, 1,440-row v2 dataset during the project, but the core research question and both targets are unchanged — approved before either build began.
- Notably, the proposal already anticipated two issues that surfaced later and were fixed in v2: it flagged that `hotel_id` would be collinear with hotel attributes and dominate feature importance (→ became the `hotel_tier` fix in §3), and that the €3,000 threshold might be imbalanced, requiring F1/ROC-AUC over accuracy (→ exactly the metric choice made in §10.1 of the report).

### ☑ Report runs end-to-end from a clean checkout; seed is set

- `analysis.html` is fully self-contained
- To reproduce from source: `quarto render analysis.qmd --to html` from repo root
- `np.random.seed(42)` at top; all sklearn estimators use `random_state=42`
- `requirements.txt` covers all Python dependencies

### ☑ No leakage — preprocessing is inside the pipeline / training split

- `StandardScaler` and `OneHotEncoder` both wrapped inside `sklearn.Pipeline` with `ColumnTransformer`
- Pipelines are fitted **only on X_train** (1,152 rows, 80% of 1,440)
- `StratifiedKFold(5)` for classification and `KFold(5)` for regression, used for cross-validated scoring — applied to training data only
- Test set (288 rows) untouched until final evaluation

### ☑ Both tasks have a full model ladder including XGBoost and a leaderboard on the test set

Both tasks were benchmarked against the same 6-model ladder (Dummy → Logistic/Ridge → Decision Tree → Random Forest → GradBoost → XGBoost), scored on the same 288-row held-out test set. Full leaderboards below.

**Classification (6 models — Dummy to XGBoost):**

| Model | CV F1 | Test F1 | AUC | Train-Test Gap |
|-------|-------|---------|-----|----------------|
| Dummy (majority) | 0.356 | 0.356 | 0.500 | 0.000 |
| Logistic (C=1) | 0.730 | 0.734 | 0.816 | 0.007 |
| Decision Tree (full) | 0.721 | 0.719 | 0.719 | 0.143 ⚠ overfits |
| Random Forest (n=300) | 0.775 | 0.762 | 0.838 | 0.129 |
| GradBoost (max_depth=3) | 0.812 | 0.804 | 0.905 | 0.069 |
| **XGBoost (tuned) ★** | **0.814** | **0.805** | **0.905** | **0.064** |

**Regression (6 models — Dummy to GradBoost):**

| Model | CV R² | Test R² | MAE € | RMSE € | Train-Test Gap |
|-------|-------|---------|-------|--------|----------------|
| Dummy (mean) | −0.001 | −0.000 | €3,321 | €5,469 | 0.000 |
| Ridge (α=1) | 0.513 | 0.570 | €2,337 | €3,592 | −0.046 |
| Decision Tree (full) | 0.382 | 0.455 | €2,163 | €3,745 | 0.390 ⚠ overfits |
| Random Forest (n=300) | 0.571 | 0.615 | €1,943 | €3,189 | 0.217 |
| XGBoost (tuned) | 0.731 | 0.724 | €1,732 | €2,822 | 0.070 |
| **GradBoost (max_depth=3) ★** | **0.725** | **0.736** | **€1,677** | **€2,741** | **0.070** |

**Decision Tree is in the ladder on purpose — to show overfitting, then show how we beat it.** Grown to full depth, it hits a near-perfect Train F1=1.000 in classification but only 0.719 on test (gap=0.143), and gap=0.390 in regression — textbook memorisation of the training rows rather than learning a real pattern. We tried to beat this two ways:

1. **Regularise the winning models.** GradBoost's `max_depth=3` (and matching `n_estimators=300`, `learning_rate=0.1`) was chosen manually as a deliberate regularisation measure — shallow trees can't memorise individual rows the way an unconstrained tree can. XGBoost's hyperparameters were found via a real `GridSearchCV` (12 combinations × 5-fold CV, training data only, in both `analysis.qmd` §5 and §6) searching the same three parameters — the search independently converged on the identical values, which is good evidence the manual GradBoost choice wasn't a lucky guess. Either way, this brought the gap down from 0.143–0.390 (Decision Tree) to 0.064–0.070, roughly a fifth the size, while still beating every simpler baseline on test score (see leaderboards above).
2. **Reconsider how much generalisation the task actually needs.** During this process we realised the app's real users — Lartisien Voyager travellers — only ever care about rates at these exact 24 known Lartisien Collection properties, not luxury hotels in general. The model therefore never has to generalise to an unseen brand; it only has to interpolate between month/room combinations for hotels it has already seen in training. That is a narrower, easier target than a general-purpose luxury-hotel pricing model, which is a legitimate part of why a 0.06–0.07 gap is credible here rather than a lucky split — though it's also why we're explicit that tier-based generalisation to a brand-new, unlisted property should be treated as directional, not certain (see the VIF note on hotel tier below).

### ☑ A Dummy baseline is in both leaderboards

- **Classification baseline:** `DummyClassifier(most_frequent)` — F1=0.356, AUC=0.500
- **Regression baseline:** `DummyRegressor(mean)` — R²=−0.000, MAE=€3,321
- Both winner models beat the Dummy by a large margin

### ☑ Overfitting addressed and discussed

Full explanation and numbers are under the Decision Tree row above — in short: measured directly via train-test gap for every model (not just asserted), reduced ~5× via `max_depth` regularisation (manual for GradBoost, `GridSearchCV`-found for XGBoost), and the remaining gap is credible because the task is scoped to 24 known hotels rather than open-world generalisation.

### ☑ Live app opens and works for someone else

- Tested in Safari private browsing — loads correctly at the Vercel URL
- 1,728 pre-computed predictions (24 hotels × 12 months × 6 rooms) bundled as static JSON — works instantly, no backend needed
- Hotel list covers 24 Lartisien-verified properties (Italy, Austria, Switzerland, France, Monaco, Turkey, Hungary, Greece, Spain, Bali, Japan, Tanzania, South Africa, French Polynesia, Caribbean)

### ☑ All five deliverables present; README links the app

README.md in repo root has a "🚀 Live App" section with the Vercel URL. All five deliverables are in the repo and this submission folder.

### ☑ AI-workflow reflection included

See `ai_reflection.md` in this folder and the repo root. Covers: Claude Code usage, MCP tools (GitHub, Playwright, Vercel, Context7), verification approach, what was checked manually, and estimated cost/effort.

---

## Key Results Summary

| Task | Best Model | Metric | Value | vs Dummy |
|------|-----------|--------|-------|----------|
| Classification | XGBoost (tuned) | Test F1-macro | **0.805** | +0.449 |
| Classification | XGBoost (tuned) | Test ROC-AUC | **0.905** | +0.405 |
| Regression | GradBoost (max_depth=3) | Test R² | **0.736** | +0.736 |
| Regression | GradBoost (max_depth=3) | Test MAE | **€1,677/night** | −€1,644 |
| Regression | GradBoost (max_depth=3) | Test RMSE | **€2,741/night** | −€2,728 |

**Top pricing drivers (v2 — plain-English feature names):**
- **Type of Room** (Deluxe → Presidential Suite) — 44% regression importance, the single biggest driver
- **How Exclusive Is the Hotel?** (Scale 1–3, replaces 23 individual brand dummies) — 13%
- **How Close to Christmas?** (smooth spike in the days before 25 Dec) — 11%
- **Which Month? / Seasonality** (two combined cyclic waves capturing summer + winter peaks) — 12%

**Honest limitations:** IQR outlier removal excluded Cheval Blanc Randheli and The Brando (extreme medians). Model does not capture real-time promotions, flash sales, or year-over-year price inflation. RMSE ≈ €2,741 means Presidential Suite predictions carry proportionally higher uncertainty.

---

## How to Run Locally

```bash
# Clone
git clone https://github.com/Aruni333/lartisien-ml-project.git
cd lartisien-ml-project

# Python dependencies
pip install -r requirements.txt

# Render the report (runs full ML pipeline in QMD code cells)
quarto render analysis.qmd --to html
quarto render presentation.qmd --to revealjs

# React app (static predictions, no backend needed)
cd frontend && npm install && npm run dev
# → http://localhost:5173
```
