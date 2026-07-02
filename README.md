# Lartisien Collection — ML Rate Intelligence

**MADA 2025 Final Project · Minka Borec**

> Can machine learning tell a Lartisien Voyager traveller when a property is likely to be low-demand, lower-priced, and more available — so they can pick the month that gives them the same room for less money, and the property to themselves?

---

## 🚀 Live App (D2)

**[▶ Launch the Rate Intelligence App](https://lartisien-aruni999.vercel.app)**

Select one of 24 Lartisien Collection hotels, a month of arrival, and a room category → instant tier classification (Premium / Standard) + estimated EUR rate.

---

## Project Summary

| | |
|--|--|
| **Dataset v2** | 1,440 rows · 24 Lartisien-verified hotels · 6 room types · 10 sample dates |
| **EDA** | Preliminary IQR outlier detection removed 2 extreme hotels from 26 |
| **Classification** | Is the booking Premium (> €3,000/night)? |
| **Regression** | What is the estimated nightly rate (EUR)? |
| **Classification winner** | XGBoost — F1-macro = **0.805** · ROC-AUC = **0.905** |
| **Regression winner** | GradBoost — R² = **0.736** · MAE = **€1,677/night** · RMSE = **€2,741** |

---

## Deliverables

| # | Deliverable | File |
|---|------------|------|
| D1 | Reproducible analysis report (Quarto → HTML) | [`analysis.qmd`](analysis.qmd) · [`analysis.html`](analysis.html) |
| D2 | Deployed React app | [`frontend/`](frontend/) · [live link](https://lartisien-aruni999.vercel.app) |
| D3 | AI workflow reflection | [`ai_reflection.md`](ai_reflection.md) |
| D4 | Reveal.js slide deck | [`presentation.qmd`](presentation.qmd) · [`presentation.html`](presentation.html) |
| D5 | Executive summary | [`executive_summary.md`](executive_summary.md) |

---

## Repository Structure

```
lartisien-ml-project/
├── analysis.qmd              # D1 · Full reproducible Quarto report (runs end-to-end)
├── presentation.qmd          # D4 · Reveal.js slide deck
├── ai_reflection.md          # D3 · AI workflow reflection
├── executive_summary.md      # D5 · Plain-language executive summary
├── hotels_rates.csv          # Clean dataset (1,440 rows, 24 hotels)
├── classifier.pkl            # Trained XGBoost classification pipeline
├── regressor.pkl             # Trained GradBoost regression pipeline
├── requirements.txt          # Python dependencies
├── frontend/                 # React + Vite web app (D2)
│   └── src/data/
│       ├── predictions.json  # 1,728 pre-computed predictions
│       └── leaderboard.json  # 6-model leaderboard for both tasks
└── submission/               # All five deliverables + SUBMISSION.md
```

---

## How to Run the Analysis Report (D1)

```bash
# Clone
git clone https://github.com/Aruni333/lartisien-ml-project.git
cd lartisien-ml-project

# Install Python dependencies (includes scikit-learn, xgboost, statsmodels, quarto)
pip install -r requirements.txt

# Render the report (runs the full ML pipeline in code cells)
quarto render analysis.qmd --to html
# → analysis.html (self-contained, opens in any browser)

# Render slides
quarto render presentation.qmd --to revealjs
# → presentation.html
```

---

## How to Run the App Locally

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

The app uses 1,728 pre-computed predictions in `src/data/predictions.json` — no backend server required.

---

## Top Findings (v2 — Interpretable Features)

1. **Room category** is the dominant pricing signal (44% regression importance) — the single strongest driver
2. **Hotel tier (1–3)** (13%) — replaced 23 brand-specific OHE dummies with one interpretable ordinal; model now generalises to new properties in the same tier
3. **Christmas proximity** (11%) — 1/(days_to_Christmas+1) captures the non-linear rate surge in December
4. **Seasonality** — month sin+cos combined (~12%) captures summer peaks at beach/coastal properties
5. **Overfitting honestly addressed:** Decision Tree (full depth) shows Train F1=1.000 vs Test F1=0.719; XGBoost corrects this (gap=0.064)

---

## Tech Stack

- Python 3.12 · scikit-learn · XGBoost · pandas · numpy · matplotlib · seaborn · statsmodels
- React 19 + Vite (deployed app)
- Quarto (report + slides)
- Vercel (deployment)
- Claude Code + MCP (AI-assisted development — see ai_reflection.md)
