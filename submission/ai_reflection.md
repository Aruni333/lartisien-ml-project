# D3 — AI Workflow Reflection

**Project:** Lartisien Collection ML Rate Intelligence (v2 Rebuild)
**Author:** Minka Borec · MADA 2025

---

## Which AI tools were used

This project was built with **Claude Code** (claude-sonnet-4-6), the Anthropic CLI assistant. The v2 rebuild was a complete rewrite of dataset, features, and model training following a detailed code review session.

Specific Claude Code features relied on:

| Feature | How it was used |
|---------|----------------|
| **Agentic code execution** | Claude wrote and ran all Python scripts end-to-end — dataset construction, preliminary EDA, feature engineering, VIF audit, full model ladder, evaluation |
| **Read / Edit / Write tools** | Every source file was written by Claude and iteratively corrected; analysis.qmd, presentation.qmd, Word doc all updated programmatically |
| **Context7 MCP** | Used to fetch current scikit-learn and XGBoost API documentation to ensure Pipeline, GridSearchCV, and XGBClassifier calls matched the installed versions |
| **GitHub MCP** | Used to push all deliverables to the public repository in a single commit |
| **Vercel MCP** | Used to deploy the React app and check build logs |
| **Multi-step error recovery** | When scripts failed (matplotlib `labels` parameter, f-string format error, COLORS array too short, Word doc style KeyErrors), Claude diagnosed root causes and fixed within the same session |
| **python-docx** | Used to programmatically write EDA, feature engineering, and model evaluation sections to the project proposal Word document, including manual XML-level table borders |

---

## What changed in v2 (and why)

The v1 model had a fundamental interpretability problem: it used hotel_id as 23 OHE dummies, which caused it to memorise specific brand identities. Two outlier hotels (Cheval Blanc Randheli, The Brando) dominated feature importance because their extreme rates (~€9,930 and €7,085 median) made their brand dummies the easiest signal.

**v2 changes:**
- Dataset expanded from 10 to 24 Lartisien-verified hotels (26 built, 2 removed by IQR)
- Preliminary EDA with IQR outlier detection at hotel level (not row level)
- hotel_id OHE replaced by hotel_tier (1–3 ordinal) — interpretable, generalises to new properties
- days_to_christmas dropped (r=−0.998 with month) → replaced by xmas_proximity = 1/(days+1)
- log_hotel_size dropped (VIF=12.23) and is_weekend dropped (r=−0.755 with month_sin, r=0.06 with target)
- Full 6-model ladder (added Decision Tree to explicitly demonstrate overfitting)
- RMSE and predicted-vs-actual plot added to regression evaluation
- React app updated: 24 hotels, 1,728 pre-computed predictions

---

## How I verified AI output

**1. Run and read the output.** Every script was executed and terminal output checked manually. Numbers that looked wrong triggered investigation.

**2. Cross-check against course materials.** MADA notebook and lecture notes were used as a benchmark — expected patterns (learning curves, CV vs test gaps, feature importance distributions) were compared against Claude's outputs.

**3. VIF audit.** The VIF output for v1 features showed log_hotel_size VIF=12.23 and hotel_tier VIF=9.96 (despite being different concepts, they were correlated in this dataset). Claude identified the correlation source and proposed dropping log_hotel_size — confirmed by checking the correlation matrix manually (r=0.09 with the target, so minimal predictive loss).

**4. IQR logic.** When Claude implemented hotel-level IQR detection, the fence was verified manually: Q1=€2,043, Q3=€5,428, IQR=€3,385, fence=€7,081. Cheval Blanc (€9,930) and The Brando (€7,085) correctly identified.

**5. Overfitting flags.** Decision Tree full depth shows Train F1=1.000 vs Test F1=0.719 (gap=0.143) — textbook overfitting. XGBoost gap=0.064 confirms ensembling corrects this.

**6. App spot-check.** Predictions.json was checked: Four Seasons Budapest Deluxe Room in January returns a low-rate Standard classification (expected: tier 1 city hotel in low season). Badrutt's Palace Presidential Suite in December returns Premium with high probability (expected: tier 3 Swiss ski resort at Christmas).

---

## Cost and effort estimate

| Activity | AI | Human |
|----------|-----|-------|
| v2 dataset (26→24 hotels, IQR EDA) | Claude wrote 100% | ~20 min direction + verification |
| Preliminary EDA (8 figures, IQR audit) | Claude wrote and ran | ~15 min oversight |
| Feature engineering & VIF audit | Claude wrote; human verified VIF logic | ~20 min review |
| Model ladder (6 models × 2 tasks) | Claude wrote full ladder | ~15 min results review |
| Confusion matrix + predicted-vs-actual | Claude added | ~5 min |
| Word doc sections (EDA, features, models) | Claude generated via python-docx | ~10 min review |
| React app (Predictor + Dashboard update) | Claude updated all JSX + JSON | ~10 min review |
| Quarto documents (analysis.qmd, presentation.qmd) | Claude rewrote | ~20 min review |
| Submission files (SUBMISSION.md, executive_summary, README) | Claude wrote | ~10 min review |

**Total human time:** ≈2 hours active oversight across 2 sessions  
**Total AI generation time:** ≈1 hour compute across all scripts and file edits  
**Approximate API cost:** estimated €2–5 at Claude Sonnet pricing (v2 rebuild was more extensive)

The primary human contribution was: setting direction and the research question, verifying statistical reasoning (IQR at hotel level vs row level, VIF interpretation, train-test gap meaning), cross-checking feature importance against domain knowledge, and approving model selection.

---

## Limitations of AI-assisted coding

- Claude initially added too many features and required a VIF-based audit to prune — this iterative process took several rounds
- The matplotlib `labels` parameter had changed to `tick_labels` in recent versions; Claude initially used the deprecated form
- Word document style availability varies per .docx template — required debugging (no "List Bullet", no "Table Grid" in this custom template)
- The COLORS array in the JSON serialisation was sized for 5 models, not 6, causing an IndexError after adding Decision Tree — caught and fixed
- Claude cannot verify Lartisien rate accuracy against live prices — this required human spot-checking against publicly available rate cards
