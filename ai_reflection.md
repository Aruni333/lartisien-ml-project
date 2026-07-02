# D3 — AI Workflow Reflection

**Project:** Lartisien Collection ML Rate Intelligence (v2 Rebuild)
**Author:** Minka Borec · MADA 2025

---

## Which AI tools were used

This project was built with **Claude Code** (claude-sonnet-4-6), the Anthropic CLI assistant, across multiple sessions: an initial v1 build, a full v2 rebuild (dataset, features, model training), and then an extended, multi-session review-and-fix pass immediately before submission that turned up and corrected several real errors (detailed below).

Specific Claude Code features relied on:

| Feature | How it was used |
|---------|----------------|
| **Agentic code execution** | Claude wrote and ran all Python scripts end-to-end — dataset construction, preliminary EDA, feature engineering, VIF audit, full model ladder, evaluation, and a real `GridSearchCV` hyperparameter search added during the final review pass |
| **Read / Edit / Write tools** | Every source file was written by Claude and iteratively corrected across many rounds; `analysis.qmd`, `presentation.qmd`, the executive summary `.md`/`.docx`, `SUBMISSION.md`, and the React app were each revised multiple times as issues surfaced |
| **Context7 MCP** | Used to fetch current scikit-learn/XGBoost API documentation so `Pipeline`, `GridSearchCV`, and `XGBClassifier` calls matched the installed versions |
| **GitHub MCP** | Used to inspect the live GitHub repo file-by-file (not just push) — this is how a major desync between local and remote was caught: see "What the review pass caught" below |
| **Vercel MCP / CLI** | Used to inspect deployment status, list aliases, generate a temporary access-bypass link to view a protected deployment, and ultimately trigger a real redeploy + re-alias once the stale live app was identified |
| **Playwright MCP** | Used to actually load the live app in a real browser and read its rendered content — not just assume the deployed code matched the repo |
| **python-docx** | Used to programmatically rebuild the executive summary Word document from scratch after the existing one was found to be stale v1 content |
| **Multi-step error recovery** | Matplotlib `labels`→`tick_labels` deprecation, a `COLORS` array sized for 5 models breaking after a 6th was added, Word-doc style `KeyError`s on a custom template — all diagnosed and fixed within-session |

---

## What changed in v2 (and why)

The v1 model had a fundamental interpretability problem: it used `hotel_id` as 23 OHE dummies, which caused it to memorise specific brand identities. Two outlier hotels (Cheval Blanc Randheli, The Brando) dominated feature importance because their extreme rates (~€9,930 and €7,085 median) made their brand dummies the easiest signal.

**v2 changes:**
- Dataset expanded from 10 to 24 Lartisien-verified hotels (26 built, 2 removed by IQR)
- Preliminary EDA with IQR outlier detection at hotel level (not row level)
- `hotel_id` OHE replaced by `hotel_tier` (1–3 ordinal) — interpretable, generalises to new properties
- `days_to_christmas` dropped (r=−0.998 with month) → replaced by `xmas_proximity` = 1/(days+1)
- `log_hotel_size` dropped (VIF=12.23) and `is_weekend` dropped (r=−0.755 with month_sin, r=0.06 with target)
- Full 6-model ladder (added Decision Tree to explicitly demonstrate overfitting)
- RMSE and predicted-vs-actual plot added to regression evaluation
- React app updated: 24 hotels, 1,728 pre-computed predictions

---

## What the pre-submission review pass caught

This is the part of the process that took the most human time, and it's worth documenting concretely because it's the clearest evidence of actual supervision rather than trusting AI output at face value. Working through the submission file-by-file against the actual live app and the actual GitHub repo (not just the local working copy) surfaced six distinct, real problems that a lighter review would have missed:

1. **The live app required a Vercel login to open at all** (Deployment Protection was on) — would have failed "app opens for someone else" outright had it not been caught by actually navigating to the URL.
2. **GitHub was serving a completely different, stale v1 app** — 10 hotels including the two outliers the v2 analysis explicitly removed, wrong metrics in the footer, old color scheme. Caught by fetching the actual file contents from the GitHub API and diffing against local, not by assuming a prior push had worked.
3. **The report claimed `GridSearchCV`-tuned models and generated learning curves that didn't actually exist in the code** — caught by grepping the report source for the claimed function calls and finding none. This was later properly fixed by adding a real, working `GridSearchCV` search rather than just softening the false claim.
4. **The executive summary `.docx` was untouched v1 content** — cited a hotel (Cheval Blanc Randheli) as a top pricing driver that the model doesn't even use as a feature anymore, and that was itself removed from the training data as an outlier. Caught by extracting and reading the actual paragraph text of the Word document, not just assuming the `.md` and `.docx` were in sync.
5. **The report's own outlier-detection code was silently finding zero outliers** while the prose next to it claimed to have removed two specific hotels with specific dollar figures — because the CSV it loaded had already had those hotels excluded upstream, before this notebook ever ran. Caught by literally re-running the IQR computation against the file the code actually loads and comparing the printed output to the prose claim.
6. **The deployed app's footer metrics didn't match the report's metrics**, and neither matched what a fresh `vercel deploy` would produce, because the Vercel project had no GitHub integration — deployments only ever happened via manual CLI runs. Caught by comparing three independent sources (report numbers, live app DOM content via Playwright, and the Vercel deployment list) rather than trusting any single one.

Each of these was verified against primary evidence (actual computed output, actual file content, actual rendered page) before being called a bug and fixed, and each fix was re-verified after the fact (re-rendered the report, re-deployed the app, re-fetched from GitHub to confirm the push landed).

## How AI output was verified (ongoing, throughout)

**1. Run and read the output.** Every script was executed and terminal output checked manually. Numbers that looked wrong triggered investigation.

**2. Cross-check against course materials.** MADA notebook and lecture notes were used as a benchmark — expected patterns (CV vs. test gaps, feature importance distributions) were compared against Claude's outputs.

**3. VIF audit.** The VIF output for v1 features showed `log_hotel_size` VIF=12.23 and `hotel_tier` VIF=9.96 (despite being different concepts, they were correlated in this dataset). Claude identified the correlation source and proposed dropping `log_hotel_size` — confirmed by checking the correlation matrix manually (r=0.09 with the target, so minimal predictive loss).

**4. IQR logic.** The hotel-level IQR fence was independently recomputed by hand against the real 26-hotel file (not just trusted from the code's printed output): Q1=€2,043, Q3=€5,428, IQR=€3,385, fence=€7,081. Cheval Blanc (€9,930) and The Brando (€7,085) correctly identified — and this recomputation is what surfaced bug #5 above.

**5. Overfitting flags.** Decision Tree full depth shows Train F1=1.000 vs Test F1=0.719 (gap=0.143) — textbook overfitting. XGBoost's `GridSearchCV` search was checked against the manually-chosen GradBoost hyperparameters — the search independently converged on the same `max_depth=3, n_estimators=300, learning_rate=0.1`, which is itself a form of verification that the manual choice wasn't a lucky guess.

**6. App spot-check, done twice.** Once against the static `predictions.json` values directly (Four Seasons Budapest Deluxe Room in January → low-rate Standard, as expected for a tier-1 city hotel in low season), and a second time against the actual live, rendered page in a real browser via Playwright — which is what caught bug #1 and bug #6 above; a JSON-only check would have missed both.

---

## Cost and effort estimate

| Activity | AI | Human |
|----------|-----|-------|
| v1 → v2 dataset rebuild (26→24 hotels, IQR EDA) | Claude wrote 100% | ~25 min direction + verification |
| Preliminary EDA (8+ figures, IQR audit) | Claude wrote and ran | ~20 min oversight |
| Feature engineering & VIF audit | Claude wrote; human verified VIF logic by hand | ~25 min review |
| Model ladder (6 models × 2 tasks) + real XGBoost `GridSearchCV` | Claude wrote full ladder and tuning search | ~25 min results review |
| Confusion matrix, classification report, predicted-vs-actual | Claude added | ~10 min |
| Word doc rebuild (after v1-stale version was caught) | Claude regenerated via python-docx | ~15 min review + content correction |
| React app (Predictor + Dashboard, footer bug fix, redeploy) | Claude updated JSX/JSON, ran deploy | ~20 min review + live verification |
| Quarto documents (`analysis.qmd`, `presentation.qmd`), multiple full rewrite passes | Claude rewrote repeatedly as issues surfaced | ~45 min review across passes |
| **Pre-submission audit** (GitHub-vs-local diff, live-app browser check, docx content extraction, outlier-code re-verification, Vercel deployment/alias inspection) | Claude ran the checks and reported findings | **~90 min** — this was the single largest chunk of human time, spent deciding what counted as a real bug vs. a stylistic nit, and confirming each fix actually landed |
| Submission files (`SUBMISSION.md`, executive summary, README, business-question reframing, feature-name simplification) | Claude wrote/rewrote | ~30 min review and redirection across several rounds |

**Total human time:** ≈4.5–5 hours of active, hands-on supervision across the full project (initial build + v2 rebuild + the pre-submission audit above), not a single quick pass.
**Total AI generation/compute time:** ≈2–3 hours across all scripts, renders, and file edits.
**Approximate API cost:** estimated €8–15 at Claude Sonnet pricing, reflecting the extended review-and-fix session on top of the original build.

The primary human contribution was not writing code — it was **deciding what to trust and what to re-verify independently**: setting the research direction, checking statistical reasoning by hand (IQR fence, VIF interpretation, train-test gap meaning), and — critically — refusing to accept "the code runs" as proof that the submission was correct. Several of the bugs above (stale GitHub content, a report claiming a computation it wasn't actually running, a live app showing different numbers than the report) would not have been caught by reading the code in isolation; they only surfaced by cross-checking independent sources of truth against each other.

---

## Limitations of AI-assisted coding

- Claude initially added too many features and required a VIF-based audit to prune — this iterative process took several rounds.
- The matplotlib `labels` parameter had changed to `tick_labels` in recent versions; Claude initially used the deprecated form.
- Word document style availability varies per `.docx` template — required debugging (no "List Bullet", no "Table Grid" in this custom template).
- The `COLORS` array in the JSON serialisation was sized for 5 models, not 6, causing an `IndexError` after adding Decision Tree — caught and fixed.
- **Claude will confidently narrate a computation as having happened (e.g. "GridSearchCV tuned this model," "learning curves confirm convergence") without it actually being in the code** — this was the single biggest category of issue found in the pre-submission audit, and the only reliable defence was re-running the actual code and reading its actual output rather than trusting the prose describing it.
- Claude cannot verify Lartisien rate accuracy against live prices, and cannot on its own notice that a *deployed* artifact (GitHub repo, live Vercel app) has drifted from the local working copy — that required explicitly fetching and reading the remote state, which doesn't happen unless a human asks for it.
