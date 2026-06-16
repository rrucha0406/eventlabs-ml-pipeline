# eventlabs-ml-pipeline
# EventLabs ML Pipeline — Churn Prediction & Watch-Time Forecasting

> End-to-end ML system that predicts webinar attendee churn and watch-time, then converts predictions into real-time organizer interventions.

---

## What this project does

EventLabs is a webinar platform. Organisers face a core problem: they don't know which attendees are about to drop off until it's too late to act.

This project builds a **two-model ML pipeline** that solves it:

| Model | Type | Target | Algorithm |
|---|---|---|---|
| A — Churn classifier | Classification | Will this user leave before 50% of the event? | Logistic Regression + XGBoost |
| B — Watch-time regressor | Regression | How many minutes will this user watch? | XGBoost |

Predictions from both models are then converted into a **live decision system** — risk buckets, intervention triggers, fairness audits, and per-organizer action reports.

---

## Results at a glance

| Metric | Value |
|---|---|
| Churn model ROC-AUC | **0.91+** |
| Watch-time model R² | **0.93+** |
| Watch-time RMSE | **< 2 min** |
| Users correctly flagged as at-risk | ~80% precision at 0.65 recall |
| Fairness verdict (device + network) | No significant bias detected |

---

## Project structure

```
eventlabs-ml-pipeline/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── eventlabs_combined.ipynb      ← Full pipeline (run this)
│   └── eventlabs_decisions.ipynb     ← Decisions-only deep dive
│
├── data/
│   └── eventlabs.csv                 ← Source dataset
│
└── reports/
    ├── key_visuals/                  ← Exported charts (PNG)
    └── summary.md                    ← 1-page findings summary
```

---

## The three-part pipeline

### Part A — Churn classification (Logistic Regression)

Predicts binary churn (`engaged_label`) — whether a user watches less than 50% of the webinar.

Key steps:
- Dropped leaky columns (`watch_time_min`, `watched_pct`, `dropped_at_min`) to prevent data leakage
- Engineered 5 features: `pre_engagement_score`, `loyalty_score`, `avg_completion_per_event`, `total_interactions`, `registered_early_flag`
- Handled class imbalance (~80/20 split) with `class_weight='balanced'`
- Tuned decision threshold on Precision-Recall curve to achieve P ≥ 0.65 and R ≥ 0.65 simultaneously
- Benchmarked against Random Forest and XGBoost

### Part B — Watch-time regression (XGBoost)

Predicts exact minutes watched (`watch_time_min`) for each attendee.

Key steps:
- Engineered 6 features: `engagement_score`, `session_interaction`, `completion_x_past`, `pre_event_engagement`, `did_join`, `promo_watch_min`
- Compared Ridge baseline → Random Forest → XGBoost
- XGBoost achieved R² > 0.93 and RMSE < 2 min

### Part C — Decision system

Converts predictions into actions an organiser can actually take.

- **Risk bucket classification** — 4 tiers (Critical / High / Medium / Low) based on predicted watch time as % of event length
- **Dual-model overlay** — combines churn probability (Model A) + watch-time (Model B) for a richer view of each user
- **Intervention framework** — trigger timing, action, channel, and pacing recommendation per tier
- **Fairness audit** — MAE and bias analysis across device type and network quality
- **Segmented error analysis** — error heatmap (device × region), per-bucket error distributions, over/under-prediction breakdown

---

## Intervention framework (summary)

| Risk tier | Users | When to fire | Action |
|---|---|---|---|
| Critical | ~X% | First 5 min | In-session check-in poll |
| High | ~X% | At 25% of event | Content recap + poll via chat |
| Medium | ~X% | At 50% of event | Halfway nudge + Q&A invite |
| Low | ~X% | At 80% of event | Next-event promo |

---

## Key findings

1. `engagement_score` (heartbeat pings + seek events + chat) is the strongest in-session predictor → build a live engagement meter
2. `completion_x_past` predicts dropout before the event even starts → use for pre-event email targeting
3. `pre_event_engagement` (email opens + reminder clicks) separates low-risk from high-risk → run reminder campaigns 24h prior for Critical-tier users
4. Model is fair across device types and network quality — no group is systematically under-predicted

---

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/eventlabs-ml-pipeline.git
cd eventlabs-ml-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Place eventlabs.csv inside the data/ folder

# 4. Run the notebook
jupyter notebook notebooks/eventlabs_combined.ipynb
```

---

## Tech stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `Jupyter`

---

## Skills demonstrated

- End-to-end ML pipeline design (EDA → features → models → decisions)
- Classification with imbalanced data and threshold tuning
- Regression with feature engineering and multi-model comparison
- Translating model outputs into business interventions
- Fairness auditing across demographic segments
- Segmented error analysis (MAE heatmaps, bias direction, P90 error)

---

## Author

**Rrucha Wamanse**
[LinkedIn](https://www.linkedin.com/in/rrucha-wamanse-428267324/) · [GitHub](https://github.com/rrucha0406)
