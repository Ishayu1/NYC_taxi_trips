# NYC Taxi Trip Duration — Project Status & Handoff

**Last updated:** June 2026  
**Purpose:** This document describes the full current state of the repository so a collaborator (or their coding agent) can pick up the project without re-reading every notebook. It covers what was built, how it fits together, key results, and what is not yet committed to GitHub.

---

## 1. Project overview

| Item | Detail |
|------|--------|
| **Course** | DSC 148 |
| **Dataset** | NYC Taxi Trip Duration (Kaggle-style competition data) |
| **Research question** | How accurately can we predict NYC taxi trip duration using **only information available at pickup time**, and which spatial/temporal feature groups contribute most to predictive performance? |
| **Target** | `trip_duration` (seconds) |
| **Prediction setting** | Pickup-time prediction only — no post-trip information |
| **Primary metric** | `rmse_log` — RMSE on `log1p(trip_duration)` |
| **Secondary metrics** | `rmse_sec`, `mae_sec` (on back-transformed seconds, predictions clipped at 0) |

### Leakage rules (enforced in code)

Do **not** use as model inputs:

- `dropoff_datetime`
- Implied **speed** (speed is used only for EDA/outlier filtering in `preprocess.py`)
- Any feature derived from `trip_duration`

Allowed at pickup time: coordinates, Manhattan/haversine distance, vendor/passenger metadata, pickup datetime-derived time features.

---

## 2. Repository layout

```
NYC_taxi_trips/
├── data/                          # NOT in git — see Data setup
│   ├── train.csv                  # ~1.46M rows, 191 MB
│   └── test.csv                   # ~625K rows, 68 MB
├── EDA.ipynb                      # Exploratory analysis (39 cells)
├── modeling.ipynb                 # Modeling & results pipeline (25 cells)
├── preprocess.py                  # Shared cleaning, features, splits, metrics
├── requirements.txt               # Python dependencies
├── README.md                      # Setup instructions for data + modeling
├── PROJECT_STATUS.md              # This file
├── results/                       # Saved metric tables (CSV) — currently untracked
│   ├── baseline_results.csv
│   ├── model_comparison.csv
│   ├── ablation_results.csv
│   ├── hyperparameter_sensitivity.csv
│   ├── error_by_segment.csv
│   ├── error_mae_by_distance.csv
│   ├── error_mae_by_hour.csv
│   └── error_mae_by_duration.csv
└── figures/
    └── error_analysis_mae.png     # MAE by distance / hour / duration
```

**Git ignore:** `data/*.csv`, `.DS_Store`, standard Python/Jupyter artifacts (see `.gitignore`).

---

## 3. Collaboration history (git)

Recent commits on `main`:

| Commit | Author / source | Summary |
|--------|-----------------|---------|
| `5b71dbc` | Initial | Empty repo scaffold |
| `047bc0b` | Ishayu | Added `EDA.ipynb`; excluded large CSVs from git (GitHub 100 MB limit) |
| `8f04189` | Ishayu | Updated `README.md` with data setup instructions |
| `3aaa012` | Partner (Palina) | Extended `EDA.ipynb`; added `preprocess.py`, initial `modeling.ipynb`, `requirements.txt` |
| `b055eca` | Merge | PR #1 merged (`palina/further-eda`) |
| `83c511e` | Ishayu | Local EDA additions (later superseded by merge) |
| `97bb8cc` | Merge | Merged partner EDA with local work; kept partner's extended `EDA.ipynb` |

### Division of work (approximate)

**Partner (Palina) — EDA & preprocessing foundation**

- Extended `EDA.ipynb` with data-quality checks, categorical/temporal/spatial analysis, train vs test comparison, modeling implications table
- Created `preprocess.py`: cleaning rules, Manhattan distance, time features, log target, time-based split, evaluation metrics
- Started `modeling.ipynb` with load/clean/split and simple baselines
- Added `requirements.txt` and modeling section in `README.md`

**Ishayu — modeling/results phase (mostly local, not yet pushed)**

- Completed full `modeling.ipynb` pipeline: baselines, tree/boosting models, ablation, error analysis, hyperparameter sensitivity, report summary
- Extended `preprocess.py` with additional features and ablation feature groups
- Added `xgboost` and `lightgbm` to `requirements.txt`
- Generated all files under `results/` and `figures/`

### Current git status (as of last check)

Branch `main` is **ahead of `origin/main` by 2 commits** (merge + earlier local commit). Additionally, these changes exist **locally but are not committed**:

- Modified: `modeling.ipynb`, `preprocess.py`, `requirements.txt`
- Untracked: `results/`, `figures/`, `PROJECT_STATUS.md`

**Action for partner:** Pull latest, then sync on whether to commit/push `results/`, `figures/`, and the modeling updates together.

---

## 4. Data setup

CSV files are **not** in the repository. Each collaborator must download them and place:

```
data/train.csv
data/test.csv
```

Run Jupyter from the **project root** so paths like `data/train.csv` resolve correctly.

After cleaning (~98.7% of rows retained):

| Split | Rows | Date range |
|-------|------|------------|
| **Train** | 1,303,294 | 2016-01-01 → 2016-06-12 |
| **Validation** | 136,068 | 2016-06-13 → 2016-06-30 |

Validation cutoff: **`2016-06-13`** (`DEFAULT_VAL_START` in `preprocess.py`) — mimics late-June test distribution.

---

## 5. EDA (`EDA.ipynb`) — current content

39 cells covering:

1. **Load & inspect** — `train.csv`, missing values (none), drop `id` and `dropoff_datetime` (leakage)
2. **Outliers** — 99th percentile filter on `trip_duration` for some plots; log transform for skew
3. **Temporal patterns** — median duration by pickup hour
4. **Correlations** — numeric heatmap; Manhattan distance vs duration scatter
5. **Data quality rules** — NYC bounding box, duration bounds [60s, 24h], passenger count [1–6], implied speed [1, 120] km/h, Jan–Jun 2016 pickups
6. **Categorical EDA** — `vendor_id`, `passenger_count`, `store_and_fwd_flag`
7. **Temporal EDA** — volume and median duration by month and day of week
8. **Spatial EDA** — pickup scatter plots (50k sample)
9. **Train vs test** — schema and pickup-time distribution comparison
10. **Modeling implications** — markdown table linking EDA findings to modeling choices (log target, time split, feature groups)

All cleaning rules in `preprocess.py` were designed to match this EDA.

---

## 6. Preprocessing module (`preprocess.py`)

Central module — **reuse this instead of duplicating logic** in notebooks.

### Key functions

| Function | Purpose |
|----------|---------|
| `load_train()` / `load_test()` | Load CSV; drop `id`, `dropoff_datetime` (train only) |
| `clean_dataframe()` | Apply quality filters; add `manhattan_km` |
| `quality_check_summary()` | Per-rule failure counts (EDA table) |
| `add_time_features()` | Hour, DOW, month, weekend, rush hour, cyclic hour |
| `add_log_target()` | Adds `log_trip_duration = log1p(trip_duration)` |
| `build_features()` | Full pickup-time feature matrix (16 columns) |
| `select_features(X, group)` | Subset for ablation (`FEATURE_GROUPS`) |
| `train_val_split_by_date()` | Time-based train/val split |
| `evaluate_predictions()` | Returns `rmse_log`, `rmse_sec`, `mae_sec` |

### Engineered features (16 total, all pickup-time safe)

**Metadata (3):** `vendor_id`, `passenger_count`, `store_and_fwd_flag` (encoded N→0, Y→1)

**Time (7):** `pickup_hour`, `pickup_dow`, `pickup_month`, `is_weekend`, `is_rush_hour`, `hour_sin`, `hour_cos`

- Rush hour: weekday hours {7, 8, 9, 16, 17, 18, 19}

**Distance (2):** `manhattan_km`, `haversine_km`

**Coordinates (4):** pickup/dropoff latitude and longitude

### Feature groups for ablation (`FEATURE_GROUPS`)

| Group key | Features |
|-----------|----------|
| `metadata_only` | 3 metadata columns |
| `time_only` | 7 time columns |
| `distance_only` | manhattan + haversine |
| `coordinates_only` | 4 lat/lon columns |
| `distance_time` | distance + time |
| `distance_coordinates` | distance + coordinates |
| `full` | all 16 |

---

## 7. Modeling notebook (`modeling.ipynb`)

Runs top-to-bottom in ~2 minutes on full data (Apple Silicon / modern CPU). Sections:

| Section | Content |
|---------|---------|
| **1. Load, clean, split** | Uses `preprocess.py`; prints row counts |
| **2. Baselines** | 6 baselines → `results/baseline_results.csv` |
| **3. Stronger models** | Tree/boosting models → `results/model_comparison.csv` |
| **4. Hyperparameter sensitivity** | One-at-a-time XGBoost sweep → `results/hyperparameter_sensitivity.csv` |
| **5. Feature ablation** | HGB on 7 feature groups → `results/ablation_results.csv` |
| **6. Error analysis** | Best model (XGBoost) on validation set; plots + segment CSVs |
| **7. Summary** | Report-ready bullet points |

### Models trained

**Baselines**

| Model | Feature set |
|-------|-------------|
| Mean log duration | none |
| Median duration | none |
| Fixed speed (~16.5 km/h from train median) | manhattan_km |
| Linear regression | distance_only |
| Linear regression | full |
| Ridge (α=1) | full |

**Stronger models** (all on full 16 features, target = `log_trip_duration`, predict via `expm1`)

| Model | Notes |
|-------|-------|
| DecisionTreeRegressor | max_depth=16, min_samples_leaf=50 |
| RandomForestRegressor | 50 trees, max_samples=0.15 (runtime trade-off on 1.3M rows) |
| HistGradientBoostingRegressor | max_depth=8, 200 iterations |
| XGBRegressor | max_depth=8, lr=0.1, 200 trees, hist method |
| LGBMRegressor | same hyperparams (optional if installed) |

---

## 8. Results (validation set)

All metrics on **136,068 validation trips** (pickup ≥ 2016-06-13). Lower is better.

### 8.1 Baselines (`results/baseline_results.csv`)

| Model | rmse_log | rmse_sec | mae_sec |
|-------|----------|----------|---------|
| Ridge (full) | 0.512 | 1371 | 370 |
| Linear (full) | 0.512 | 1371 | 370 |
| Linear (distance only) | 0.540 | 1123 | 374 |
| Fixed speed 16.5 km/h | 0.543 | 1028 | 424 |
| Median duration | 0.743 | 1050 | 482 |
| Mean log duration | 0.744 | 1053 | 483 |

### 8.2 Full model comparison (`results/model_comparison.csv`)

| Rank | Model | rmse_log | rmse_sec | mae_sec |
|------|-------|----------|----------|---------|
| 1 | **XGBoost** | **0.324** | **779** | **195** |
| 2 | HistGradientBoosting | 0.336 | 783 | 202 |
| 3 | LightGBM | 0.337 | 783 | 203 |
| 4 | Random Forest | 0.339 | 786 | 204 |
| 5 | Decision Tree | 0.357 | 793 | 217 |
| 6–11 | Linear / Ridge / baselines | 0.512–0.744 | — | — |

**Best model:** XGBoost on full features. Primary selection criterion: `rmse_log`; `mae_sec` ≈ 195 s (~3.3 min) as interpretable secondary metric.

### 8.3 Feature ablation (`results/ablation_results.csv`)

Model: HistGradientBoostingRegressor (fixed across groups).

| Feature set | # features | rmse_log | mae_sec |
|-------------|------------|----------|---------|
| **full** | 16 | **0.336** | **202** |
| distance + coordinates | 6 | 0.393 | 257 |
| distance + time | 9 | 0.393 | 241 |
| coordinates only | 4 | 0.413 | 265 |
| distance only | 2 | 0.439 | 285 |
| time only | 7 | 0.733 | 476 |
| metadata only | 3 | 0.742 | 482 |

**Interpretation:** Spatial information (distance + coordinates) drives most of the gain. Time features add modest improvement on top. Metadata alone is near the global-median baseline.

### 8.4 Hyperparameter sensitivity (`results/hyperparameter_sensitivity.csv`)

One-at-a-time sweep around default XGBoost (`max_depth=8`, `learning_rate=0.1`, `n_estimators=200`).

| Varied param | Values | rmse_log range | Best in sweep |
|--------------|--------|----------------|---------------|
| max_depth | 6, 8, 10 | 0.315 – 0.337 | 10 → 0.315 |
| learning_rate | 0.05, 0.1, 0.2 | 0.318 – 0.334 | 0.2 → 0.318 |
| n_estimators | 100, 200, 300 | 0.318 – 0.335 | 300 → 0.318 |

Metrics stay in a **narrow band (~0.315–0.337)**; default settings are reasonable but not fully tuned.

### 8.5 Error analysis (`figures/error_analysis_mae.png`, `results/error_*.csv`)

Best model: XGBoost. Key patterns:

**MAE by Manhattan distance**

| Bucket | MAE (sec) |
|--------|-----------|
| 0–2 km | 128 |
| 2–5 km | 172 |
| 5–10 km | 243 |
| 10–20 km | 342 |
| 20+ km | 646 |

**MAE by true trip duration**

| Bucket | MAE (sec) |
|--------|-----------|
| < 5 min | 96 |
| 5–10 min | 114 |
| 10–20 min | 172 |
| 20–40 min | 295 |
| 40+ min | 833 |

**MAE by segment (`results/error_by_segment.csv`)**

| Segment | MAE (sec) | rmse_log |
|---------|-----------|----------|
| Non-rush hour | 190 | 0.321 |
| Weekday | 195 | 0.322 |
| Weekend | 194 | 0.332 |
| Rush hour | 206 | 0.331 |

**Interpretation for report**

- Errors grow with distance and trip length (harder to predict long trips).
- Rush-hour pickups have ~8% higher MAE than off-peak — traffic variability.
- Model performs best on short, local trips; worst on long/outlier-duration trips.
- Likely high-error cases: airport/bridge routes, unusual congestion, Manhattan distance ≠ driven path.

---

## 9. Environment & how to run

```bash
cd NYC_taxi_trips
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ensure data/train.csv and data/test.csv exist
jupyter notebook EDA.ipynb      # exploratory analysis
jupyter notebook modeling.ipynb # full modeling pipeline (~2 min)
```

**Dependencies:** pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, lightgbm

Re-running `modeling.ipynb` regenerates all CSVs in `results/` and `figures/error_analysis_mae.png`.

---

## 10. Report-ready summary (copy into paper)

1. **Question:** Pickup-time prediction of taxi trip duration using spatial, temporal, and metadata features.
2. **Data:** 1.44M cleaned training trips; time-based validation on last ~2.5 weeks of June 2016.
3. **Best model:** XGBoost — validation `rmse_log = 0.324`, `mae_sec = 195`.
4. **Feature importance (ablation):** Distance and coordinates dominate; time features help incrementally; metadata negligible.
5. **Error patterns:** Performance degrades for long distances/durations and rush-hour pickups.
6. **Leakage control:** No `dropoff_datetime`, speed, or duration-derived features in models.

---

## 11. Known limitations & possible next steps

| Topic | Status |
|-------|--------|
| Test set predictions / Kaggle submission | **Not done** — no submission file generated |
| Cross-validation | **Not done** — single time-based holdout only |
| Full hyperparameter tuning | Partial — one-at-a-time XGBoost sensitivity only |
| SHAP / feature importances | **Not done** |
| Final tuned XGBoost (depth=10, more trees) | Identified in sensitivity but not used in main comparison table |
| Git push of modeling results | **Pending** — partner should pull and agree on commit |

### Suggested next tasks for either collaborator

1. Commit and push `modeling.ipynb`, `preprocess.py`, `requirements.txt`, `results/`, `figures/`, this handoff doc.
2. Optionally re-run main comparison with best sensitivity settings (e.g. XGB depth=10, n_estimators=300).
3. Generate test-set predictions for submission if required by the assignment.
4. Transfer tables/plots from `results/` and `figures/` into the final write-up.

---

## 12. Quick reference for coding agents

**If asked to add a model:** Use `load_train` → `clean_dataframe` → `add_log_target` → `train_val_split_by_date` → `build_features`. Fit on `y_train_log`, predict with `np.maximum(np.expm1(pred_log), 0)`, evaluate with `evaluate_predictions(y_val, y_pred_sec)`.

**If asked to add a feature:** Add to `add_time_features()` or new helper in `preprocess.py`, include in `build_features()` and update `FEATURE_GROUPS` / `FULL_FEATURES`. Verify it is available at pickup time.

**If asked to change the validation split:** Modify `DEFAULT_VAL_START` in `preprocess.py` and re-run both notebooks.

**Do not:** Put `data/*.csv` in git. Do not use speed or `dropoff_datetime` as predictors.

---

*End of handoff document.*
