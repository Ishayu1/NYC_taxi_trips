# NYC_taxi_trips

Exploratory analysis of NYC yellow taxi trip data.

## Data setup

The dataset files are **not** included in this repository (they exceed GitHub’s file size limits). Download them from [kaggle](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data) and place them in the structure below locally before running the notebook.

Store the CSV files in a `data/` directory at the project root:

```
NYC_taxi_trips/
├── data/
│   ├── train.csv
│   └── test.csv
├── EDA.ipynb          # exploratory analysis only
├── preprocess.py    # cleaning, features, time-based split
├── modeling.ipynb     # baselines and (later) tuned models
├── requirements.txt
└── README.md
```

- **`data/train.csv`** — required; used by `EDA.ipynb` (`pd.read_csv("data/train.csv")`)
- **`data/test.csv`** — optional for EDA; keep it in the same folder if you use it for modeling or submission

Paths are relative to the project root, so run Jupyter from this directory (or open the notebook with the workspace set to the repo root).

## Reproducing results (from a fresh checkout)
1. Open a terminal in the project root (`NYC_taxi_trips/`).
2. Set up a Python environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and put the files here:
   - `data/train.csv` (required)
   - `data/test.csv` (optional; used only by EDA or if you later extend modeling)

4. Generate the EDA figure (optional but useful):
   - run `EDA.ipynb`
   - output: `figures/trip_duration_distributions.png`

5. Generate the model comparison + ablation results:
   - run `modeling.ipynb`
   - the notebook reads `data/train.csv`, uses a time-based validation split starting on `2016-06-13`, and writes outputs to `figures/` and `results/`.

### Expected outputs
After running `EDA.ipynb` and `modeling.ipynb`, you should see files like:
- `figures/trip_duration_distributions.png`
- `figures/error_analysis_mae.png`
- `results/baseline_results.csv`
- `results/model_comparison.csv`
- `results/ablation_results.csv`
- `results/hyperparameter_sensitivity.csv`
- `results/error_mae_by_distance.csv`
- `results/error_mae_by_hour.csv`
- `results/error_mae_by_duration.csv`
- `results/error_by_segment.csv`

Note: depending on your versions of `xgboost`/`lightgbm` and system libraries, the exact numeric metrics may vary slightly, but the pipeline and output file locations remain the same.

## Modeling (entry point)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook modeling.ipynb
```

Cleaning and splits live in `preprocess.py` (same rules as `EDA.ipynb`). Validation uses pickups on or after **2016-06-13** to mimic late-June test data.
