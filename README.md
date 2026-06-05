# NYC_taxi_trips

Exploratory analysis of NYC yellow taxi trip data.

## Data setup

The dataset files are **not** included in this repository (they exceed GitHub’s file size limits). Download them from your course or competition source and place them locally before running the notebook.

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

## Modeling

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook modeling.ipynb
```

Cleaning and splits live in `preprocess.py` (same rules as `EDA.ipynb`). Validation uses pickups on or after **2016-06-13** to mimic late-June test data.
