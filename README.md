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
├── EDA.ipynb
└── README.md
```

- **`data/train.csv`** — required; used by `EDA.ipynb` (`pd.read_csv("data/train.csv")`)
- **`data/test.csv`** — optional for EDA; keep it in the same folder if you use it for modeling or submission

Paths are relative to the project root, so run Jupyter from this directory (or open the notebook with the workspace set to the repo root).
