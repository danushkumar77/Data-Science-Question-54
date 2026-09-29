# Smart City: Does Weather Affect Traffic?

Reproducible end-to-end analysis of traffic sensor speeds, traffic volumes, and hourly rainfall. If a `traffic*.csv` or `smart_city*.csv` file is in the project root, the pipeline uses the first alphabetically. Otherwise, it creates the seeded 15-road, 90-day synthetic dataset at `data/traffic_raw.csv`. Synthetic results demonstrate the workflow; they are not measurements from a real city.

## Setup

Use Python 3.11 or newer. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Run the full pipeline:

```powershell
python -m src.traffic_weather_analysis
```

Or open `notebook/traffic_weather_analysis.ipynb` and run all cells from top to bottom. Both entry points use the same analysis functions and fixed seed (`42`). The script writes cleaned data, descriptive tables, statistical test results, figures, model metrics, a serialized model, `REPORT.md`, and `REPORT.pdf`.

## Project map

- `data/traffic_raw.csv`: supplied or generated input data.
- `data/traffic_clean.csv`: validated analysis data; derived time and weekday labels are recalculated.
- `src/traffic_weather_analysis.py`: reusable generation, cleaning, statistical analysis, modeling, charting, and reporting functions.
- `notebook/traffic_weather_analysis.ipynb`: narrated end-to-end analysis.
- `outputs/figures/`: publication-quality 200 dpi figures.
- `outputs/road_sensitivity.csv`: per-road effects, bootstrap intervals, and Benjamini-Hochberg adjusted p-values.
- `outputs/model_metrics.csv`: time-held-out model and baseline metrics, including the no-volume tree variant.
- `outputs/decision_tree_model.joblib`: fitted tree pipeline with preprocessing.
- `REPORT.md` and `REPORT.pdf`: planner-facing results, limitations, and action plan.

## Assumptions and interpretation

Speeds outside 5-90 km/h and negative volumes/rainfall are treated as sensor errors, then imputed; valid IQR outliers are retained and reported. Missing speeds and volumes use the road-by-time-of-day median, with a road/global fallback. Time of day and weekday are derived from timestamps. Rain is defined as rainfall greater than zero. Rain sensitivity classes are High at a speed drop of at least 20%, Moderate at 8% to under 20%, and Low below 8%; FDR uses alpha 0.05. Congestion is speed below 60% of that road's training-period 85th percentile. These thresholds are explicit design choices.

The data are observational (or synthetic when no input CSV is supplied), so findings are associations, not causal effects. Accidents, roadworks, events, visibility, and route choice are not modeled. Traffic volume can be partly downstream of weather; compare the model variants with and without it. Validate findings against audited local sensors before making operational decisions.