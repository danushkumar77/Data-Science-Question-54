"""Reproducible smart-city traffic and weather analysis pipeline."""

from __future__ import annotations

import itertools
import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RANDOM_STATE = 42
BOOTSTRAP_RESAMPLES = 2_000
ALPHA = 0.05
ROAD_COUNT = 15
DAY_COUNT = 90
HOURS_PER_DAY = 24
SPEED_MIN_KMH = 5.0
SPEED_MAX_KMH = 90.0
HEAVY_RAIN_MM = 4.0
FREE_FLOW_PERCENTILE = 0.85
CONGESTION_FREE_FLOW_SHARE = 0.60
HIGH_SENSITIVITY_DROP_PCT = 20.0
MODERATE_SENSITIVITY_DROP_PCT = 8.0
PALETTE = {
    "Clear": "#2A9D8F",
    "Cloudy": "#6C8E9B",
    "Light Rain": "#E9C46A",
    "Heavy Rain": "#E76F51",
    "Fog": "#9C89B8",
    "Dry": "#2A9D8F",
    "Rain": "#E76F51",
    "accent": "#264653",
}
WEATHER_ORDER = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Fog"]
TIME_ORDER = ["Night", "Morning Peak", "Midday", "Evening Peak"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titleweight": "semibold",
})
sns.set_theme(style="whitegrid", palette="colorblind")


def ensure_directories() -> None:
    """Create the stable project output structure."""
    for folder in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _time_of_day(timestamp: pd.Series) -> pd.Series:
    """Map clock hours to the specified four operational periods."""
    hour = timestamp.dt.hour
    conditions = [hour.between(7, 9), hour.between(10, 15), hour.between(16, 18)]
    return pd.Series(
        np.select(conditions, ["Morning Peak", "Midday", "Evening Peak"], default="Night"),
        index=timestamp.index,
    )


def _generate_synthetic_data() -> pd.DataFrame:
    """Generate correlated city weather and road-level hourly observations."""
    rng = np.random.default_rng(RANDOM_STATE)
    start = pd.Timestamp("2025-01-06 00:00:00")
    timestamps = pd.date_range(start, periods=DAY_COUNT * HOURS_PER_DAY, freq="h")
    hours = timestamps.hour.to_numpy()
    weekend = timestamps.dayofweek >= 5
    peak = ((hours >= 7) & (hours <= 9)) | ((hours >= 16) & (hours <= 18))

    # A shared hourly weather process makes rainfall comparable across sensors.
    wet = rng.random(len(timestamps)) < 0.20
    rainfall_by_hour = np.zeros(len(timestamps), dtype=float)
    rainfall_by_hour[wet] = rng.gamma(shape=0.8, scale=4.2, size=wet.sum())
    conditions = np.full(len(timestamps), "Clear", dtype=object)
    dry_weather = rng.choice(["Clear", "Cloudy", "Fog"], size=(~wet).sum(), p=[0.68, 0.25, 0.07])
    conditions[~wet] = dry_weather
    conditions[wet] = np.where(rainfall_by_hour[wet] >= HEAVY_RAIN_MM, "Heavy Rain", "Light Rain")

    road_ids = [f"R{i:02d}" for i in range(1, ROAD_COUNT + 1)]
    free_flow = rng.uniform(53, 79, ROAD_COUNT)
    base_volume = rng.uniform(380, 870, ROAD_COUNT)
    # Four underpass/arterial corridors are highly exposed; two are sheltered.
    rain_sensitivity = rng.uniform(0.24, 0.32, ROAD_COUNT)
    rain_sensitivity[4:13] = rng.uniform(0.055, 0.12, 9)
    rain_sensitivity[13:] = rng.uniform(0.005, 0.025, 2)
    road_records = []
    for road_index, road_id in enumerate(road_ids):
        rain = rainfall_by_hour
        condition = conditions
        # Peak traffic increases exposure and delay when rain is present.
        rain_fraction = np.where(rain >= HEAVY_RAIN_MM, 1.0, np.where(rain > 0, 0.48, 0.0))
        peak_multiplier = np.where(peak, 1.20, 1.0)
        speed_factor = np.where(peak, 0.76, 1.0) * np.where(weekend, 1.055, 1.0)
        speed_factor *= 1 - rain_sensitivity[road_index] * rain_fraction * peak_multiplier
        speed = free_flow[road_index] * speed_factor + rng.normal(0, 4.2, len(timestamps))
        speed = np.clip(speed, SPEED_MIN_KMH, SPEED_MAX_KMH)
        volume_factor = np.where(peak, 1.58, 1.0) * np.where(weekend, 0.73, 1.0)
        # Rain-related diversion can reduce counts even while speeds fall.
        volume_factor *= np.where(rain > 0, 0.94, 1.0)
        volume = base_volume[road_index] * volume_factor + rng.normal(0, 85, len(timestamps))
        volume = np.clip(volume, 25, None)
        road_records.append(pd.DataFrame({
            "Timestamp": timestamps,
            "Road_ID": road_id,
            "Traffic_Speed": np.round(speed, 2),
            "Traffic_Volume": np.round(volume, 1),
            "Rainfall": np.round(rain, 2),
            "Weather_Condition": condition,
            "Time_of_Day": _time_of_day(pd.Series(timestamps)).to_numpy(),
            "Weekday": timestamps.day_name(),
        }))
    data = pd.concat(road_records, ignore_index=True)

    # Messy-data rate is about 4% of rows, spread across distinct failure modes.
    row_count = len(data)
    for column, share in [("Traffic_Speed", 0.006), ("Traffic_Volume", 0.006), ("Rainfall", 0.006)]:
        selected = rng.choice(row_count, size=int(row_count * share), replace=False)
        data.loc[selected, column] = np.nan
    selected = rng.choice(row_count, size=int(row_count * 0.003), replace=False)
    data.loc[selected[: len(selected) // 2], "Traffic_Speed"] = 0
    data.loc[selected[len(selected) // 2 :], "Traffic_Speed"] = 175
    selected = rng.choice(row_count, size=int(row_count * 0.003), replace=False)
    data.loc[selected, "Traffic_Volume"] = -30
    selected = rng.choice(row_count, size=int(row_count * 0.003), replace=False)
    data.loc[selected, "Rainfall"] = -2
    selected = rng.choice(row_count, size=int(row_count * 0.005), replace=False)
    data.loc[selected[: len(selected) // 2], "Time_of_Day"] = "Night"
    data.loc[selected[len(selected) // 2 :], "Weekday"] = "Sunday"
    selected = rng.choice(row_count, size=int(row_count * 0.004), replace=False)
    data.loc[selected[: len(selected) // 2], "Weather_Condition"] = "heavy rain"
    data.loc[selected[len(selected) // 2 :], "Weather_Condition"] = "HEAVY RAIN "
    selected = rng.choice(row_count, size=18, replace=False)
    data["Timestamp"] = data["Timestamp"].astype("object")
    data.loc[selected, "Timestamp"] = pd.to_datetime(data.loc[selected, "Timestamp"]).dt.strftime("%m/%d/%Y %H:%M")
    # Exact duplicates are intentionally added to exercise duplicate handling.
    data = pd.concat([data, data.sample(n=28, random_state=RANDOM_STATE)], ignore_index=True)
    return data


def _quality_snapshot(data: pd.DataFrame, stage: str) -> dict:
    """Summarize missingness, duplicate rows, and out-of-domain sensor values."""
    snapshot = {"stage": stage, "rows": int(len(data)), "duplicate_rows": int(data.duplicated().sum())}
    for column in ["Timestamp", "Traffic_Speed", "Traffic_Volume", "Rainfall", "Weather_Condition", "Time_of_Day", "Weekday"]:
        snapshot[f"missing_{column}"] = int(data[column].isna().sum()) if column in data else 0
    speed = pd.to_numeric(data["Traffic_Speed"], errors="coerce")
    volume = pd.to_numeric(data["Traffic_Volume"], errors="coerce")
    rain = pd.to_numeric(data["Rainfall"], errors="coerce")
    snapshot["invalid_speed"] = int(((speed < SPEED_MIN_KMH) | (speed > SPEED_MAX_KMH)).sum())
    snapshot["invalid_volume"] = int((volume < 0).sum())
    snapshot["invalid_rainfall"] = int((rain < 0).sum())
    return snapshot


def load_and_clean() -> tuple[pd.DataFrame, dict]:
    """Load a supplied CSV or generate data, clean it, and save the clean table."""
    ensure_directories()
    candidates = sorted(
        path for path in [*ROOT.glob("traffic*.csv"), *ROOT.glob("smart_city*.csv"),
                          *DATA_DIR.glob("traffic*.csv"), *DATA_DIR.glob("smart_city*.csv")]
        if path not in {DATA_DIR / "traffic_raw.csv", DATA_DIR / "traffic_clean.csv"}
    )
    raw_path = candidates[0] if candidates else DATA_DIR / "traffic_raw.csv"
    if candidates:
        raw = pd.read_csv(raw_path)
    else:
        raw = _generate_synthetic_data()
        raw.to_csv(raw_path, index=False)

    required = ["Timestamp", "Road_ID", "Traffic_Speed", "Traffic_Volume", "Rainfall", "Weather_Condition", "Time_of_Day", "Weekday"]
    missing_columns = sorted(set(required) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")
    before = _quality_snapshot(raw, "before")

    # Mixed formats are expected; invalid timestamps alone are dropped because time is essential.
    data = raw.drop_duplicates().copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="mixed", errors="coerce")
    invalid_time = int(data["Timestamp"].isna().sum())
    data = data.dropna(subset=["Timestamp"]).copy()
    for column in ["Traffic_Speed", "Traffic_Volume", "Rainfall"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Weather_Condition"] = data["Weather_Condition"].astype("string").str.strip().str.title()
    valid_weather = set(WEATHER_ORDER)
    data.loc[~data["Weather_Condition"].isin(valid_weather), "Weather_Condition"] = pd.NA

    # Domain-impossible measurements become missing; plausible IQR outliers are retained.
    invalid_speed = (data["Traffic_Speed"] < SPEED_MIN_KMH) | (data["Traffic_Speed"] > SPEED_MAX_KMH)
    invalid_volume = data["Traffic_Volume"] < 0
    invalid_rainfall = data["Rainfall"] < 0
    data.loc[invalid_speed, "Traffic_Speed"] = np.nan
    data.loc[invalid_volume, "Traffic_Volume"] = np.nan
    data.loc[invalid_rainfall, "Rainfall"] = np.nan

    # Normalize labels first, then use them to conditionally impute missing rain values.
    rain_label = data["Weather_Condition"].isin(["Light Rain", "Heavy Rain"])
    data.loc[data["Rainfall"].isna() & ~rain_label, "Rainfall"] = 0.0
    for weather_name in ["Light Rain", "Heavy Rain"]:
        weather_mask = data["Weather_Condition"].eq(weather_name)
        weather_median = data.loc[weather_mask, "Rainfall"].median()
        if pd.isna(weather_median):
            weather_median = 1.0 if weather_name == "Light Rain" else HEAVY_RAIN_MM
        data.loc[data["Rainfall"].isna() & weather_mask, "Rainfall"] = weather_median
    data["Rainfall"] = data["Rainfall"].fillna(0.0)

    # Weather is reconciled to measured rain; dry weather subtype is retained when valid.
    raining = data["Rainfall"] > 0
    data.loc[raining & (data["Rainfall"] >= HEAVY_RAIN_MM), "Weather_Condition"] = "Heavy Rain"
    data.loc[raining & (data["Rainfall"] < HEAVY_RAIN_MM), "Weather_Condition"] = "Light Rain"
    data.loc[~raining & ~data["Weather_Condition"].isin(["Clear", "Cloudy", "Fog"]), "Weather_Condition"] = "Clear"
    data.loc[data["Weather_Condition"].isna() & ~raining, "Weather_Condition"] = "Clear"

    # Road/time medians preserve systematic traffic patterns better than global imputation.
    data["Time_of_Day"] = _time_of_day(data["Timestamp"])
    data["Weekday"] = data["Timestamp"].dt.day_name()
    for column in ["Traffic_Speed", "Traffic_Volume"]:
        group_median = data.groupby(["Road_ID", "Time_of_Day"], observed=True)[column].transform("median")
        road_median = data.groupby("Road_ID", observed=True)[column].transform("median")
        data[column] = data[column].fillna(group_median).fillna(road_median).fillna(data[column].median())
    data["is_weekend"] = data["Timestamp"].dt.dayofweek >= 5
    data["hour"] = data["Timestamp"].dt.hour
    data = data.sort_values(["Timestamp", "Road_ID"]).reset_index(drop=True)

    after = _quality_snapshot(data, "after")
    quality = {
        "source": str(raw_path.relative_to(ROOT)),
        "synthetic": not bool(candidates),
        "invalid_timestamps_dropped": invalid_time,
        "before": before,
        "after": after,
        "decisions": [
            "Exact duplicate rows were dropped before parsing to prevent repeated sensor readings from weighting estimates.",
            "Unparseable timestamps were dropped because time is required for all temporal controls; mixed formats were parsed with format='mixed'.",
            "Traffic speeds outside 5-90 km/h, negative volumes, and negative rainfall were set missing as physically/domain-impossible glitches.",
            "Missing or impossible speed and volume were median-imputed within Road_ID x Time_of_Day, then by road; valid domain-range extremes were retained.",
            "Missing rainfall was set to zero for dry labels and to the weather-condition median for rain labels; weather labels were reconciled to rainfall thresholds.",
            "Time_of_Day and Weekday were always recomputed from Timestamp to resolve stale or contradictory labels.",
            "IQR outliers were flagged in summaries, not capped or removed: high but valid counts/rainfall can be genuine events and capping would erase evidence.",
        ],
    }
    data.to_csv(DATA_DIR / "traffic_clean.csv", index=False)
    pd.DataFrame([before, after]).to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
    print("DATA QUALITY SUMMARY (before / after)")
    print(pd.DataFrame([before, after]).set_index("stage").T.to_string())
    print("\nCleaned dtypes")
    print(data.dtypes.to_string())
    print(f"Dropped unparseable timestamps: {invalid_time}")
    return data, quality


def _descriptive_table(data: pd.DataFrame, group_column: str | None) -> pd.DataFrame:
    """Build mean, median, spread, extremes, skew, and count tables."""
    measures = ["Traffic_Speed", "Traffic_Volume", "Rainfall"]
    if group_column is None:
        grouped = data[measures].agg(["mean", "median", "std", "min", "max", "skew", "count"])
        return grouped.T.reset_index(names="measure")
    pieces = []
    for measure in measures:
        summary = data.groupby(group_column, observed=True)[measure].agg(["mean", "median", "std", "min", "max", "skew", "count"]).reset_index()
        summary.insert(1, "measure", measure)
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True)


def exploratory_analysis(data: pd.DataFrame) -> dict:
    """Write descriptive tables for every requested grouping and correlations."""
    groupings = {"overall": None, "weather": "Weather_Condition", "road": "Road_ID",
                 "time_of_day": "Time_of_Day", "weekday": "Weekday"}
    tables = {}
    for name, column in groupings.items():
        table = _descriptive_table(data, column)
        table.to_csv(OUTPUT_DIR / f"descriptive_{name}.csv", index=False)
        tables[name] = table
    correlation_columns = ["Traffic_Speed", "Traffic_Volume", "Rainfall", "hour", "is_weekend"]
    correlations = data[correlation_columns].corr(numeric_only=True)
    correlations.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    iqr_rows = []
    for column in ["Traffic_Speed", "Traffic_Volume", "Rainfall"]:
        q1, q3 = data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_rows.append({"measure": column, "lower_fence": low, "upper_fence": high,
                         "outlier_count": int(((data[column] < low) | (data[column] > high)).sum())})
    pd.DataFrame(iqr_rows).to_csv(OUTPUT_DIR / "iqr_outlier_summary.csv", index=False)
    print("Overall descriptive statistics")
    print(tables["overall"].round(3).to_string(index=False))
    print("\nWeather-group descriptive statistics")
    print(tables["weather"].round(3).to_string(index=False))
    print("\nNumeric correlation matrix")
    print(correlations.round(3).to_string())
    return {"tables": tables, "correlations": correlations, "iqr": pd.DataFrame(iqr_rows)}


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-standard-deviation Cohen's d for a minus b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def _bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Percentile bootstrap for mean difference, percent change, and Cohen's d."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    difference_samples, percent_samples, d_samples = [], [], []
    # Batches bound memory while still taking exactly the requested resamples.
    batch_size = 80
    for start in range(0, n_boot, batch_size):
        batch = min(batch_size, n_boot - start)
        a_sample = a[rng.integers(0, len(a), size=(batch, len(a)))]
        b_sample = b[rng.integers(0, len(b), size=(batch, len(b)))]
        mean_a, mean_b = a_sample.mean(axis=1), b_sample.mean(axis=1)
        difference_samples.extend((mean_a - mean_b).tolist())
        percent_samples.extend(((mean_a - mean_b) / np.maximum(mean_b, 1e-9) * 100).tolist())
        var_a, var_b = a_sample.var(axis=1, ddof=1), b_sample.var(axis=1, ddof=1)
        pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
        d_samples.extend(np.divide(mean_a - mean_b, pooled, out=np.zeros_like(mean_a), where=pooled > 0).tolist())
    difference = float(np.mean(a) - np.mean(b))
    percent = float((np.mean(a) - np.mean(b)) / np.mean(b) * 100)
    d_value = _cohens_d(a, b)
    return {"difference": difference, "difference_ci": np.nanpercentile(difference_samples, [2.5, 97.5]).tolist(),
            "percent_change": percent, "percent_ci": np.nanpercentile(percent_samples, [2.5, 97.5]).tolist(),
            "cohens_d": d_value, "d_ci": np.nanpercentile(d_samples, [2.5, 97.5]).tolist()}


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cliff's delta in O(n log n), positive when a tends to exceed b."""
    a, b = np.asarray(a, dtype=float), np.sort(np.asarray(b, dtype=float))
    greater = np.searchsorted(b, a, side="left")
    less_or_equal = np.searchsorted(b, a, side="right")
    less = len(b) - less_or_equal
    return float((greater.sum() - less.sum()) / (len(a) * len(b)))


def _bootstrap_cliffs_delta(a: np.ndarray, b: np.ndarray, seed: int) -> list[float]:
    """Percentile bootstrap CI; cap each empirical group sample to bound runtime."""
    rng = np.random.default_rng(seed)
    cap = 4_000
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a_base = rng.choice(a, size=min(cap, len(a)), replace=False)
    b_base = rng.choice(b, size=min(cap, len(b)), replace=False)
    estimates = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        a_sample = a_base[rng.integers(0, len(a_base), size=len(a_base))]
        b_sample = b_base[rng.integers(0, len(b_base), size=len(b_base))]
        estimates[index] = _cliffs_delta(a_sample, b_sample)
    return np.nanpercentile(estimates, [2.5, 97.5]).tolist()


def _magnitude(value: float, cutoffs: tuple[float, float] = (0.5, 0.8)) -> str:
    magnitude = abs(value)
    return "small" if magnitude < cutoffs[0] else "medium" if magnitude < cutoffs[1] else "large"


def _dunn_bonferroni(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Dunn pairwise comparisons with tie correction and Bonferroni adjustment."""
    clean = frame[["Weather_Condition", outcome]].dropna()
    clean = clean[clean["Weather_Condition"].isin(WEATHER_ORDER)]
    ranks = stats.rankdata(clean[outcome].to_numpy())
    clean = clean.assign(_rank=ranks)
    n_total = len(clean)
    tie_counts = pd.Series(clean[outcome]).value_counts().to_numpy()
    tie_correction = 1 - ((tie_counts**3 - tie_counts).sum() / (n_total**3 - n_total)) if n_total > 1 else 1
    variance = n_total * (n_total + 1) / 12 * tie_correction
    results = []
    pairs = list(itertools.combinations([name for name in WEATHER_ORDER if name in set(clean["Weather_Condition"])], 2))
    for first, second in pairs:
        a = clean.loc[clean["Weather_Condition"] == first, "_rank"]
        b = clean.loc[clean["Weather_Condition"] == second, "_rank"]
        z_score = (a.mean() - b.mean()) / math.sqrt(variance * (1 / len(a) + 1 / len(b)))
        results.append({"group_1": first, "group_2": second, "z": z_score, "p_raw": 2 * norm.sf(abs(z_score))})
    if results:
        adjusted = multipletests([row["p_raw"] for row in results], method="bonferroni")[1]
        for row, p_adj in zip(results, adjusted):
            row["p_bonferroni"] = float(p_adj)
    return pd.DataFrame(results)


def _shapiro_sample(values: pd.Series) -> float:
    sample = values.dropna().to_numpy()
    if len(sample) > 5_000:
        sample = np.random.default_rng(RANDOM_STATE).choice(sample, 5_000, replace=False)
    return float(stats.shapiro(sample).pvalue) if 3 <= len(sample) <= 5_000 else float("nan")


def _road_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate road-level rain effects and control the false-discovery rate."""
    rows = []
    for index, (road, group) in enumerate(data.groupby("Road_ID", sort=True)):
        dry = group.loc[group["Rainfall"] <= 0, "Traffic_Speed"].to_numpy()
        rain = group.loc[group["Rainfall"] > 0, "Traffic_Speed"].to_numpy()
        bootstrap = _bootstrap_difference(rain, dry, RANDOM_STATE + index + 100)
        test = stats.ttest_ind(rain, dry, equal_var=False)
        drop_pct = -bootstrap["percent_change"]
        if drop_pct >= HIGH_SENSITIVITY_DROP_PCT:
            sensitivity = "High"
        elif drop_pct >= MODERATE_SENSITIVITY_DROP_PCT:
            sensitivity = "Moderate"
        else:
            sensitivity = "Low"
        rows.append({
            "Road_ID": road, "n_rain": len(rain), "n_dry": len(dry),
            "mean_speed_rain": float(np.mean(rain)), "mean_speed_dry": float(np.mean(dry)),
            "speed_drop_pct": drop_pct, "speed_drop_ci_low": -bootstrap["percent_ci"][1],
            "speed_drop_ci_high": -bootstrap["percent_ci"][0],
            "cohens_d_rain_minus_dry": bootstrap["cohens_d"], "cohens_d_ci_low": bootstrap["d_ci"][0],
            "cohens_d_ci_high": bootstrap["d_ci"][1], "cohens_d_magnitude": _magnitude(bootstrap["cohens_d"]),
            "p_value": float(test.pvalue), "sensitivity": sensitivity,
        })
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = multipletests(result["p_value"], alpha=ALPHA, method="fdr_bh")[1]
    result["fdr_significant"] = result["p_fdr_bh"] < ALPHA
    return result.sort_values("speed_drop_pct", ascending=False).reset_index(drop=True)


def _make_inference_figures(data: pd.DataFrame, road_results: pd.DataFrame, binary_stats: dict,
                            adjusted_means: pd.DataFrame, dose: pd.DataFrame) -> None:
    """Create the four publication-ready statistical figures."""
    fig, ax = plt.subplots(figsize=(10.5, 6))
    sns.boxplot(data=data, x="Weather_Condition", y="Traffic_Speed", hue="Weather_Condition",
                order=WEATHER_ORDER, hue_order=WEATHER_ORDER, palette=PALETTE,
                legend=False, showfliers=False, ax=ax)
    ax.set(title="Traffic speed across observed weather conditions",
           xlabel="Weather condition", ylabel="Traffic speed (km/h)")
    annotation = (f"Rain vs dry: Welch t = {binary_stats['speed_welch_t']:.2f}, "
                  f"p = {binary_stats['speed_welch_p']:.2g}; Cohen's d = {binary_stats['speed_effect']['cohens_d']:.2f} "
                  f"(95% CI {binary_stats['speed_effect']['d_ci'][0]:.2f}, {binary_stats['speed_effect']['d_ci'][1]:.2f})")
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"})
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_speed_by_weather.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    pivot = adjusted_means.pivot(index="Time_of_Day", columns="Weather_Condition", values="adjusted_speed")
    pivot = pivot.reindex(index=TIME_ORDER, columns=WEATHER_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5,
                cbar_kws={"label": "Adjusted mean speed (km/h)"}, ax=ax)
    ax.set(title="Road- and weekday-adjusted speed by time and weather",
           xlabel="Weather condition", ylabel="Time of day")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_adjusted_speed_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    plot_data = road_results.sort_values("speed_drop_pct", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E76F51" if high else ("#E9C46A" if sig else "#6C8E9B")
              for high, sig in zip(plot_data["sensitivity"] == "High", plot_data["fdr_significant"])]
    xerr = np.vstack([
        np.maximum(0, plot_data["speed_drop_pct"] - plot_data["speed_drop_ci_low"]),
        np.maximum(0, plot_data["speed_drop_ci_high"] - plot_data["speed_drop_pct"]),
    ])
    ax.barh(plot_data["Road_ID"], plot_data["speed_drop_pct"], xerr=xerr, color=colors,
            alpha=0.9, capsize=3)
    ax.axvline(HIGH_SENSITIVITY_DROP_PCT, color="#E76F51", linestyle="--", linewidth=1,
               label=f"High threshold ({HIGH_SENSITIVITY_DROP_PCT:.0f}%)")
    ax.set(title="Rain-associated speed drop by road (95% bootstrap CI)",
           xlabel="Speed reduction in rain vs dry (%)", ylabel="Road")
    ax.legend(title="Red: high sensitivity; gold: FDR-significant")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_road_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for measure, label, color in [("Traffic_Speed", "Mean speed (km/h)", "#2A9D8F"),
                                  ("Traffic_Volume", "Mean volume (vehicles/hour)", "#E76F51")]:
        axis = axes[0] if measure == "Traffic_Speed" else axes[1]
        subset = dose.melt(id_vars=["rain_bin", "peak_group"], value_vars=[measure], var_name="measure", value_name="value")
        sns.lineplot(data=subset, x="rain_bin", y="value", hue="peak_group", marker="o",
                     palette={"Peak": color, "Off-peak": "#6C8E9B"}, ax=axis)
        axis.set(xlabel="Rainfall (mm/hour)", ylabel=label, title=f"{label} by rainfall dose")
        axis.legend(title="Time period")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_rainfall_dose_response.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def statistical_analysis(data: pd.DataFrame) -> dict:
    """Run group comparisons, stratification, robust regression, and heterogeneity tests."""
    group_test_rows, posthoc = [], {}
    for outcome in ["Traffic_Speed", "Traffic_Volume"]:
        groups = [data.loc[data["Weather_Condition"] == weather, outcome].dropna().to_numpy() for weather in WEATHER_ORDER]
        normality = {weather: _shapiro_sample(data.loc[data["Weather_Condition"] == weather, outcome]) for weather in WEATHER_ORDER}
        levene_p = float(stats.levene(*groups, center="median").pvalue)
        kw = stats.kruskal(*groups)
        n = sum(map(len, groups))
        epsilon_squared = max(0.0, (kw.statistic - len(groups) + 1) / (n - len(groups)))
        group_test_rows.append({"outcome": outcome, "test": "Kruskal-Wallis", "statistic": float(kw.statistic),
                                "p_value": float(kw.pvalue), "epsilon_squared": float(epsilon_squared),
                                "epsilon_magnitude": _magnitude(epsilon_squared, (0.06, 0.14)),
                                "levene_p": levene_p, "normality_p_min": float(np.nanmin(list(normality.values()))),
                                "normality_by_weather": json.dumps(normality)})
        posthoc[outcome] = _dunn_bonferroni(data, outcome)
        posthoc[outcome].to_csv(OUTPUT_DIR / f"posthoc_{outcome.lower()}.csv", index=False)
    group_tests = pd.DataFrame(group_test_rows)
    group_tests.to_csv(OUTPUT_DIR / "weather_group_tests.csv", index=False)

    rain = data["Rainfall"] > 0
    speed_rain = data.loc[rain, "Traffic_Speed"].to_numpy()
    speed_dry = data.loc[~rain, "Traffic_Speed"].to_numpy()
    volume_rain = data.loc[rain, "Traffic_Volume"].to_numpy()
    volume_dry = data.loc[~rain, "Traffic_Volume"].to_numpy()
    speed_welch = stats.ttest_ind(speed_rain, speed_dry, equal_var=False)
    volume_welch = stats.ttest_ind(volume_rain, volume_dry, equal_var=False)
    speed_effect = _bootstrap_difference(speed_rain, speed_dry, RANDOM_STATE + 1)
    volume_effect = _bootstrap_difference(volume_rain, volume_dry, RANDOM_STATE + 2)
    speed_delta = _cliffs_delta(speed_rain, speed_dry)
    volume_delta = _cliffs_delta(volume_rain, volume_dry)
    binary_stats = {
        "speed_welch_t": float(speed_welch.statistic), "speed_welch_p": float(speed_welch.pvalue),
        "volume_welch_t": float(volume_welch.statistic), "volume_welch_p": float(volume_welch.pvalue),
        "speed_effect": speed_effect, "volume_effect": volume_effect,
        "speed_cliffs_delta": speed_delta, "volume_cliffs_delta": volume_delta,
        "speed_cliffs_ci": _bootstrap_cliffs_delta(speed_rain, speed_dry, RANDOM_STATE + 3),
        "volume_cliffs_ci": _bootstrap_cliffs_delta(volume_rain, volume_dry, RANDOM_STATE + 4),
        "speed_cliffs_magnitude": _magnitude(speed_delta, (0.33, 0.474)),
        "volume_cliffs_magnitude": _magnitude(volume_delta, (0.33, 0.474)),
        "rain_rows": int(rain.sum()), "dry_rows": int((~rain).sum()),
    }
    binary_stats["speed_magnitude"] = _magnitude(speed_effect["cohens_d"])
    binary_stats["volume_magnitude"] = _magnitude(volume_effect["cohens_d"])
    binary_stats["speed_decision"] = "Reject H0" if speed_welch.pvalue < ALPHA else "Fail to reject H0"
    binary_stats["volume_decision"] = "Reject H0_volume" if volume_welch.pvalue < ALPHA else "Fail to reject H0_volume"

    data = data.copy()
    data["rain_binary"] = rain
    data["weekday_type"] = np.where(data["is_weekend"], "Weekend", "Weekday")
    stratified_rows = []
    for (time_name, day_type), group in data.groupby(["Time_of_Day", "weekday_type"], observed=True):
        wet_values = group.loc[group["rain_binary"], "Traffic_Speed"].to_numpy()
        dry_values = group.loc[~group["rain_binary"], "Traffic_Speed"].to_numpy()
        if len(wet_values) > 1 and len(dry_values) > 1:
            result = stats.ttest_ind(wet_values, dry_values, equal_var=False)
            effect = _cohens_d(wet_values, dry_values)
            stratified_rows.append({"Time_of_Day": time_name, "weekday_type": day_type,
                                   "n_rain": len(wet_values), "n_dry": len(dry_values),
                                   "mean_speed_rain": np.mean(wet_values), "mean_speed_dry": np.mean(dry_values),
                                   "speed_drop_pct": (np.mean(dry_values) - np.mean(wet_values)) / np.mean(dry_values) * 100,
                                   "cohens_d_rain_minus_dry": effect, "p_value": float(result.pvalue)})
    stratified = pd.DataFrame(stratified_rows)
    stratified["p_fdr_bh"] = multipletests(stratified["p_value"], method="fdr_bh")[1]
    stratified.to_csv(OUTPUT_DIR / "stratified_rain_comparisons.csv", index=False)

    # Requested continuous-rainfall specification with HC3 robust standard errors.
    base_formula = "Traffic_Speed ~ Rainfall + C(Time_of_Day) + C(Weekday) + C(Road_ID)"
    unadjusted_model = smf.ols("Traffic_Speed ~ Rainfall", data=data).fit()
    robust_model = smf.ols(base_formula, data=data).fit(cov_type="HC3")
    interaction_formula = "Traffic_Speed ~ Rainfall * C(Time_of_Day) + C(Weekday) + C(Road_ID)"
    interaction_model = smf.ols(interaction_formula, data=data).fit(cov_type="HC3")
    rainfall_ci = robust_model.conf_int().loc["Rainfall"].tolist()
    controlled = {
        "formula": base_formula, "rainfall_coefficient": float(robust_model.params["Rainfall"]),
        "rainfall_ci": [float(rainfall_ci[0]), float(rainfall_ci[1])],
        "rainfall_p": float(robust_model.pvalues["Rainfall"]), "r_squared": float(robust_model.rsquared),
        "interaction_formula": interaction_formula,
        "interaction_coefficients": {key: float(value) for key, value in interaction_model.params.items() if "Rainfall" in key},
        "interaction_ci": {key: [float(value) for value in interaction_model.conf_int().loc[key]] for key in interaction_model.params.index if "Rainfall" in key},
    }
    binary_stats["binary_speed_decision"] = binary_stats["speed_decision"]
    binary_stats["speed_decision"] = "Reject H0" if controlled["rainfall_p"] < ALPHA else "Fail to reject H0"
    controlled["decision"] = binary_stats["speed_decision"]
    with open(OUTPUT_DIR / "regression_summary.txt", "w", encoding="utf-8") as output:
        output.write(robust_model.summary().as_text())
        output.write("\n\nRainfall x Time_of_Day interaction model\n")
        output.write(interaction_model.summary().as_text())

    road_results = _road_sensitivity(data)
    road_results.to_csv(OUTPUT_DIR / "road_sensitivity.csv", index=False)
    additive = smf.ols("Traffic_Speed ~ Rainfall + C(Road_ID) + C(Time_of_Day) + C(Weekday)", data=data).fit()
    road_interaction = smf.ols("Traffic_Speed ~ Rainfall * C(Road_ID) + C(Time_of_Day) + C(Weekday)", data=data).fit()
    interaction_test = sm.stats.anova_lm(additive, road_interaction)
    heterogeneity = {
        "f_statistic": float(interaction_test.iloc[1]["F"]),
        "p_value": float(interaction_test.iloc[1]["Pr(>F)"]),
        "road_drop_spread_percentage_points": float(road_results["speed_drop_pct"].max() - road_results["speed_drop_pct"].min()),
        "high_sensitivity_roads": road_results.loc[road_results["sensitivity"] == "High", "Road_ID"].tolist(),
        "fdr_significant_roads": road_results.loc[road_results["fdr_significant"], "Road_ID"].tolist(),
        "decision": "Evidence of heterogeneous road-level rain effects" if interaction_test.iloc[1]["Pr(>F)"] < ALPHA else "Insufficient evidence of road-level heterogeneity",
    }
    dose_data = data.copy()
    dose_data["rain_bin"] = pd.cut(dose_data["Rainfall"], bins=[-0.001, 0, 2.5, 7.5, np.inf],
                                    labels=["0", "0-2.5", "2.5-7.5", "7.5+"])
    dose_data["peak_group"] = np.where(dose_data["Time_of_Day"].isin(["Morning Peak", "Evening Peak"]), "Peak", "Off-peak")
    dose = dose_data.groupby(["rain_bin", "peak_group"], observed=False)[["Traffic_Speed", "Traffic_Volume"]].mean().reset_index()
    dose.to_csv(OUTPUT_DIR / "rainfall_dose_response.csv", index=False)

    # Standardized cell predictions hold road and weekday mix constant.
    adjusted_model = smf.ols("Traffic_Speed ~ C(Time_of_Day) * C(Weather_Condition) + Rainfall + C(Weekday) + C(Road_ID)", data=data).fit()
    standard = pd.MultiIndex.from_product([data["Road_ID"].unique(), data["Weekday"].unique()], names=["Road_ID", "Weekday"]).to_frame(index=False)
    adjusted_rows = []
    for time_name in TIME_ORDER:
        for weather_name in WEATHER_ORDER:
            cell = data[(data["Time_of_Day"] == time_name) & (data["Weather_Condition"] == weather_name)]
            if cell.empty:
                continue
            predictions = standard.assign(Time_of_Day=time_name, Weather_Condition=weather_name,
                                          Rainfall=float(cell["Rainfall"].median()))
            adjusted_rows.append({"Time_of_Day": time_name, "Weather_Condition": weather_name,
                                  "adjusted_speed": float(adjusted_model.predict(predictions).mean())})
    adjusted_means = pd.DataFrame(adjusted_rows)
    adjusted_means.to_csv(OUTPUT_DIR / "adjusted_weather_speed.csv", index=False)
    _make_inference_figures(data, road_results, binary_stats, adjusted_means, dose)

    binary_stats["multi_group_tests"] = group_tests.to_dict(orient="records")
    binary_stats["posthoc"] = {name: frame.to_dict(orient="records") for name, frame in posthoc.items()}
    raw_controlled = {"raw_speed_rain_minus_dry": speed_effect["difference"],
                      "raw_speed_rain_minus_dry_ci": speed_effect["difference_ci"],
                      "raw_speed_percent_change": speed_effect["percent_change"],
                      "unadjusted_rainfall_per_mm": float(unadjusted_model.params["Rainfall"]),
                      "controlled_rainfall_per_mm": controlled["rainfall_coefficient"],
                      "controlled_rainfall_ci": controlled["rainfall_ci"]}
    return {"binary": binary_stats, "group_tests": group_tests, "stratified": stratified,
            "controlled": controlled, "road_sensitivity": road_results,
            "heterogeneity": heterogeneity, "dose": dose,
            "adjusted_means": adjusted_means, "raw_vs_controlled": raw_controlled}


def _model_metrics(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray,
                   name: str, variant: str) -> dict:
    return {"model": name, "variant": variant,
            "accuracy": accuracy_score(y_true, prediction),
            "precision": precision_score(y_true, prediction, zero_division=0),
            "recall": recall_score(y_true, prediction, zero_division=0),
            "f1": f1_score(y_true, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probability) if len(np.unique(y_true)) > 1 else np.nan}


def _model_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data[["Rainfall", "Weather_Condition", "Time_of_Day", "Weekday", "Road_ID",
                     "Traffic_Volume", "hour", "is_weekend"]].copy()
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / HOURS_PER_DAY)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / HOURS_PER_DAY)
    return features


def _preprocessor(include_volume: bool, scale_numeric: bool = False) -> ColumnTransformer:
    categorical = ["Weather_Condition", "Time_of_Day", "Weekday", "Road_ID"]
    numeric = ["Rainfall", "hour", "hour_sin", "hour_cos", "is_weekend"]
    if include_volume:
        numeric.append("Traffic_Volume")
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                              ("numeric", numeric_transformer, numeric)])


def _extract_rules(tree_model: DecisionTreeClassifier, feature_names: np.ndarray, maximum: int = 5) -> list[str]:
    """Return the most-supported readable leaf rules from a fitted tree."""
    structure = tree_model.tree_
    rules = []

    def visit(node: int, conditions: list[str]) -> None:
        left, right = structure.children_left[node], structure.children_right[node]
        if left == right:
            counts = structure.value[node][0]
            predicted = int(np.argmax(counts))
            total = int(structure.n_node_samples[node])
            confidence = float(counts[predicted] / max(counts.sum(), 1))
            rules.append((total, f"IF {' AND '.join(conditions) if conditions else 'all inputs'} THEN Congested={predicted} (leaf support={total}, training purity={confidence:.0%})"))
            return
        feature = feature_names[structure.feature[node]]
        threshold = structure.threshold[node]
        label = feature.replace("categorical__", "").replace("numeric__", "")
        if label.startswith("Weather_Condition_"):
            label = f"Weather={label.removeprefix('Weather_Condition_')}"
            left_condition, right_condition = f"NOT {label}", label
        elif label.startswith("Time_of_Day_"):
            label = f"Time={label.removeprefix('Time_of_Day_')}"
            left_condition, right_condition = f"NOT {label}", label
        elif label.startswith("Weekday_"):
            label = f"Weekday={label.removeprefix('Weekday_')}"
            left_condition, right_condition = f"NOT {label}", label
        elif label.startswith("Road_ID_"):
            label = f"Road={label.removeprefix('Road_ID_')}"
            left_condition, right_condition = f"NOT {label}", label
        else:
            left_condition, right_condition = f"{label} <= {threshold:.2f}", f"{label} > {threshold:.2f}"
        visit(left, conditions + [left_condition])
        visit(right, conditions + [right_condition])

    visit(0, [])
    return [text for _, text in sorted(rules, reverse=True)[:maximum]]


def train_and_evaluate_models(data: pd.DataFrame) -> dict:
    """Tune a time-aware decision tree and compare downstream/baseline models."""
    data = data.sort_values(["Timestamp", "Road_ID"]).reset_index(drop=True).copy()
    start = data["Timestamp"].min()
    duration = data["Timestamp"].max() - start + pd.Timedelta(hours=1)
    split_time = start + duration * 0.8333333333
    train_mask = data["Timestamp"] < split_time
    train = data.loc[train_mask].copy()
    test = data.loc[~train_mask].copy()
    if train.empty or test.empty:
        raise ValueError("Time-aware split produced an empty train or test set.")

    # The target threshold uses training-period speeds only, never held-out observations.
    free_flow = train.groupby("Road_ID")["Traffic_Speed"].quantile(FREE_FLOW_PERCENTILE)
    data["free_flow_train"] = data["Road_ID"].map(free_flow)
    data["Congested"] = (data["Traffic_Speed"] < CONGESTION_FREE_FLOW_SHARE * data["free_flow_train"]).astype(int)
    train = data.loc[train_mask].copy()
    test = data.loc[~train_mask].copy()
    y_train, y_test = train["Congested"].to_numpy(), test["Congested"].to_numpy()
    x_train, x_test = _model_features(train), _model_features(test)
    print(f"Time split: train through {train['Timestamp'].max()}, test from {test['Timestamp'].min()}")
    print(f"Free-flow training-only thresholds (km/h):\n{free_flow.round(1).to_string()}")
    print(f"Class balance, train: {pd.Series(y_train).value_counts(normalize=True).sort_index().to_dict()}")
    print(f"Class balance, test: {pd.Series(y_test).value_counts(normalize=True).sort_index().to_dict()}")
    print("Traffic_Volume is a legitimate operational sensor feature, but can partly mediate the weather response; the no-volume variant checks remaining rain signal.")

    cv = TimeSeriesSplit(n_splits=5)
    parameter_grid = {
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_leaf": [20, 50],
        "classifier__min_samples_split": [2, 10],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__class_weight": [None, "balanced"],
        "classifier__ccp_alpha": [0.0, 0.001],
    }
    metric_rows, cv_rows, best_models = [], [], {}
    tree_for_figures = None
    for variant, include_volume in [("with_volume", True), ("without_volume", False)]:
        feature_subset_train = x_train if include_volume else x_train.drop(columns="Traffic_Volume")
        feature_subset_test = x_test if include_volume else x_test.drop(columns="Traffic_Volume")
        tree_pipeline = Pipeline([("preprocessor", _preprocessor(include_volume)),
                                  ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))])
        grid = GridSearchCV(tree_pipeline, parameter_grid, scoring="roc_auc", cv=cv, n_jobs=1,
                            refit=True, return_train_score=False, error_score="raise")
        grid.fit(feature_subset_train, y_train)
        best = grid.best_estimator_
        best_models[variant] = best
        predicted = best.predict(feature_subset_test)
        probability = best.predict_proba(feature_subset_test)[:, 1]
        metric_rows.append(_model_metrics(y_test, predicted, probability, "Decision tree", variant))
        cv_rows.append({"model": "Decision tree", "variant": variant, "cv_roc_auc_mean": grid.best_score_,
                        "cv_roc_auc_std": float(grid.cv_results_["std_test_score"][grid.best_index_]),
                        "best_params": json.dumps(grid.best_params_, default=str)})
        print(f"Best tree {variant}: CV ROC-AUC={grid.best_score_:.3f} +/- {cv_rows[-1]['cv_roc_auc_std']:.3f}; {grid.best_params_}")
        if include_volume:
            tree_for_figures = (best, y_test, predicted)

    # Logistic regression is a linear comparator; majority prediction anchors class imbalance.
    logistic = Pipeline([("preprocessor", _preprocessor(True, scale_numeric=True)),
                         ("classifier", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE))])
    logistic.fit(x_train, y_train)
    logistic_pred = logistic.predict(x_test)
    logistic_prob = logistic.predict_proba(x_test)[:, 1]
    metric_rows.append(_model_metrics(y_test, logistic_pred, logistic_prob, "Logistic regression", "with_volume"))
    cv_rows.append({"model": "Logistic regression", "variant": "with_volume", "cv_roc_auc_mean": np.nan, "cv_roc_auc_std": np.nan, "best_params": ""})
    majority_class = int(np.bincount(y_train).argmax())
    majority_pred = np.full(len(y_test), majority_class)
    metric_rows.append(_model_metrics(y_test, majority_pred, majority_pred.astype(float), "Majority baseline", "baseline"))
    cv_rows.append({"model": "Majority baseline", "variant": "baseline", "cv_roc_auc_mean": np.nan, "cv_roc_auc_std": np.nan, "best_params": ""})
    metrics = pd.DataFrame(metric_rows).merge(pd.DataFrame(cv_rows), on=["model", "variant"], how="left")
    metrics["train_rows"], metrics["test_rows"] = len(train), len(test)
    metrics["positive_rate_test"] = float(y_test.mean())
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    joblib.dump(best_models["with_volume"], OUTPUT_DIR / "decision_tree_model.joblib")

    best, y_test, predicted = tree_for_figures
    transformed_names = best.named_steps["preprocessor"].get_feature_names_out()
    tree = best.named_steps["classifier"]
    importances = pd.DataFrame({"feature": transformed_names, "importance": tree.feature_importances_}).sort_values("importance", ascending=False)
    importances.to_csv(OUTPUT_DIR / "tree_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(tree, feature_names=transformed_names, class_names=["Not congested", "Congested"],
              filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title("Tuned decision tree (display limited to depth 3)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_decision_tree.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    top_importance = importances.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_importance["feature"].str.replace("categorical__", "", regex=False).str.replace("numeric__", "", regex=False),
            top_importance["importance"], color="#2A9D8F")
    ax.set(title="Decision-tree feature importance", xlabel="Normalized importance", ylabel="Encoded feature")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=["Not congested", "Congested"], yticklabels=["Not congested", "Congested"], ax=ax)
    ax.set(title="Held-out decision-tree confusion matrix", xlabel="Predicted class", ylabel="Observed class")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    rules = _extract_rules(tree, transformed_names)
    return {"metrics": metrics, "cv": pd.DataFrame(cv_rows), "rules": rules, "confusion_matrix": matrix.tolist(),
            "feature_importance": importances, "free_flow_train": free_flow.to_dict(),
            "train_rows": len(train), "test_rows": len(test), "split_time": split_time.isoformat(),
            "positive_rate_test": float(y_test.mean()), "best_params": grid.best_params_,
            "model_path": "outputs/decision_tree_model.joblib"}


def _fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def write_report(data: pd.DataFrame, quality: dict, eda: dict, inference: dict, model: dict) -> str:
    """Build a result-driven planner report and a PDF export when ReportLab is available."""
    binary = inference["binary"]
    road = inference["road_sensitivity"]
    controlled = inference["controlled"]
    tests = inference["group_tests"].set_index("outcome")
    tree_metrics = model["metrics"].query("model == 'Decision tree' and variant == 'with_volume'").iloc[0]
    tree_no_volume_metrics = model["metrics"].query("model == 'Decision tree' and variant == 'without_volume'").iloc[0]
    logistic_metrics = model["metrics"].query("model == 'Logistic regression'").iloc[0]
    baseline_metrics = model["metrics"].query("model == 'Majority baseline'").iloc[0]
    top_roads = road.head(5)
    top_names = ", ".join(top_roads["Road_ID"].tolist())
    decision_speed = binary["speed_decision"]
    peak_cells = inference["stratified"][inference["stratified"]["Time_of_Day"].isin(["Morning Peak", "Evening Peak"])]
    peak_cell = peak_cells.sort_values("speed_drop_pct", ascending=False).iloc[0]
    summary = (
        f"Across {len(data):,} cleaned hourly sensor readings from {data['Road_ID'].nunique()} roads, "
        f"rain was associated with a {abs(binary['speed_effect']['percent_change']):.1f}% lower mean speed "
        f"(Cohen's d={binary['speed_effect']['cohens_d']:.2f}; 95% CI "
        f"{binary['speed_effect']['d_ci'][0]:.2f} to {binary['speed_effect']['d_ci'][1]:.2f}). "
        f"After controlling for time of day, weekday, and road, each additional mm/hour was associated "
        f"with {abs(controlled['rainfall_coefficient']):.2f} km/h lower speed "
        f"(95% CI {controlled['rainfall_ci'][0]:.2f} to {controlled['rainfall_ci'][1]:.2f}). "
        f"Effects varied by corridor; prioritize {top_names}. These are observational/synthetic associations, not causal estimates."
    )
    key_insights = [
        f"Rain vs dry speed: {binary['speed_effect']['percent_change']:.1f}% mean change (95% bootstrap CI {binary['speed_effect']['percent_ci'][0]:.1f}% to {binary['speed_effect']['percent_ci'][1]:.1f}%); Cohen's d={binary['speed_effect']['cohens_d']:.2f} ({binary['speed_magnitude']}); Cliff's delta={binary['speed_cliffs_delta']:.2f} (95% bootstrap CI {binary['speed_cliffs_ci'][0]:.2f} to {binary['speed_cliffs_ci'][1]:.2f}; {binary['speed_cliffs_magnitude']}). Welch p={_fmt_p(binary['speed_welch_p'])}; controlled-H0 decision: {decision_speed}.",
        f"Controlled rainfall slope: {controlled['rainfall_coefficient']:.2f} km/h per mm/hour (robust HC3 95% CI {controlled['rainfall_ci'][0]:.2f} to {controlled['rainfall_ci'][1]:.2f}; p={_fmt_p(controlled['rainfall_p'])}); model R-squared={controlled['r_squared']:.3f}.",
        f"Weather-group speed test: Kruskal-Wallis H={tests.loc['Traffic_Speed', 'statistic']:.1f}, p={_fmt_p(tests.loc['Traffic_Speed', 'p_value'])}, epsilon-squared={tests.loc['Traffic_Speed', 'epsilon_squared']:.3f} ({tests.loc['Traffic_Speed', 'epsilon_magnitude']}); Dunn pairwise results use Bonferroni correction.",
        f"Rain-related volume change: {binary['volume_effect']['percent_change']:.1f}% (95% bootstrap CI {binary['volume_effect']['percent_ci'][0]:.1f}% to {binary['volume_effect']['percent_ci'][1]:.1f}%); Cohen's d={binary['volume_effect']['cohens_d']:.2f}; Welch p={_fmt_p(binary['volume_welch_p'])}; decision: {binary['volume_decision']}.",
        f"Road effects are heterogeneous: road-by-rain interaction F={inference['heterogeneity']['f_statistic']:.1f}, p={_fmt_p(inference['heterogeneity']['p_value'])}; observed drop spread={inference['heterogeneity']['road_drop_spread_percentage_points']:.1f} percentage points. The largest peak-period stratified drop was {peak_cell['speed_drop_pct']:.1f}% in {peak_cell['Time_of_Day']} ({peak_cell['weekday_type']}); high-sensitivity roads: {', '.join(inference['heterogeneity']['high_sensitivity_roads']) or 'none'}.",
        f"The largest estimated rain-associated speed drops are {', '.join(f'{row.Road_ID} ({row.speed_drop_pct:.1f}%, d={row.cohens_d_rain_minus_dry:.2f}, FDR q={row.p_fdr_bh:.3g})' for row in top_roads.itertuples())}.",
        f"The held-out decision tree achieved ROC-AUC={tree_metrics['roc_auc']:.3f}, F1={tree_metrics['f1']:.3f}, recall={tree_metrics['recall']:.3f}; without volume, ROC-AUC remained {tree_no_volume_metrics['roc_auc']:.3f} but default-threshold recall was {tree_no_volume_metrics['recall']:.3f}. Logistic regression ROC-AUC={logistic_metrics['roc_auc']:.3f}, majority baseline ROC-AUC={baseline_metrics['roc_auc']:.3f}. Tree CV ROC-AUC={tree_metrics['cv_roc_auc_mean']:.3f} +/- {tree_metrics['cv_roc_auc_std']:.3f}.",
    ]
    actions = [
        ("Inspect drainage, underpass pumps, and flood paths", top_names, "Road sensitivity ranking; prioritize High roads and FDR-significant corridors", "High", "0-6 months", "Reduce standing water and preserve wet-weather capacity"),
        ("Pilot rain-triggered adaptive signal timing", "High-sensitivity arterials; peak approaches", "Rain x peak-hour speed effects and dose-response figure", "High", "3-9 months", "Reduce queue growth during rainfall peaks"),
        ("Deploy dynamic speed limits and variable message signs", "High- and Moderate-sensitivity corridors", "Observed speed losses and heavy-rain dose response", "High", "3-6 months", "Smooth speeds and warn drivers before bottlenecks"),
        ("Offer rerouting and corridor alerts", ", ".join(top_roads.head(3)["Road_ID"]), "Largest road-level rain effects; revalidate against live data", "Med", "3-6 months", "Divert trips before constrained links saturate"),
        ("Staff incident response for peak-hour rain events", "Citywide peak periods; deploy first to ranked roads", "Rain effects are stronger/heterogeneous across corridors", "Med", "Next wet season", "Shorten incident clearance and queue duration"),
        ("Track post-deployment KPIs", "Citywide and by road", "Baseline: speed drop in rain, congestion hours/rain event, incident rate", "High", "Monthly; evaluate after 6 months", "Quantify operational benefit and avoid relying on modeled association"),
    ]
    action_table = "| Action | Roads/Area | Evidence | Priority | Timeline | Expected impact |\n|---|---|---|---|---|---|\n"
    for row in actions:
        action_table += "| " + " | ".join(str(item).replace("|", "/") for item in row) + " |\n"
    insights_md = "\n".join(f"{index + 1}. {insight}" for index, insight in enumerate(key_insights))
    rules_md = "\n".join(f"- `{rule}`" for rule in model["rules"])
    road_table = "| Rank | Road | Speed drop | 95% bootstrap CI | Cohen's d (95% CI) | FDR q | Class |\n|---:|---|---:|---:|---:|---:|---|\n"
    for rank, row in enumerate(road.head(10).itertuples(), start=1):
        road_table += (f"| {rank} | {row.Road_ID} | {row.speed_drop_pct:.1f}% | "
                       f"[{row.speed_drop_ci_low:.1f}, {row.speed_drop_ci_high:.1f}]% | "
                       f"{row.cohens_d_rain_minus_dry:.2f} [{row.cohens_d_ci_low:.2f}, {row.cohens_d_ci_high:.2f}] | "
                       f"{row.p_fdr_bh:.3g} | {row.sensitivity} |\n")
    report = f"""# Smart City: Does Weather Affect Traffic?

**Speed null (H0):** Rainfall has no effect on mean traffic speed: mean speed is equal in rain and dry conditions after controlling for time of day and weekday.  
**Speed alternative (H1):** Rainfall reduces traffic speed and therefore is associated with increased congestion.  
**Volume null (H0-volume):** Rainfall does not change mean traffic volume.  
**Volume alternative (H1-volume):** Rainfall changes mean traffic volume.  
Significance level: alpha = {ALPHA:.2f}.

## Executive summary
{summary}

## Formal test and raw-versus-controlled comparison
The primary speed test is OLS with HC3 robust standard errors, adjusted for time of day, weekday, and road. Decision: **{decision_speed}** (rainfall coefficient {controlled['rainfall_coefficient']:.2f} km/h per mm/hour; 95% CI {controlled['rainfall_ci'][0]:.2f}, {controlled['rainfall_ci'][1]:.2f}; p={_fmt_p(controlled['rainfall_p'])}). The simple binary rain-vs-dry Welch test gives a {binary['speed_effect']['difference']:.2f} km/h difference ({binary['speed_effect']['percent_change']:.1f}%; p={_fmt_p(binary['speed_welch_p'])}; binary-test decision: {binary['binary_speed_decision']}).

The unadjusted continuous rainfall slope is {inference['raw_vs_controlled']['unadjusted_rainfall_per_mm']:.2f} km/h per mm/hour; after controlling for road, time of day, and weekday it is {controlled['rainfall_coefficient']:.2f} km/h per mm/hour. The binary contrast compares all wet hours with dry hours and therefore also reflects rainfall intensity and the mix of observations; the regression coefficients compare one additional mm/hour. Their different magnitudes are not directly interchangeable. The slope change after adjustment describes the contribution of road/calendar/time composition in this dataset; remaining unmeasured conditions can still confound the association. Volume test decision: **{binary['volume_decision']}** (Welch p={_fmt_p(binary['volume_welch_p'])}).

## Key insights
{insights_md}

## Visual evidence
![Traffic speed by weather](outputs/figures/01_speed_by_weather.png)

![Adjusted weather and time heatmap](outputs/figures/02_adjusted_speed_heatmap.png)

![Ranked road sensitivity](outputs/figures/03_road_sensitivity.png)

![Rainfall dose-response](outputs/figures/04_rainfall_dose_response.png)

![Decision tree](outputs/figures/05_decision_tree.png)

![Feature importance](outputs/figures/06_feature_importance.png)

![Confusion matrix](outputs/figures/07_confusion_matrix.png)

## Road sensitivity ranking
Sensitivity thresholds, stated before interpretation: High = at least {HIGH_SENSITIVITY_DROP_PCT:.0f}% lower speed in rain; Moderate = {MODERATE_SENSITIVITY_DROP_PCT:.0f}% to under {HIGH_SENSITIVITY_DROP_PCT:.0f}%; Low = under {MODERATE_SENSITIVITY_DROP_PCT:.0f}%. P-values are adjusted across roads with Benjamini-Hochberg FDR at alpha={ALPHA:.2f}. Cohen's d magnitude labels use 0.2/0.5/0.8 as small/medium/large; epsilon-squared uses 0.01/0.06/0.14.

{road_table}

## Decision-tree interpretation
Congestion is defined as speed below {CONGESTION_FREE_FLOW_SHARE:.0%} of that road's training-period 85th-percentile speed. The free-flow reference uses only the first approximately 75 of 90 days. No Traffic_Speed feature is used. Traffic_Volume is a plausible sensor input available to a traffic operator, but can be partly downstream of weather-related diversion and congestion; compare both tree variants.

Held-out metrics (last approximately 15 days): accuracy {tree_metrics['accuracy']:.3f}, precision {tree_metrics['precision']:.3f}, recall {tree_metrics['recall']:.3f}, F1 {tree_metrics['f1']:.3f}, ROC-AUC {tree_metrics['roc_auc']:.3f}; confusion matrix [[{model['confusion_matrix'][0][0]}, {model['confusion_matrix'][0][1]}], [{model['confusion_matrix'][1][0]}, {model['confusion_matrix'][1][1]}]]. Tuned tree time-series CV ROC-AUC: {tree_metrics['cv_roc_auc_mean']:.3f} +/- {tree_metrics['cv_roc_auc_std']:.3f}. Without volume, ROC-AUC is {tree_no_volume_metrics['roc_auc']:.3f} (with-volume: {tree_metrics['roc_auc']:.3f}), but default-threshold recall is only {tree_no_volume_metrics['recall']:.3f}, so the ranking signal remains while positive-class decisions are inadequate. Logistic-regression ROC-AUC: {logistic_metrics['roc_auc']:.3f}; majority baseline ROC-AUC: {baseline_metrics['roc_auc']:.3f}. See `outputs/model_metrics.csv` for all metrics.

Extracted tree rules (encoded categories are indicator splits; 1 means the named category is present):
{rules_md}

## Data quality and assumptions
Input source: `{quality['source']}`; synthetic data: **{quality['synthetic']}**; rows after cleaning: {len(data):,}. The synthetic observations are generated with seed {RANDOM_STATE}; they are not evidence about a real city. {quality['invalid_timestamps_dropped']} unparseable timestamps were dropped. Exact duplicates were removed; impossible speeds outside 5-90 km/h, negative volume, and negative rainfall were set missing and imputed by road/time medians or rainfall/weather rules. Time period and weekday were recalculated from timestamp. Valid IQR outliers were retained and listed in `outputs/iqr_outlier_summary.csv`.

The group normality checks use Shapiro-Wilk samples capped at 5,000 observations; Levene tests assess variance equality. Because large samples make normality tests sensitive and group variances may differ, Kruskal-Wallis plus Dunn-Bonferroni is reported for weather groups, alongside Welch tests for the binary comparison. Bootstrap intervals use {BOOTSTRAP_RESAMPLES:,} seeded resamples. Statistical significance is not practical significance.

These are observational associations. Rainfall may coincide with unmeasured accidents, roadworks, events, visibility changes, or route choice. Synthetic-data effects are designed into the generator and must not be presented as city measurements. Congestion is a policy-defined threshold, class prevalence affects accuracy, and Traffic_Volume may be downstream of rainfall. Validate with audited local sensors, incident/event covariates, and prospective pilots before operational deployment.

## Practical action plan
{action_table}

## Reproducibility
Run `python -m src.traffic_weather_analysis` from the project root, or run the notebook top to bottom. The model is saved at `outputs/decision_tree_model.joblib`; source and all generated artifacts are listed in `README.md`.
"""
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    _write_pdf_report(summary, key_insights, actions, tree_metrics, logistic_metrics)
    return report


def _write_pdf_report(summary: str, insights: list[str], actions: list[tuple],
                      tree_metrics: pd.Series, logistic_metrics: pd.Series) -> None:
    """Export a concise PDF with key findings, figures, and actions."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    except ImportError:
        warnings.warn("ReportLab is unavailable; skipping optional PDF export.")
        return
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCentered", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#264653")))
    story = [Paragraph("Smart City: Does Weather Affect Traffic?", styles["TitleCentered"]), Spacer(1, 0.18 * inch),
             Paragraph("Executive summary", styles["Heading2"]), Paragraph(summary, styles["BodyText"]), Spacer(1, 0.12 * inch),
             Paragraph("Key findings", styles["Heading2"])]
    for item in insights:
        story.append(Paragraph("&bull; " + item.replace("&", "&amp;"), styles["BodyText"]))
        story.append(Spacer(1, 0.06 * inch))
    story.extend([Spacer(1, 0.12 * inch), Paragraph("Model evaluation", styles["Heading2"]),
                  Paragraph(f"Decision tree: ROC-AUC {tree_metrics['roc_auc']:.3f}, F1 {tree_metrics['f1']:.3f}; logistic regression ROC-AUC {logistic_metrics['roc_auc']:.3f}.", styles["BodyText"]),
                  PageBreak(), Paragraph("Visual evidence", styles["Heading1"])])
    for image_name in ["01_speed_by_weather.png", "02_adjusted_speed_heatmap.png", "03_road_sensitivity.png",
                       "04_rainfall_dose_response.png", "06_feature_importance.png", "07_confusion_matrix.png"]:
        image_path = FIGURE_DIR / image_name
        if image_path.exists():
            story.append(Image(str(image_path), width=6.8 * inch, height=3.55 * inch, kind="proportional"))
            story.append(Spacer(1, 0.08 * inch))
    story.extend([PageBreak(), Paragraph("Practical action plan", styles["Heading1"])])
    table_rows = [["Action", "Roads / area", "Priority", "Timeline"]]
    for action, area, evidence, priority, timeline, impact in actions:
        table_rows.append([Paragraph(action, styles["BodyText"]), Paragraph(area, styles["BodyText"]), priority, timeline])
    table = Table(table_rows, colWidths=[2.0 * inch, 1.7 * inch, 0.65 * inch, 1.2 * inch], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#264653")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c7c9")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf4f2")]),
                               ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 5)]))
    story.append(table)
    story.extend([Spacer(1, 0.18 * inch), Paragraph("Limitations", styles["Heading2"]),
                  Paragraph("The data are synthetic in this run; results demonstrate the reproducible method, not a real city's traffic response. Observational association is not causation. Congestion threshold is a design choice; accidents, roadworks, events, visibility, and route choice are not modeled.", styles["BodyText"])])
    document = SimpleDocTemplate(str(ROOT / "REPORT.pdf"), pagesize=letter, rightMargin=0.65 * inch,
                                 leftMargin=0.65 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch)
    document.build(story)


def run_all() -> dict:
    """Execute every project stage and return concise results for the notebook."""
    data, quality = load_and_clean()
    eda = exploratory_analysis(data)
    inference = statistical_analysis(data)
    model = train_and_evaluate_models(data)
    report = write_report(data, quality, eda, inference, model)
    print("\nPrimary results")
    print(f"Speed H0 decision: {inference['binary']['speed_decision']} (Welch p={inference['binary']['speed_welch_p']:.3g})")
    print(f"Controlled rainfall slope: {inference['controlled']['rainfall_coefficient']:.3f} km/h/mm; 95% CI {inference['controlled']['rainfall_ci']}")
    print(f"Road heterogeneity: {inference['heterogeneity']['decision']} (p={inference['heterogeneity']['p_value']:.3g})")
    tree_roc = model["metrics"].query("model == 'Decision tree' and variant == 'with_volume'").iloc[0]["roc_auc"]
    print(f"Best tree test ROC-AUC: {tree_roc:.3f}")
    print(f"Report: {ROOT / 'REPORT.md'}; PDF: {ROOT / 'REPORT.pdf'}")
    return {"data": data, "quality": quality, "eda": eda, "inference": inference, "model": model, "report": report}


if __name__ == "__main__":
    run_all()