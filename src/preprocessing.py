"""Loading, cleaning, feature engineering, and PM2.5 categorization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "date_and_time", "pm25", "o3", "temperature", "relative_humidity",
    "dew_point", "precipitation", "rain", "pressure_msl",
    "surface_pressure", "cloud_cover", "vapour_pressure_deficit",
    "wind_speed", "wind_direction", "wind_gusts",
]

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}


def load_and_clean_data(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load the CSV, validate its schema, and apply conservative cleaning.

    Numeric gaps are interpolated by time only when bounded on both sides, then
    filled with the month-hour median (and finally the global median). This
    preserves seasonal and diurnal structure better than a single global value.
    """
    data = pd.read_csv(path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    report = {
        "rows_loaded": len(data),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_before": data.isna().sum().to_dict(),
    }
    data = data.drop_duplicates().copy()
    data["date_and_time"] = pd.to_datetime(data["date_and_time"], errors="coerce")
    invalid_dates = int(data["date_and_time"].isna().sum())
    data = data.dropna(subset=["date_and_time"]).sort_values("date_and_time")

    numeric_columns = [c for c in EXPECTED_COLUMNS if c != "date_and_time"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data = data.set_index("date_and_time")
    data[numeric_columns] = data[numeric_columns].interpolate(
        method="time", limit=3, limit_area="inside"
    )
    month_hour = [data.index.month, data.index.hour]
    for column in numeric_columns:
        grouped_median = data[column].groupby(month_hour).transform("median")
        data[column] = data[column].fillna(grouped_median).fillna(data[column].median())

    data = data.reset_index()
    data["hour"] = data["date_and_time"].dt.hour
    data["month"] = data["date_and_time"].dt.month
    data["season"] = data["month"].map(SEASON_BY_MONTH)

    report.update({
        "invalid_dates_removed": invalid_dates,
        "rows_after_cleaning": len(data),
        "missing_after": data.isna().sum().to_dict(),
    })
    return data, report


def categorize_pm25(series: pd.Series) -> pd.Series:
    """Categorize PM2.5 using US EPA 24-hour concentration breakpoints.

    Values are hourly observations, so these labels are concentration-severity
    categories for pattern mining and must not be interpreted as official AQI.
    """
    bins = [-np.inf, 9.0, 35.4, 55.4, 125.4, 225.4, np.inf]
    labels = ["Good", "Moderate", "USG", "Unhealthy", "Very_Unhealthy", "Hazardous"]
    return pd.cut(series, bins=bins, labels=labels, ordered=True)

