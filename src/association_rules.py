from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from .preprocessing import categorize_pm25

BIN_FEATURES = [
    "temperature", "relative_humidity", "wind_speed", "pressure_msl",
    "cloud_cover", "vapour_pressure_deficit",
]


def _tertile_labels(series: pd.Series, name: str) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, 3, labels=[f"{name}=Low", f"{name}=Medium", f"{name}=High"])


def build_transactions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical = pd.DataFrame(index=data.index)
    for feature in BIN_FEATURES:
        categorical[feature] = _tertile_labels(data[feature], feature)

    wet = (data["precipitation"].fillna(0) > 0) | (data["rain"].fillna(0) > 0)
    categorical["rain_state"] = np.where(wet, "rain=Wet", "rain=Dry")
    categorical["wind_direction"] = pd.cut(
        data["wind_direction"] % 360,
        bins=[-0.001, 45, 135, 225, 315, 360],
        labels=["wind=N", "wind=E", "wind=S", "wind=W", "wind=N"],
        ordered=False,
    )
    severity = categorize_pm25(data["pm25"])
    categorical["pm25_severity"] = "PM25=" + severity.astype("string")

    transactions = categorical.astype("string").values.tolist()
    encoder = TransactionEncoder()
    encoded = pd.DataFrame(
        encoder.fit(transactions).transform(transactions),
        columns=encoder.columns_, index=data.index,
    )
    return encoded, categorical


def mine_high_pm25_rules(
    encoded: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.35,
    min_lift: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    itemsets = apriori(encoded, min_support=min_support, use_colnames=True, max_len=4)
    if itemsets.empty:
        return itemsets, pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    targets = {"PM25=Unhealthy", "PM25=Very_Unhealthy", "PM25=Hazardous"}
    rules = rules[
        rules["consequents"].map(lambda values: len(values) == 1 and bool(values & targets))
        & rules["antecedents"].map(lambda values: not any(str(v).startswith("PM25=") for v in values))
        & (rules["lift"] >= min_lift)
    ].copy()
    rules["antecedent_text"] = rules["antecedents"].map(lambda x: " & ".join(sorted(x)))
    rules["consequent_text"] = rules["consequents"].map(lambda x: next(iter(x)))
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    return itemsets, rules

