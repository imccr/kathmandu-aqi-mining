from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


HIGH_PM25_THRESHOLD = 55.5


CLASSIFIER_FEATURES = [
    "temperature", "relative_humidity", "dew_point", "precipitation",
    "pressure_msl", "cloud_cover", "vapour_pressure_deficit",
    "wind_speed", "wind_direction", "wind_gusts",
]


def make_labels(data: pd.DataFrame, threshold: float = HIGH_PM25_THRESHOLD) -> pd.Series:
    return (data["pm25"] >= threshold).astype(int)


def build_models(random_state: int = 42) -> dict:
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=random_state
        ),
        "Naive Bayes": GaussianNB(),
        "k-NN (k=15)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=15)
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_state
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
    }


def chronological_split(
    data: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = HIGH_PM25_THRESHOLD,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Timestamp]:
    features = features or CLASSIFIER_FEATURES
    ordered = data.sort_values("date_and_time").reset_index(drop=True)
    cut = int(len(ordered) * train_fraction)
    features_frame = ordered[features]
    labels = make_labels(ordered, threshold)
    split_time = ordered["date_and_time"].iloc[cut - 1]
    return (
        features_frame.iloc[:cut], features_frame.iloc[cut:],
        labels.iloc[:cut], labels.iloc[cut:], split_time,
    )


def _score_row(name: str, y_true, y_pred, y_proba) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def evaluate_holdout(
    data: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = HIGH_PM25_THRESHOLD,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict, tuple[pd.DataFrame, pd.Series]]:
    features = features or CLASSIFIER_FEATURES
    X_train, X_test, y_train, y_test, _ = chronological_split(
        data, features, threshold, train_fraction
    )
    fitted, rows = {}, []
    for name, model in build_models(random_state).items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        rows.append(_score_row(name, y_test, predictions, probabilities))
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return results, fitted, (X_test, y_test)


def naive_random_holdout(
    data: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = HIGH_PM25_THRESHOLD,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    features = features or CLASSIFIER_FEATURES
    X, y = data[features], make_labels(data, threshold)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    rows = []
    for name, model in build_models(random_state).items():
        model.fit(X_train, y_train)
        rows.append(
            _score_row(
                name, y_test, model.predict(X_test),
                model.predict_proba(X_test)[:, 1],
            )
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def rolling_cv_scores(
    data: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = HIGH_PM25_THRESHOLD,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    features = features or CLASSIFIER_FEATURES
    X, y = data.sort_values("date_and_time")[features], make_labels(
        data.sort_values("date_and_time"), threshold
    )
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for name, model in build_models(random_state).items():
        auc = cross_val_score(model, X, y, cv=splitter, scoring="roc_auc")
        f1 = cross_val_score(model, X, y, cv=splitter, scoring="f1")
        rows.append({
            "model": name,
            "cv_auc_mean": auc.mean(), "cv_auc_std": auc.std(),
            "cv_f1_mean": f1.mean(), "cv_f1_std": f1.std(),
        })
    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)


def seasonal_confound_auc(
    data: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = HIGH_PM25_THRESHOLD,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    features = features or CLASSIFIER_FEATURES
    frame = data.sort_values("date_and_time").copy()
    frame["month"] = frame["date_and_time"].dt.month
    frame["hour"] = frame["date_and_time"].dt.hour
    labels = make_labels(frame, threshold)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    feature_sets = {
        "weather only": features,
        "time only (month + hour)": ["month", "hour"],
        "weather + time": features + ["month", "hour"],
    }
    rows = []
    for label, columns in feature_sets.items():
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
        auc = cross_val_score(
            model, frame[columns], labels, cv=splitter, scoring="roc_auc"
        )
        rows.append({
            "feature_set": label, "n_features": len(columns),
            "cv_auc_mean": auc.mean(), "cv_auc_std": auc.std(),
        })
    return pd.DataFrame(rows)


def tree_feature_importance(
    model: DecisionTreeClassifier, features: list[str] | None = None
) -> pd.Series:
    features = features or CLASSIFIER_FEATURES
    return pd.Series(model.feature_importances_, index=features).sort_values(
        ascending=False
    )


def tree_rules_text(
    model: DecisionTreeClassifier,
    features: list[str] | None = None,
    max_depth: int = 3,
) -> str:
    features = features or CLASSIFIER_FEATURES
    return export_text(model, feature_names=list(features), max_depth=max_depth)


def confusion_frame(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    matrix = confusion_matrix(y_test, model.predict(X_test))
    return pd.DataFrame(
        matrix,
        index=["actual: Not high", "actual: High"],
        columns=["predicted: Not high", "predicted: High"],
    )


def roc_points(model, X_test: pd.DataFrame, y_test: pd.Series):
    probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    return fpr, tpr, roc_auc_score(y_test, probabilities)
