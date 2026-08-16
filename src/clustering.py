from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = [
    "temperature", "relative_humidity", "precipitation", "pressure_msl",
    "cloud_cover", "vapour_pressure_deficit", "wind_speed",
]


def evaluate_kmeans(
    data: pd.DataFrame,
    features: list[str] | None = None,
    k_values=range(2, 9),
    random_state: int = 42,
) -> tuple[pd.DataFrame, StandardScaler]:
    
    features = features or DEFAULT_FEATURES
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[features])
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(scaled)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(scaled, labels),
        })
    return pd.DataFrame(rows), scaler


def fit_kmeans(
    data: pd.DataFrame,
    n_clusters: int,
    features: list[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler, pd.DataFrame]:
    
    features = features or DEFAULT_FEATURES
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[features])
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    result = data.copy()
    result["cluster"] = model.fit_predict(scaled)
    projection = PCA(n_components=2).fit_transform(scaled)
    pca_frame = pd.DataFrame(projection, columns=["PC1", "PC2"], index=result.index)
    pca_frame["cluster"] = result["cluster"]
    return result, model, scaler, pca_frame


def cluster_profile(
    clustered: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    
    features = features or DEFAULT_FEATURES
    profile = clustered.groupby("cluster")[[*features, "pm25"]].mean().round(2)
    profile.insert(0, "n_observations", clustered.groupby("cluster").size())
    profile["share_percent"] = (100 * profile["n_observations"] / len(clustered)).round(1)
    return profile

