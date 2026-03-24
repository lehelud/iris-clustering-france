"""
Module de clustering des IRIS français.
Version finale :
  - Compatible avec dataset enrichi (data_collection.py)
  - Pipeline robuste
  - Sélection du nombre optimal de clusters améliorée
  - Nettoyage renforcé
  - PCA + UMAP
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import logging

# ---------------------------------------------------------------------------
# Tentative d'import UMAP
# ---------------------------------------------------------------------------
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Préparation des features
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, feature_cols: list = None):
    """
    Prépare la matrice de features :
      - Suppression colonnes vides
      - Imputation médiane
      - Standardisation
    """

    if feature_cols is None:
        exclude = {
            "IRIS", "COM", "LIBCOM", "TYP_IRIS", "LAB_IRIS", "UU2020",
            "REG", "DEP", "TRIRIS", "GRD_QUART", "LIBIRIS", "NOMCOM",
            "CODGEO", "LIBGEO"
        }
        feature_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]
        ]

    # Suppression colonnes 100% vides
    df_temp = df[feature_cols].dropna(axis=1, how="all")
    feature_cols = df_temp.columns.tolist()

    logger.info(f"{len(feature_cols)} features retenues.")

    df_feat = df[["IRIS"] + feature_cols].copy()
    df_feat = df_feat[df_feat[feature_cols].notna().any(axis=1)]

    if df_feat.empty:
        logger.error("Aucune donnée exploitable après nettoyage.")
        return None, None, None, None, None

    iris_codes = df_feat["IRIS"].values
    X = df_feat[feature_cols].values

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    logger.info(f"Matrice finale : {X_scaled.shape}")
    return X_scaled, iris_codes, feature_cols, scaler, imputer

# ---------------------------------------------------------------------------
# Sélection du K optimal
# ---------------------------------------------------------------------------

def find_optimal_k(X_scaled: np.ndarray, k_min: int = 3, k_max: int = 12):
    results = []
    sample_size = min(5000, len(X_scaled))

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels, sample_size=sample_size)
        db = davies_bouldin_score(X_scaled, labels)

        results.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        logger.info(f"Test K={k} → silhouette={sil:.3f}")

    df_metrics = pd.DataFrame(results)

    # Sélection robuste : silhouette - 0.1 * Davies-Bouldin
    df_metrics["score"] = df_metrics["silhouette"] - df_metrics["davies_bouldin"] * 0.1
    k_optimal = int(df_metrics.loc[df_metrics["score"].idxmax(), "k"])

    return df_metrics, k_optimal

# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    return labels, km

def compute_pca(X_scaled):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled), pca

def compute_umap(X_scaled):
    if not UMAP_AVAILABLE:
        return None, None
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30)
    return reducer.fit_transform(X_scaled), reducer

# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_clusters(df_original, iris_codes, labels, feature_cols):
    df_result = pd.DataFrame({"IRIS": iris_codes, "cluster": labels})
    df_merged = df_result.merge(df_original[["IRIS"] + feature_cols], on="IRIS", how="left")

    profile = df_merged.groupby("cluster")[feature_cols].mean().round(2)
    profile["n_iris"] = df_merged.groupby("cluster")["IRIS"].count()

    return df_merged, profile

# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_results(df_labeled, profile, df_metrics):
    df_labeled.to_parquet(os.path.join(RESULTS_DIR, "iris_clustered.parquet"), index=False)
    profile.to_csv(os.path.join(RESULTS_DIR, "cluster_profile.csv"))
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_k_selection.csv"), index=False)

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_full_pipeline(df):

    logger.info("Dataset déjà enrichi via data_collection.py — aucune étape supplémentaire.")

    # Préparation
    X_scaled, iris_codes, feature_cols, scaler, imputer = prepare_features(df)
    if X_scaled is None:
        return None

    # Sélection du K optimal
    df_metrics, k_opt = find_optimal_k(X_scaled)
    logger.info(f"K optimal = {k_opt}")

    # Clustering final
    labels, model = run_kmeans(X_scaled, k_opt)

    # Réductions de dimension
    X_pca, _ = compute_pca(X_scaled)
    X_umap, _ = compute_umap(X_scaled)

    # Profiling
    df_labeled, profile = profile_clusters(df, iris_codes, labels, feature_cols)
    df_labeled["pca_x"], df_labeled["pca_y"] = X_pca[:, 0], X_pca[:, 1]

    if X_umap is not None:
        df_labeled["umap_x"], df_labeled["umap_y"] = X_umap[:, 0], X_umap[:, 1]

    save_results(df_labeled, profile, df_metrics)

    return {"k": k_opt, "profile": profile}

# ---------------------------------------------------------------------------
# Exécution directe
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_collection import build_dataset
    df_raw = build_dataset()
    results = run_full_pipeline(df_raw)
    if results:
        print(f"\n✅ Clustering terminé avec {results['k']} clusters.")
