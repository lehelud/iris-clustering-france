"""
Module de clustering des IRIS français.
Méthodes implémentées :
  - KMeans (principal)
  - Clustering hiérarchique agglomératif
  - DBSCAN
Sélection automatique du nombre optimal de clusters (Elbow + Silhouette).
Réduction de dimension pour visualisation (PCA, UMAP).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import logging

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, feature_cols: list = None):
    """
    Prépare la matrice de features :
    - Imputation des valeurs manquantes (médiane)
    - Standardisation (StandardScaler)
    Retourne X_scaled, iris_codes, feature_names
    """
    if feature_cols is None:
        # Exclure les colonnes non numériques et identifiants
        exclude = {"IRIS", "COM", "LIBCOM", "TYP_IRIS", "LAB_IRIS", "UU2020",
                   "REG", "DEP", "TRIRIS", "GRD_QUART", "LIBIRIS", "NOMCOM",
                   "CODGEO", "LIBGEO"}
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

    logger.info(f"Features sélectionnées ({len(feature_cols)}) : {feature_cols}")

    # Garder uniquement les lignes avec au moins 50% de valeurs non nulles
    df_feat = df[["IRIS"] + feature_cols].copy()
    threshold = 0.5 * len(feature_cols)
    mask = df_feat[feature_cols].notna().sum(axis=1) >= threshold
    df_feat = df_feat[mask]

    iris_codes = df_feat["IRIS"].values
    X = df_feat[feature_cols].values

    # Imputation
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    logger.info(f"Matrice préparée : {X_scaled.shape}")
    return X_scaled, iris_codes, feature_cols, scaler, imputer


# ---------------------------------------------------------------------------
# Sélection du nombre optimal de clusters
# ---------------------------------------------------------------------------

def find_optimal_k(X_scaled: np.ndarray, k_min: int = 3, k_max: int = 12, random_state: int = 42):
    """
    Calcule l'inertie (Elbow) et le score de silhouette pour k de k_min à k_max.
    Retourne le DataFrame des métriques et le k optimal (meilleur silhouette).
    """
    results = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(labels)), random_state=random_state)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        results.append({
                                   "k": k,
                                   "inertia": inertia,
                                   "silhouette": sil,
                                   "davies_bouldin": db,
                                   "calinski_harabasz": ch
                               })
        logger.info(f"k={k} | Inertie={inertia:.0f} | Silhouette={sil:.4f} | DB={db:.4f}")

    df_metrics = pd.DataFrame(results)
    k_optimal = int(df_metrics.loc[df_metrics["silhouette"].idxmax(), "k"])
    logger.info(f"Nombre optimal de clusters (silhouette max) : k={k_optimal}")
    return df_metrics, k_optimal


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

def run_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42):
    """Lance KMeans avec k clusters. Retourne les labels et le modèle."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=15, max_iter=500)
    labels = km.fit_predict(X_scaled)
    logger.info(f"KMeans k={k} : distribution = {dict(zip(*np.unique(labels, return_counts=True)))}")
    return labels, km


# ---------------------------------------------------------------------------
# Clustering Hiérarchique
# ---------------------------------------------------------------------------

def run_hierarchical(X_scaled: np.ndarray, k: int, linkage_method: str = "ward"):
    """Lance le clustering hiérarchique agglomératif."""
    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = hc.fit_predict(X_scaled)
    logger.info(f"Hiérarchique k={k} ({linkage_method}) : {dict(zip(*np.unique(labels, return_counts=True)))}")
    return labels, hc


def compute_linkage_matrix(X_scaled: np.ndarray, method: str = "ward"):
    """Calcule la matrice de linkage pour le dendrogramme."""
    return linkage(X_scaled[:2000], method=method)  # Limite à 2000 points pour perf


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def run_dbscan(X_scaled: np.ndarray, eps: float = 0.5, min_samples: int = 10):
    """Lance DBSCAN. Le label -1 correspond aux outliers."""
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(f"DBSCAN eps={eps} : {n_clusters} clusters, {n_noise} outliers")
    return labels, db


# ---------------------------------------------------------------------------
# Réduction de dimension
# ---------------------------------------------------------------------------

def compute_pca(X_scaled: np.ndarray, n_components: int = 2):
    """Réduit la dimension avec PCA."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    logger.info(f"PCA {n_components}D : variance expliquée = {explained.sum():.2%}")
    return X_pca, pca


def compute_umap(X_scaled: np.ndarray, n_components: int = 2):
    """Réduit la dimension avec UMAP (si disponible)."""
    if not UMAP_AVAILABLE:
        logger.warning("UMAP non disponible, utilisation de PCA à la place.")
        return compute_pca(X_scaled, n_components)
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=30, min_dist=0.1)
    X_umap = reducer.fit_transform(X_scaled)
    return X_umap, reducer


# ---------------------------------------------------------------------------
# Profiling des clusters
# ---------------------------------------------------------------------------

def profile_clusters(df_original: pd.DataFrame, iris_codes: np.ndarray,
                                                       labels: np.ndarray, feature_cols: list):
    """
    Calcule le profil (moyenne standardisée) de chaque cluster.
    Retourne un DataFrame avec le profil et les statistiques descriptives.
    """
    df_result = pd.DataFrame({"IRIS": iris_codes, "cluster": labels})
    df_merged = df_result.merge(df_original[["IRIS"] + feature_cols], on="IRIS", how="left")

    # Stats par cluster
    profile = df_merged.groupby("cluster")[feature_cols].mean().round(2)
    profile["n_iris"] = df_merged.groupby("cluster")["IRIS"].count()
    profile = profile.sort_index()

    return df_merged, profile


# ---------------------------------------------------------------------------
# Sauvegarde des résultats
# ---------------------------------------------------------------------------

def save_results(df_labeled: pd.DataFrame, profile: pd.DataFrame,
                                              df_metrics: pd.DataFrame, method: str = "kmeans"):
    """Sauvegarde les résultats du clustering."""
    df_labeled.to_parquet(os.path.join(RESULTS_DIR, f"iris_clustered_{method}.parquet"), index=False)
    profile.to_csv(os.path.join(RESULTS_DIR, f"cluster_profile_{method}.csv"))
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_k_selection.csv"), index=False)
    logger.info(f"Résultats sauvegardés dans '{RESULTS_DIR}/'")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_full_pipeline(df: pd.DataFrame, k: int = None, method: str = "kmeans"):
    """
    Pipeline complet :
    1. Préparation des features
    2. Sélection du k optimal (si non fourni)
    3. Clustering
    4. Réduction de dimension (PCA)
    5. Profiling
    6. Sauvegarde
    """
    X_scaled, iris_codes, feature_cols, scaler, imputer = prepare_features(df)

    # Sélection K optimal
    df_metrics, k_optimal = find_optimal_k(X_scaled)
    if k is None:
        k = k_optimal

    # Clustering
    if method == "kmeans":
        labels, model = run_kmeans(X_scaled, k)
    elif method == "hierarchical":
        labels, model = run_hierarchical(X_scaled, k)
    elif method == "dbscan":
        labels, model = run_dbscan(X_scaled)
        k = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        raise ValueError(f"Méthode inconnue : {method}")

    # Réduction de dimension
    X_pca, pca_model = compute_pca(X_scaled)
    X_umap, umap_model = compute_umap(X_scaled)

    # Profiling
    df_labeled, profile = profile_clusters(df, iris_codes, labels, feature_cols)

    # Ajout des coordonnées de visualisation
    df_labeled["pca_x"] = X_pca[:, 0]
    df_labeled["pca_y"] = X_pca[:, 1]
    if X_umap is not None and X_umap.shape[1] >= 2:
        df_labeled["umap_x"] = X_umap[:, 0]
        df_labeled["umap_y"] = X_umap[:, 1]

    # Sauvegarde
    save_results(df_labeled, profile, df_metrics, method=method)

    return {
        "df_labeled": df_labeled,
        "profile": profile,
        "df_metrics": df_metrics,
        "k": k,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "X_umap": X_umap,
        "feature_cols": feature_cols,
        "model": model,
        "pca": pca_model,
    }


if __name__ == "__main__":
    from data_collection import build_dataset
    df = build_dataset()
    results = run_full_pipeline(df)
    print(f"\nClustering terminé : {results['k']} clusters")
    print(results["profile"])
