"""
Module de clustering des IRIS français.
Méthodes implémentées :
  - KMeans (principal)
  - Clustering hiérarchique agglomératif
  - Sélection automatique du nombre optimal de clusters
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import logging

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    - Nettoyage des colonnes totalement vides
    - Imputation des valeurs manquantes (médiane)
    - Standardisation (StandardScaler)
    """
    if feature_cols is None:
        exclude = {"IRIS", "COM", "LIBCOM", "TYP_IRIS", "LAB_IRIS", "UU2020",
                   "REG", "DEP", "TRIRIS", "GRD_QUART", "LIBIRIS", "NOMCOM",
                   "CODGEO", "LIBGEO"}
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

    # --- CORRECTION : Nettoyage pour éviter Found array with 0 sample(s) ---
    # 1. Supprimer les colonnes qui sont 100% vides (NaN)
    df_temp = df[feature_cols].dropna(axis=1, how='all')
    feature_cols = df_temp.columns.tolist()
    
    logger.info(f"Features sélectionnées ({len(feature_cols)})")

    # 2. Filtrage des lignes : on garde les IRIS qui ont au moins 1 donnée valide
    df_feat = df[["IRIS"] + feature_cols].copy()
    mask = df_feat[feature_cols].notna().any(axis=1)
    df_feat = df_feat[mask]

    if len(df_feat) == 0:
        logger.error("ERREUR CRITIQUE : Aucune donnée exploitable après filtrage des NaN.")
        return None, None, None, None, None

    iris_codes = df_feat["IRIS"].values
    X = df_feat[feature_cols].values

    # Imputation (remplissage des trous par la médiane)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # Standardisation (moyenne 0, écart-type 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    logger.info(f"Matrice préparée avec succès : {X_scaled.shape}")
    return X_scaled, iris_codes, feature_cols, scaler, imputer


# ---------------------------------------------------------------------------
# Sélection et Modèles
# ---------------------------------------------------------------------------

def find_optimal_k(X_scaled: np.ndarray, k_min: int = 3, k_max: int = 10, random_state: int = 42):
    results = []
    # On réduit l'échantillon pour la silhouette si le dataset est énorme
    sample_size = min(5000, len(X_scaled))
    
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=random_state)
        db = davies_bouldin_score(X_scaled, labels)
        
        results.append({
            "k": k, "inertia": km.inertia_, "silhouette": sil, "davies_bouldin": db
        })
        logger.info(f"Test k={k} terminé.")

    df_metrics = pd.DataFrame(results)
    k_optimal = int(df_metrics.loc[df_metrics["silhouette"].idxmax(), "k"])
    return df_metrics, k_optimal

def run_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=15)
    labels = km.fit_predict(X_scaled)
    return labels, km

def compute_pca(X_scaled: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled), pca

def compute_umap(X_scaled: np.ndarray):
    if not UMAP_AVAILABLE:
        return None, None
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30)
    return reducer.fit_transform(X_scaled), reducer

def profile_clusters(df_original: pd.DataFrame, iris_codes: np.ndarray, labels: np.ndarray, feature_cols: list):
    df_result = pd.DataFrame({"IRIS": iris_codes, "cluster": labels})
    df_merged = df_result.merge(df_original[["IRIS"] + feature_cols], on="IRIS", how="left")
    profile = df_merged.groupby("cluster")[feature_cols].mean().round(2)
    profile["n_iris"] = df_merged.groupby("cluster")["IRIS"].count()
    return df_merged, profile

def save_results(df_labeled, profile, df_metrics, method="kmeans"):
    df_labeled.to_parquet(os.path.join(RESULTS_DIR, f"iris_clustered_{method}.parquet"), index=False)
    profile.to_csv(os.path.join(RESULTS_DIR, f"cluster_profile_{method}.csv"))
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_k_selection.csv"), index=False)

# ---------------------------------------------------------------------------
# Pipeline Principal
# ---------------------------------------------------------------------------

def run_full_pipeline(df: pd.DataFrame):
    # 1. Préparation
    X_scaled, iris_codes, feature_cols, scaler, imputer = prepare_features(df)
    if X_scaled is None: return None

    # 2. Recherche du K optimal
    df_metrics, k_opt = find_optimal_k(X_scaled)
    logger.info(f"K optimal détecté : {k_opt}")

    # 3. Clustering Final
    labels, model = run_kmeans(X_scaled, k_opt)

    # 4. Réductions de dimension pour la Map 2D
    X_pca, _ = compute_pca(X_scaled)
    X_umap, _ = compute_umap(X_scaled)

    # 5. Profiling et Sauvegarde
    df_labeled, profile = profile_clusters(df, iris_codes, labels, feature_cols)
    df_labeled["pca_x"], df_labeled["pca_y"] = X_pca[:, 0], X_pca[:, 1]
    if X_umap is not None:
        df_labeled["umap_x"], df_labeled["umap_y"] = X_umap[:, 0], X_umap[:, 1]

    save_results(df_labeled, profile, df_metrics)
    return {"k": k_opt, "profile": profile}

if __name__ == "__main__":
    from data_collection import build_dataset
    df_raw = build_dataset()
    results = run_full_pipeline(df_raw)
    if results:
        print(f"\n✅ Clustering terminé avec {results['k']} clusters.")