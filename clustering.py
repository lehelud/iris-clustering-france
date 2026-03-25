"""
Module de clustering des IRIS français.
Version améliorée :
  - Ingénierie de features : ratios sociodémographiques (PCT_CS*, LOG_POP, PCT_IMM, etc.)
  - Sélection du K optimal automatique (silhouette + davies_bouldin + calinski_harabasz)
  - Nommage automatique des clusters basé sur les profils relatifs
  - PCA + UMAP
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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
# Features préférées pour le clustering (ratios vs comptes bruts)
# Les ratios rendent le clustering invariant à la taille de l'IRIS
# ---------------------------------------------------------------------------

PREFERRED_FEATURES = [
    # Taille (log pour réduire l'asymétrie)
    'LOG_POP',
    # Structure socio-professionnelle (% pop 15+)
    'PCT_CS1', 'PCT_CS2', 'PCT_CS3', 'PCT_CS4', 'PCT_CS5', 'PCT_CS6', 'PCT_CS7', 'PCT_CS8',
    # Démographie
    'PART_JEUNES', 'PART_SENIORS', 'PCT_ACTIFS',
    # Diversité
    'PCT_IMM', 'PCT_ETR',
    # Économie locale
    'DENSITE_ETAB',
    # Marché du travail et logement (si disponibles)
    'TAUX_CHOM', 'TAUX_VACANCE', 'TAUX_PROPRIO',
]

# ---------------------------------------------------------------------------
# Ingénierie de features : ratios sociodémographiques
# ---------------------------------------------------------------------------

def engineer_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features normalisées (ratios/pourcentages) à partir des comptes bruts.
    Rend le clustering invariant à la taille de l'IRIS, ce qui améliore
    significativement la qualité de la segmentation territoriale.
    """
    result = df.copy()

    def get_col(*candidates):
        for c in candidates:
            if c in result.columns:
                return result[c]
        return None

    def safe_ratio(num, den):
        n = pd.to_numeric(num, errors='coerce')
        d = pd.to_numeric(den, errors='coerce').replace(0, np.nan)
        r = n / d
        return r.where(np.isfinite(r), other=np.nan)

    pop = get_col('P21_POP', 'P20_POP')
    pop15p_csp = get_col('C21_POP15P', 'C20_POP15P')

    # Taille de l'IRIS (log-transformée pour réduire l'asymétrie de distribution)
    if pop is not None:
        result['LOG_POP'] = np.log1p(pd.to_numeric(pop, errors='coerce'))

    # Catégories socio-professionnelles en % de la pop 15+
    if pop15p_csp is not None:
        for cs_num in range(1, 9):
            cs_col = get_col(f'C21_POP15P_CS{cs_num}', f'C20_POP15P_CS{cs_num}')
            if cs_col is not None:
                result[f'PCT_CS{cs_num}'] = safe_ratio(cs_col, pop15p_csp)

    if pop is not None:
        # Proportion de la population en âge actif (20-64 ans)
        pop2064 = get_col('P21_POP2064', 'P20_POP2064')
        if pop2064 is not None:
            result['PCT_ACTIFS'] = safe_ratio(pop2064, pop)

        # Proportion d'immigrés et d'étrangers
        pop_imm = get_col('P21_POP_IMM', 'P20_POP_IMM')
        if pop_imm is not None:
            result['PCT_IMM'] = safe_ratio(pop_imm, pop)

        pop_etr = get_col('P21_POP_ETR', 'P20_POP_ETR')
        if pop_etr is not None:
            result['PCT_ETR'] = safe_ratio(pop_etr, pop)

        # Densité d'établissements pour 1000 habitants
        nb_etab = get_col('NB_ETABLISSEMENTS')
        if nb_etab is not None:
            result['DENSITE_ETAB'] = safe_ratio(nb_etab, pop) * 1000

    new_cols = [c for c in result.columns if c not in df.columns]
    logger.info(f"Features ratio créées : {new_cols}")
    return result


# ---------------------------------------------------------------------------
# Préparation des features
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, feature_cols: list = None):
    """
    Prépare la matrice de features :
      - Utilise en priorité les features ratio (PREFERRED_FEATURES) si disponibles
      - Suppression colonnes vides
      - Imputation médiane
      - Standardisation
    """
    exclude = {
        "IRIS", "COM", "LIBCOM", "TYP_IRIS", "LAB_IRIS", "UU2020",
        "REG", "DEP", "TRIRIS", "GRD_QUART", "LIBIRIS", "NOMCOM",
        "CODGEO", "LIBGEO",
        "NB_DEPLACEMENTS",  # toujours 0, sans valeur discriminante
    }

    if feature_cols is None:
        # Priorité aux features ratio pour un clustering qualitatif
        preferred_available = [
            c for c in PREFERRED_FEATURES
            if c in df.columns and df[c].dtype in [np.float64, np.int64, float, int, np.float32]
        ]

        # Compléter avec les features Filosofi (revenus, pauvreté, inégalités) si disponibles
        filosofi_cols = [
            c for c in df.columns
            if any(c.upper().startswith(p) for p in ['DISP_', 'TP6', 'GI', 'D1'])
            and df[c].dtype in [np.float64, np.int64, float, int, np.float32]
        ]

        if len(preferred_available) >= 5:
            feature_cols = list(dict.fromkeys(preferred_available + filosofi_cols))
            logger.info(f"Features ratio sélectionnées en priorité ({len(feature_cols)} features).")
        else:
            # Fallback : toutes les colonnes numériques (comportement original)
            feature_cols = [
                c for c in df.columns
                if c not in exclude
                and df[c].dtype in [np.float64, np.int64, float, int, np.float32]
            ]
            logger.info(f"Fallback : toutes les features numériques ({len(feature_cols)} features).")

    # Suppression colonnes 100% vides
    df_temp = df[feature_cols].dropna(axis=1, how="all")
    feature_cols = df_temp.columns.tolist()

    logger.info(f"{len(feature_cols)} features retenues : {feature_cols}")

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
        ch = calinski_harabasz_score(X_scaled, labels)

        results.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "inertia": km.inertia_,
        })
        logger.info(
            f"Test K={k} → silhouette={sil:.3f}, "
            f"davies_bouldin={db:.3f}, calinski_harabasz={ch:.1f}"
        )

    df_metrics = pd.DataFrame(results)

    # Score composite normalisé : silhouette (60%) + Davies-Bouldin inversé (25%) + Calinski-Harabasz (15%)
    def norm01(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    sil_norm = norm01(df_metrics["silhouette"])
    db_norm = 1 - norm01(df_metrics["davies_bouldin"])   # inversé : plus bas = mieux
    ch_norm = norm01(df_metrics["calinski_harabasz"])

    df_metrics["score"] = 0.60 * sil_norm + 0.25 * db_norm + 0.15 * ch_norm
    k_optimal = int(df_metrics.loc[df_metrics["score"].idxmax(), "k"])

    logger.info(f"K optimal sélectionné : {k_optimal}")
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

    profile = df_merged.groupby("cluster")[feature_cols].mean().round(4)
    profile["n_iris"] = df_merged.groupby("cluster")["IRIS"].count()

    return df_merged, profile


# ---------------------------------------------------------------------------
# Nommage automatique des clusters
# ---------------------------------------------------------------------------

def name_clusters(profile: pd.DataFrame) -> dict:
    """
    Attribue un nom descriptif à chaque cluster en comparant leurs profils
    via des z-scores relatifs sur les dimensions clés.

    Returns
    -------
    dict : {cluster_id: str}
    """

    def best_col(*candidates):
        for c in candidates:
            if c in profile.columns:
                return c
        return None

    def z_series(col):
        """Z-score inter-cluster pour identifier les dimensions distinctives."""
        if col is None:
            return pd.Series(0.0, index=profile.index)
        vals = pd.to_numeric(profile[col], errors='coerce')
        std = vals.std()
        if std > 1e-9:
            return (vals - vals.mean()) / std
        return pd.Series(0.0, index=profile.index)

    # Détection des colonnes clés (avec fallbacks P21_/P20_)
    col_pop     = best_col('LOG_POP', 'P21_POP', 'P20_POP')
    col_cs1     = best_col('PCT_CS1', 'C20_POP15P_CS1', 'C21_POP15P_CS1')
    col_cs2     = best_col('PCT_CS2', 'C20_POP15P_CS2', 'C21_POP15P_CS2')
    col_cs3     = best_col('PCT_CS3', 'C20_POP15P_CS3', 'C21_POP15P_CS3')
    col_cs5     = best_col('PCT_CS5', 'C20_POP15P_CS5', 'C21_POP15P_CS5')
    col_cs6     = best_col('PCT_CS6', 'C20_POP15P_CS6', 'C21_POP15P_CS6')
    col_cs7     = best_col('PCT_CS7', 'C20_POP15P_CS7', 'C21_POP15P_CS7')
    col_jeunes  = best_col('PART_JEUNES')
    col_seniors = best_col('PART_SENIORS')
    col_imm     = best_col('PCT_IMM', 'P20_POP_IMM', 'P21_POP_IMM')
    col_chom    = best_col('TAUX_CHOM')
    col_vacance = best_col('TAUX_VACANCE')
    col_proprio = best_col('TAUX_PROPRIO')

    z_pop     = z_series(col_pop)
    z_cs1     = z_series(col_cs1)
    z_cs2     = z_series(col_cs2)
    z_cs3     = z_series(col_cs3)
    z_cs5     = z_series(col_cs5)
    z_cs6     = z_series(col_cs6)
    z_cs7     = z_series(col_cs7)
    z_jeunes  = z_series(col_jeunes)
    z_seniors = z_series(col_seniors)
    z_imm     = z_series(col_imm)
    z_chom    = z_series(col_chom)
    z_vacance = z_series(col_vacance)
    z_proprio = z_series(col_proprio)

    T = 0.6  # seuil z-score : caractéristique considérée comme distinctive

    names = {}

    for cid in profile.index:

        # --- 1. Urbanisation / taille ---
        if z_pop[cid] < -T:
            urban = "Rural"
        elif z_pop[cid] > T:
            urban = "Urbain dense"
        else:
            urban = "Périurbain / bourg"

        # --- 2. Type socio-économique ---
        socio = []
        if z_cs1[cid] > T:
            socio.append("agricole")
        if z_cs3[cid] > T:
            socio.append("aisé (cadres)")
        elif z_cs3[cid] < -T and z_cs6[cid] > T:
            socio.append("ouvrier")
        elif z_cs3[cid] < -T and z_cs5[cid] > T:
            socio.append("employés")
        if z_imm[cid] > T and z_cs5[cid] > T:
            socio.append("populaire & diversifié")
        elif z_imm[cid] > T:
            socio.append("forte immigration")
        if z_cs2[cid] > T:
            socio.append("commerçants / artisans")
        if z_chom[cid] > T:
            socio.append("précarité")

        # --- 3. Structure démographique ---
        demo = []
        if z_seniors[cid] > T:
            demo.append("vieillissant")
        elif z_jeunes[cid] > T:
            demo.append("jeune")
        if z_cs7[cid] > T:
            demo.append("retraités")

        # --- 4. Logement ---
        if z_proprio[cid] > T:
            demo.append("propriétaires")
        elif z_vacance[cid] > T:
            demo.append("logements vacants")

        # Construction du nom
        parts = [urban]
        if socio:
            parts.append(", ".join(socio))
        if demo:
            parts.append(", ".join(demo))

        names[cid] = " — ".join(parts)

    return names


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_results(df_labeled, profile, df_metrics, cluster_names=None):
    df_labeled.to_parquet(os.path.join(RESULTS_DIR, "iris_clustered.parquet"), index=False)
    profile.to_csv(os.path.join(RESULTS_DIR, "cluster_profile.csv"))
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_k_selection.csv"), index=False)

    if cluster_names:
        names_path = os.path.join(RESULTS_DIR, "cluster_names.json")
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in cluster_names.items()}, f, ensure_ascii=False, indent=2)
        logger.info(f"Noms des clusters sauvegardés : {names_path}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_full_pipeline(df):

    # Ingénierie de features : création des ratios sociodémographiques
    logger.info("Ingénierie de features : création des ratios sociodémographiques...")
    df_enriched = engineer_ratio_features(df)

    # Préparation de la matrice (priorité aux features ratio)
    X_scaled, iris_codes, feature_cols, scaler, imputer = prepare_features(df_enriched)
    if X_scaled is None:
        return None

    # Sélection du K optimal
    df_metrics, k_opt = find_optimal_k(X_scaled)
    logger.info(f"K optimal = {k_opt}")

    # Clustering final avec k optimal (mettre k_force à une valeur entière pour forcer)
    k_force = None  # None = utiliser k_opt automatiquement
    k_final = k_force if k_force else k_opt
    labels, model = run_kmeans(X_scaled, k_final)

    # Réductions de dimension
    X_pca, _ = compute_pca(X_scaled)
    X_umap, _ = compute_umap(X_scaled)

    # Profiling (utiliser le df enrichi pour avoir les features ratio dans le profil)
    df_labeled, profile = profile_clusters(df_enriched, iris_codes, labels, feature_cols)
    df_labeled["pca_x"], df_labeled["pca_y"] = X_pca[:, 0], X_pca[:, 1]

    if X_umap is not None:
        df_labeled["umap_x"], df_labeled["umap_y"] = X_umap[:, 0], X_umap[:, 1]

    # Nommage des clusters
    cluster_names = name_clusters(profile)
    logger.info("Noms des clusters :")
    for cid, name in cluster_names.items():
        logger.info(f"  Cluster {cid} ({profile.loc[cid, 'n_iris']:,} IRIS) : {name}")

    save_results(df_labeled, profile, df_metrics, cluster_names)

    return {"k": k_final, "profile": profile, "names": cluster_names}


# ---------------------------------------------------------------------------
# Exécution directe
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_collection import build_dataset
    df_raw = build_dataset()
    results = run_full_pipeline(df_raw)
    if results:
        print(f"\n✅ Clustering terminé avec {results['k']} clusters.")
        print("\nNoms des clusters :")
        for cid, name in results['names'].items():
            print(f"  Cluster {cid} : {name}")
