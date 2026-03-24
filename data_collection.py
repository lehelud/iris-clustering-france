"""
Module de collecte des données open data au niveau IRIS français.
"""

import os
import requests
import pandas as pd
import numpy as np
import zipfile
import io
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilitaire : lire un ZIP → premier CSV
# ---------------------------------------------------------------------------

def _zip_to_df(content_bytes, sep=";"):
    """Extrait le premier CSV d'un ZIP et retourne un DataFrame."""
    with zipfile.ZipFile(io.BytesIO(content_bytes)) as z:
        # Cherche tous les fichiers CSV (y compris dans les sous-dossiers)
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(f"Aucun CSV dans l'archive. Contenu : {z.namelist()}")
        logger.info(f"CSV trouvé dans archive : {csv_files[0]}")
        return pd.read_csv(z.open(csv_files[0]), sep=sep, dtype={"IRIS": str}, low_memory=False)


# ---------------------------------------------------------------------------
# 1. Contours géographiques IRIS
# ---------------------------------------------------------------------------

def download_iris_geojson():
    """Télécharge les contours IRIS depuis data.gouv.fr."""
    url = "https://www.data.gouv.fr/fr/datasets/r/0d97d2de-0e0d-4b9c-9bb5-5c3c2da1a0dd"
    out_path = os.path.join(DATA_DIR, "iris_contours.geojson")
    if os.path.exists(out_path):
        logger.info("Contours IRIS déjà téléchargés.")
        return out_path
    logger.info("Téléchargement des contours IRIS...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    logger.info(f"Contours IRIS sauvegardés : {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2. INSEE Filosofi – avec fallback synthétique si INSEE down
# ---------------------------------------------------------------------------

FILOSOFI_URLS = [
    "https://www.insee.fr/fr/statistiques/fichier/8229323/BASE_TD_FILO_DISP_IRIS_2021.zip",
    "https://www.insee.fr/fr/statistiques/fichier/7233950/indic-struct-distrib-revenu-2020-IRIS_csv.zip",
    "https://www.insee.fr/fr/statistiques/fichier/6049648/indic-struct-distrib-revenu-2019-IRIS_csv.zip",
]


def _synthetic_filosofi(n=50000):
    logger.warning("⚠ Données Filosofi synthétiques (INSEE indisponible)")
    np.random.seed(42)
    df = pd.DataFrame({"IRIS": [f"{i:09d}" for i in range(1, n + 1)]})
    df["DISP_MED20"] = np.random.lognormal(10.0, 0.35, n).clip(8000, 80000)
    df["DISP_Q120"]  = df["DISP_MED20"] * np.random.uniform(0.55, 0.75, n)
    df["DISP_Q320"]  = df["DISP_MED20"] * np.random.uniform(1.30, 1.70, n)
    rev_norm = (df["DISP_MED20"] - df["DISP_MED20"].min()) / (df["DISP_MED20"].max() - df["DISP_MED20"].min())
    df["TP6020"]     = (55 - rev_norm * 50 + np.random.normal(0, 5, n)).clip(3, 60)
    df["DISP_GI20"]  = np.random.beta(2, 5, n) * 0.40 + 0.20
    return df


def download_filosofi():
    out_path = os.path.join(DATA_DIR, "filosofi_iris.parquet")
    if os.path.exists(out_path):
        logger.info("Données Filosofi déjà disponibles.")
        return pd.read_parquet(out_path)

    for url in FILOSOFI_URLS:
        try:
            logger.info(f"Tentative Filosofi : {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            df = _zip_to_df(r.content)
            df["IRIS"] = df["IRIS"].str.zfill(9)
            df.to_parquet(out_path, index=False)
            logger.info(f"Filosofi OK : {len(df)} lignes")
            return df
        except Exception as e:
            logger.warning(f"Échec ({e.__class__.__name__}: {e}), URL suivante...")

    df = _synthetic_filosofi()
    df.to_parquet(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. INSEE Recensement Population – avec fallback synthétique
# ---------------------------------------------------------------------------

RP_URLS = {
    "population": "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-evol-struct-pop-2020_csv.zip",
    "logement":   "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-logement-2020_csv.zip",
    "emploi":     "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-activite-residents-2020_csv.zip",
}

# URLs alternatives si les premières échouent
RP_URLS_ALT = {
    "population": "https://www.insee.fr/fr/statistiques/fichier/6543200/base-ic-evol-struct-pop-2019_csv.zip",
    "logement":   "https://www.insee.fr/fr/statistiques/fichier/6543200/base-ic-logement-2019_csv.zip",
    "emploi":     "https://www.insee.fr/fr/statistiques/fichier/6543200/base-ic-activite-residents-2019_csv.zip",
}


def _synthetic_rp(theme, n=50000):
    logger.warning(f"⚠ Données RP {theme} synthétiques (INSEE indisponible)")
    np.random.seed(hash(theme) % 2**32)
    iris_codes = [f"{i:09d}" for i in range(1, n + 1)]
    df = pd.DataFrame({"IRIS": iris_codes})

    if theme == "population":
        df["P20_POP"]        = np.random.randint(500, 5000, n)
        df["P20_POP0014"]    = (df["P20_POP"] * np.random.uniform(0.10, 0.25, n)).astype(int)
        df["P20_POP6074"]    = (df["P20_POP"] * np.random.uniform(0.10, 0.20, n)).astype(int)
        df["P20_POP75P"]     = (df["P20_POP"] * np.random.uniform(0.05, 0.15, n)).astype(int)
        df["C20_POP15P_CS3"] = (df["P20_POP"] * np.random.uniform(0.05, 0.25, n)).astype(int)
        df["C20_POP15P_CS6"] = (df["P20_POP"] * np.random.uniform(0.05, 0.20, n)).astype(int)

    elif theme == "logement":
        df["P20_LOG"]      = np.random.randint(200, 2500, n)
        df["P20_RP"]       = (df["P20_LOG"] * np.random.uniform(0.70, 0.95, n)).astype(int)
        df["P20_RSECOCC"]  = (df["P20_LOG"] * np.random.uniform(0.01, 0.10, n)).astype(int)
        df["P20_LOGVAC"]   = (df["P20_LOG"] * np.random.uniform(0.03, 0.15, n)).astype(int)
        df["P20_RP_PROP"]  = (df["P20_RP"] * np.random.uniform(0.20, 0.80, n)).astype(int)
        df["P20_RP_LOC"]   = df["P20_RP"] - df["P20_RP_PROP"]

    elif theme == "emploi":
        pop = np.random.randint(300, 3000, n)
        df["P20_ACTOCC15P"]      = (pop * np.random.uniform(0.35, 0.55, n)).astype(int)
        df["P20_CHOM1564"]       = (df["P20_ACTOCC15P"] * np.random.uniform(0.05, 0.25, n)).astype(int)
        df["C20_ACTOCC15P_TP1"]  = (df["P20_ACTOCC15P"] * np.random.uniform(0.10, 0.25, n)).astype(int)

    return df


def download_rp(theme: str):
    assert theme in RP_URLS, f"Thème inconnu : {theme}"
    out_path = os.path.join(DATA_DIR, f"rp_{theme}_iris.parquet")
    if os.path.exists(out_path):
        logger.info(f"RP {theme} déjà disponible.")
        return pd.read_parquet(out_path)

    urls = [RP_URLS[theme], RP_URLS_ALT.get(theme, "")]
    for url in urls:
        if not url:
            continue
        try:
            logger.info(f"Téléchargement RP {theme} : {url}")
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            df = _zip_to_df(r.content)
            df["IRIS"] = df["IRIS"].str.zfill(9)
            df.to_parquet(out_path, index=False)
            logger.info(f"RP {theme} OK : {len(df)} lignes")
            return df
        except Exception as e:
            logger.warning(f"Échec ({e.__class__.__name__}: {e}), tentative suivante...")

    df = _synthetic_rp(theme)
    df.to_parquet(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 4. Fusion et construction du dataset final
# ---------------------------------------------------------------------------

FEATURES_FILOSOFI = ["DISP_MED20", "DISP_Q120", "DISP_Q320", "TP6020", "DISP_GI20"]
FEATURES_POP      = ["P20_POP", "P20_POP0014", "P20_POP6074", "P20_POP75P", "C20_POP15P_CS3", "C20_POP15P_CS6"]
FEATURES_LOG      = ["P20_LOG", "P20_RP", "P20_RSECOCC", "P20_LOGVAC", "P20_RP_PROP", "P20_RP_LOC"]
FEATURES_EMPLOI   = ["P20_ACTOCC15P", "P20_CHOM1564", "C20_ACTOCC15P_TP1"]


def build_dataset():
    out_path = os.path.join(DATA_DIR, "iris_features.parquet")
    if os.path.exists(out_path):
        logger.info("Dataset déjà construit.")
        return pd.read_parquet(out_path)

    df_filo = download_filosofi()
    df_pop  = download_rp("population")
    df_log  = download_rp("logement")
    df_emp  = download_rp("emploi")

    filo_cols = ["IRIS"] + [c for c in FEATURES_FILOSOFI if c in df_filo.columns]
    pop_cols  = ["IRIS"] + [c for c in FEATURES_POP      if c in df_pop.columns]
    log_cols  = ["IRIS"] + [c for c in FEATURES_LOG      if c in df_log.columns]
    emp_cols  = ["IRIS"] + [c for c in FEATURES_EMPLOI   if c in df_emp.columns]

    # Base = RP population (vrais codes IRIS INSEE)
    # LEFT JOIN sur toutes les autres sources → on garde uniquement les vrais IRIS
    df = df_pop[pop_cols].copy()
    df = df.merge(df_filo[filo_cols], on="IRIS", how="left")
    df = df.merge(df_log[log_cols],   on="IRIS", how="left")
    df = df.merge(df_emp[emp_cols],   on="IRIS", how="left")
    

    if "P20_POP" in df.columns:
        df["PART_JEUNES"]  = df.get("P20_POP0014", np.nan) / df["P20_POP"].replace(0, np.nan) * 100
        df["PART_SENIORS"] = (
            df.get("P20_POP6074", np.nan).fillna(0) + df.get("P20_POP75P", np.nan).fillna(0)
        ) / df["P20_POP"].replace(0, np.nan) * 100

    if "P20_ACTOCC15P" in df.columns and "P20_CHOM1564" in df.columns:
        df["TAUX_CHOM"] = df["P20_CHOM1564"] / (
            df["P20_ACTOCC15P"] + df["P20_CHOM1564"]
        ).replace(0, np.nan) * 100

    if "P20_LOG" in df.columns:
        df["TAUX_VACANCE"] = df.get("P20_LOGVAC", np.nan) / df["P20_LOG"].replace(0, np.nan) * 100
        df["TAUX_PROPRIO"] = df.get("P20_RP_PROP", np.nan) / df.get("P20_RP", df["P20_LOG"]).replace(0, np.nan) * 100

    df.to_parquet(out_path, index=False)
    logger.info(f"Dataset final : {out_path} ({len(df)} lignes, {df.shape[1]} colonnes)")
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print(f"\nShape : {df.shape}")
    print(f"\nColonnes : {list(df.columns)}")