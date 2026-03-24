"""
Module de collecte des données open data au niveau IRIS français.
Version : 2026-03-24_simplifiee
  - 100% données réelles INSEE (Filosofi + RP)
  - PAS de BPE / Mobilités / SIRENE (désactivés)
  - Imputations intelligentes sans warnings
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

logger.info(">>> data_collection.py version 2026-03-24_simplifiee")

# ---------------------------------------------------------------------------
# Utilitaire ZIP → DataFrame
# ---------------------------------------------------------------------------

def _zip_to_df(content_bytes, sep=";"):
    with zipfile.ZipFile(io.BytesIO(content_bytes)) as z:
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("Aucun CSV trouvé dans l'archive ZIP.")
        return pd.read_csv(z.open(csv_files[0]), sep=sep, dtype=str, low_memory=False)

# ---------------------------------------------------------------------------
# 1. Contours IRIS (optionnel)
# ---------------------------------------------------------------------------

def download_iris_geojson():
    url = "https://www.data.gouv.fr/fr/datasets/r/0d97d2de-0e0d-4b9c-9bb5-5c3c2da1a0dd"
    out_path = os.path.join(DATA_DIR, "iris_contours.geojson")
    if os.path.exists(out_path):
        return out_path
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

# ---------------------------------------------------------------------------
# 2. FILOSOFI
# ---------------------------------------------------------------------------

FILOSOFI_URLS = [
    "https://www.insee.fr/fr/statistiques/fichier/8229323/BASE_TD_FILO_DISP_IRIS_2021.zip",
    "https://www.insee.fr/fr/statistiques/fichier/7233950/indic-struct-distrib-revenu-2020-IRIS_csv.zip",
]

def download_filosofi():
    out_path = os.path.join(DATA_DIR, "filosofi_iris.parquet")
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    for url in FILOSOFI_URLS:
        try:
            logger.info(f"Téléchargement Filosofi : {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            df = _zip_to_df(r.content)
            df["IRIS"] = df["IRIS"].str.zfill(9)
            df.to_parquet(out_path, index=False)
            return df
        except Exception as e:
            logger.warning(f"Échec Filosofi : {e}")

    logger.error("❌ Aucune source Filosofi disponible.")
    return pd.DataFrame({"IRIS": []})

# ---------------------------------------------------------------------------
# 3. RP Population / Logement / Emploi
# ---------------------------------------------------------------------------

RP_URLS = {
    "population": "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-evol-struct-pop-2020_csv.zip",
    "logement":   "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-logement-2020_csv.zip",
    "emploi":     "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-activite-residents-2020_csv.zip",
}

def download_rp(theme):
    out_path = os.path.join(DATA_DIR, f"rp_{theme}_iris.parquet")
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    try:
        logger.info(f"Téléchargement RP {theme}")
        r = requests.get(RP_URLS[theme], timeout=180)
        r.raise_for_status()
        df = _zip_to_df(r.content)
        df["IRIS"] = df["IRIS"].str.zfill(9)
        df.to_parquet(out_path, index=False)
        return df
    except Exception as e:
        logger.error(f"❌ RP {theme} indisponible : {e}")
        return pd.DataFrame({"IRIS": []})

# ---------------------------------------------------------------------------
# 4. Fusion finale + imputations intelligentes
# ---------------------------------------------------------------------------

def build_dataset():
    out_path = os.path.join(DATA_DIR, "iris_features.parquet")
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    df_filo = download_filosofi()
    df_pop  = download_rp("population")
    df_log  = download_rp("logement")
    df_emp  = download_rp("emploi")

    # Base = population
    df = df_pop.copy()

    # Fusions
    df = df.merge(df_filo, on="IRIS", how="left")
    df = df.merge(df_log,  on="IRIS", how="left")
    df = df.merge(df_emp,  on="IRIS", how="left")

    # Conversion numérique + imputations
    numeric_cols = []
    for col in df.columns:
        if col == "IRIS":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().any():
            numeric_cols.append(col)

    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Ratios dérivés (avec gestion des divisions par zéro)
    def safe_div(num, den):
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        res = num / den
        res[~np.isfinite(res)] = np.nan
        return res

    if {"P20_CHOM1564", "P20_ACTOCC15P"}.issubset(df.columns):
        df["TAUX_CHOM"] = safe_div(df["P20_CHOM1564"], df["P20_ACTOCC15P"] + df["P20_CHOM1564"])

    if {"P20_LOGVAC", "P20_LOG"}.issubset(df.columns):
        df["TAUX_VACANCE"] = safe_div(df["P20_LOGVAC"], df["P20_LOG"])

    if {"P20_RP_PROP", "P20_LOG"}.issubset(df.columns):
        df["TAUX_PROPRIO"] = safe_div(df["P20_RP_PROP"], df["P20_LOG"])

    if {"P20_POP0014", "P20_POP"}.issubset(df.columns):
        df["PART_JEUNES"] = safe_div(df["P20_POP0014"], df["P20_POP"])

    if {"P20_POP6074", "P20_POP75P", "P20_POP"}.issubset(df.columns):
        df["PART_SENIORS"] = safe_div(df["P20_POP6074"] + df["P20_POP75P"], df["P20_POP"])

    df.to_parquet(out_path, index=False)
    logger.info(f"Dataset enrichi sauvegardé ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    return df

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print(df.shape)
