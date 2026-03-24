"""
Module de collecte des données open data au niveau IRIS français.
Version stable :
  - 100% données réelles
  - URLs corrigées
  - Gestion propre des sources indisponibles
  - Imputations intelligentes
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
# Utilitaire ZIP → DataFrame
# ---------------------------------------------------------------------------

def _zip_to_df(content_bytes, sep=";"):
    with zipfile.ZipFile(io.BytesIO(content_bytes)) as z:
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("Aucun CSV trouvé dans l'archive ZIP.")
        return pd.read_csv(z.open(csv_files[0]), sep=sep, dtype=str, low_memory=False)

# ---------------------------------------------------------------------------
# 1. Contours IRIS
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
# 2. FILOSOFI (URLs stables)
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
# 4. BPE (URL corrigée)
# ---------------------------------------------------------------------------

def download_bpe(year=2023):
    url = f"https://www.insee.fr/fr/statistiques/fichier/7897985/bpe_{year}.zip"
    out_path = os.path.join(DATA_DIR, f"bpe_{year}.parquet")

    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    try:
        logger.info("Téléchargement BPE…")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        df = _zip_to_df(r.content)
        df["IRIS"] = df["IRIS"].str.zfill(9)
        df["NB_EQUIPEMENTS"] = 1
        df = df.groupby("IRIS")["NB_EQUIPEMENTS"].sum().reset_index()
        df.to_parquet(out_path, index=False)
        return df
    except Exception as e:
        logger.error(f"❌ BPE indisponible : {e}")
        return pd.DataFrame({"IRIS": [], "NB_EQUIPEMENTS": []})

# ---------------------------------------------------------------------------
# 5. Mobilités domicile-travail (URL corrigée)
# ---------------------------------------------------------------------------

def download_mobpro(year=2019):
    url = f"https://www.insee.fr/fr/statistiques/fichier/5395746/mobpro_{year}.csv"
    out_path = os.path.join(DATA_DIR, f"mobpro_{year}.parquet")

    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    try:
        logger.info("Téléchargement Mobilités…")
        df = pd.read_csv(url, sep=";", dtype=str)
        df["NB_DEPLACEMENTS"] = df["IPONDI"].astype(float)
        df = df.groupby("IRIS_RES")["NB_DEPLACEMENTS"].sum().reset_index()
        df.rename(columns={"IRIS_RES": "IRIS"}, inplace=True)
        df["IRIS"] = df["IRIS"].str.zfill(9)
        df.to_parquet(out_path, index=False)
        return df
    except Exception as e:
        logger.error(f"❌ Mobilités indisponibles : {e}")
        return pd.DataFrame({"IRIS": [], "NB_DEPLACEMENTS": []})

# ---------------------------------------------------------------------------
# 6. SIRENE (URL corrigée)
# ---------------------------------------------------------------------------

def download_sirene():
    url = "https://files.data.gouv.fr/insee-sirene/StockEtablissement_utf8.zip"
    out_path = os.path.join(DATA_DIR, "sirene.parquet")

    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    try:
        logger.info("Téléchargement SIRENE…")
        r = requests.get(url, timeout=240)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            df = pd.read_csv(z.open("StockEtablissement_utf8.csv"), dtype=str)
        df["IRIS"] = df["IRIS"].str.zfill(9)
        df["NB_ETABLISSEMENTS"] = 1
        df = df.groupby("IRIS")["NB_ETABLISSEMENTS"].sum().reset_index()
        df.to_parquet(out_path, index=False)
        return df
    except Exception as e:
        logger.error(f"❌ SIRENE indisponible : {e}")
        return pd.DataFrame({"IRIS": [], "NB_ETABLISSEMENTS": []})

# ---------------------------------------------------------------------------
# 7. Fusion finale + imputations intelligentes
# ---------------------------------------------------------------------------

def build_dataset():
    out_path = os.path.join(DATA_DIR, "iris_features.parquet")
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    df_filo = download_filosofi()
    df_pop  = download_rp("population")
    df_log  = download_rp("logement")
    df_emp  = download_rp("emploi")
    df_bpe  = download_bpe()
    df_mob  = download_mobpro()
    df_sir  = download_sirene()

    df = df_pop.copy()

    df = df.merge(df_filo, on="IRIS", how="left")
    df = df.merge(df_log,  on="IRIS", how="left")
    df = df.merge(df_emp,  on="IRIS", how="left")
    df = df.merge(df_bpe,  on="IRIS", how="left")
    df = df.merge(df_mob,  on="IRIS", how="left")
    df = df.merge(df_sir,  on="IRIS", how="left")

    # Comptages → 0
    for col in ["NB_EQUIPEMENTS", "NB_DEPLACEMENTS", "NB_ETABLISSEMENTS"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Imputation médiane
    socio_cols = [
        c for c in df.columns
        if c not in ["IRIS", "COM", "TYP_IRIS", "LAB_IRIS"]
        and df[c].dtype != object
    ]

    for col in socio_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Ratios dérivés
    df["TAUX_CHOM"] = df["P20_CHOM1564"] / (df["P20_ACTOCC15P"] + df["P20_CHOM1564"])
    df["TAUX_VACANCE"] = df["P20_LOGVAC"] / df["P20_LOG"]
    df["TAUX_PROPRIO"] = df["P20_RP_PROP"] / df["P20_LOG"]
    df["PART_JEUNES"] = df["P20_POP0014"] / df["P20_POP"]
    df["PART_SENIORS"] = (df["P20_POP6074"] + df["P20_POP75P"]) / df["P20_POP"]

    df.to_parquet(out_path, index=False)
    logger.info(f"Dataset enrichi sauvegardé ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    return df

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print(df.shape)
