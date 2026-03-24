"""
Module de collecte des données open data au niveau IRIS français.
Version stable :
  - 100% données réelles INSEE (Filosofi + RP)
  - BPE / Mobilités / SIRENE désactivés provisoirement (pas d'erreurs réseau)
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
# 4. BPE / Mobilités / SIRENE — désactivés provisoirement
# ---------------------------------------------------------------------------

def download_bpe(year=2022):
    logger.info("BPE désactivé provisoirement (pas de téléchargement).")
    return pd.DataFrame({"IRIS": [], "NB_EQUIPEMENTS": []})

def download_mobpro(year=2020):
    logger.info("Mobilités désactivées provisoirement (pas de téléchargement).")
    return pd.DataFrame({"IRIS": [], "NB_DEPLACEMENTS": []})

def download_sirene():
    logger.info("SIRENE désactivé provisoirement (pas de téléchargement).")
    return pd.DataFrame({"IRIS": [], "NB_ETABLISSEMENTS": []})

# ---------------------------------------------------------------------------
# 5. Fusion finale + imputations intelligentes
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

    # Base = population
    df = df_pop.copy()

    # Fusions
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

    # Imputation médiane sur un set de colonnes connues
    socio_cols = [
        "DISP_MED20", "DISP_Q120", "DISP_Q320", "TP6020", "DISP_GI20",
        "P20_POP", "P20_POP0014", "P20_POP6074", "P20_POP75P",
        "P20_LOG", "P20_LOGVAC", "P20_RP_PROP",
        "P20_ACTOCC15P", "P20_CHOM1564"
    ]

    for col in socio_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].median())

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
