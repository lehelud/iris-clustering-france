"""
Module de collecte des données open data au niveau IRIS français.
Sources :
  - INSEE : Filosofi (revenus), RP (recensement population, logement, emploi)
    - data.gouv.fr : données géographiques IRIS
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
# 1. Contours géographiques IRIS (GeoJSON)
# ---------------------------------------------------------------------------

def download_iris_geojson():
      """Télécharge les contours IRIS depuis data.gouv.fr."""
      url = (
          "https://www.data.gouv.fr/fr/datasets/r/"
          "0d97d2de-0e0d-4b9c-9bb5-5c3c2da1a0dd"
      )
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
# 2. INSEE Filosofi – revenus et pauvreté au niveau IRIS
# ---------------------------------------------------------------------------

FILOSOFI_URL = (
      "https://www.insee.fr/fr/statistiques/fichier/7233950/"
      "indic-struct-distrib-revenu-2020-IRIS_csv.zip"
)


def download_filosofi():
      """Télécharge et parse les données Filosofi INSEE (revenus IRIS 2020)."""
    out_path = os.path.join(DATA_DIR, "filosofi_iris.parquet")
    if os.path.exists(out_path):
              logger.info("Données Filosofi déjà disponibles.")
              return pd.read_parquet(out_path)

    logger.info("Téléchargement Filosofi INSEE...")
    r = requests.get(FILOSOFI_URL, timeout=180)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
              csv_files = [n for n in z.namelist() if n.endswith(".csv")]
              logger.info(f"Fichiers CSV dans l'archive : {csv_files}")
              # On prend le premier CSV de structure
              df = pd.read_csv(z.open(csv_files[0]), sep=";", dtype={"IRIS": str}, low_memory=False)

    # Harmonisation du code IRIS sur 9 caractères
    df["IRIS"] = df["IRIS"].str.zfill(9)
    df.to_parquet(out_path, index=False)
    logger.info(f"Filosofi sauvegardé : {out_path} ({len(df)} lignes)")
    return df


# ---------------------------------------------------------------------------
# 3. INSEE Recensement Population – caractéristiques IRIS
# ---------------------------------------------------------------------------

RP_URLS = {
      "population": (
                "https://www.insee.fr/fr/statistiques/fichier/7704076/"
                "base-ic-evol-struct-pop-2020_csv.zip"
      ),
      "logement": (
                "https://www.insee.fr/fr/statistiques/fichier/7704076/"
                "base-ic-logement-2020_csv.zip"
      ),
      "emploi": (
                "https://www.insee.fr/fr/statistiques/fichier/7704076/"
                "base-ic-activite-residents-2020_csv.zip"
      ),
}


def download_rp(theme: str):
      """Télécharge une table du Recensement de la Population INSEE."""
    assert theme in RP_URLS, f"Thème inconnu : {theme}"
    out_path = os.path.join(DATA_DIR, f"rp_{theme}_iris.parquet")
    if os.path.exists(out_path):
              logger.info(f"RP {theme} déjà disponible.")
              return pd.read_parquet(out_path)

    url = RP_URLS[theme]
    logger.info(f"Téléchargement RP {theme}...")
    r = requests.get(url, timeout=180)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
              csv_files = [n for n in z.namelist() if n.endswith(".csv")]
              df = pd.read_csv(z.open(csv_files[0]), sep=";", dtype={"IRIS": str}, low_memory=False)

    df["IRIS"] = df["IRIS"].str.zfill(9)
    df.to_parquet(out_path, index=False)
    logger.info(f"RP {theme} sauvegardé : {out_path} ({len(df)} lignes)")
    return df


# ---------------------------------------------------------------------------
# 4. Fusion et construction du dataset final
# ---------------------------------------------------------------------------

FEATURES_FILOSOFI = [
      "DISP_MED20",   # Revenu médian disponible
      "DISP_Q120",    # 1er quartile revenu
      "DISP_Q320",    # 3e quartile revenu
      "TP6020",       # Taux de pauvreté
      "DISP_GI20",    # Indice de Gini
]

FEATURES_POP = [
      "P20_POP",      # Population totale
      "P20_POP0014",  # Population 0-14 ans
      "P20_POP6074",  # Population 60-74 ans
      "P20_POP75P",   # Population 75 ans et +
      "C20_POP15P_CS3",  # Cadres et professions intellectuelles
      "C20_POP15P_CS6",  # Ouvriers
]

FEATURES_LOG = [
      "P20_LOG",      # Logements totaux
      "P20_RP",       # Résidences principales
      "P20_RSECOCC",  # Résidences secondaires
      "P20_LOGVAC",   # Logements vacants
      "P20_RP_PROP",  # Propriétaires
      "P20_RP_LOC",   # Locataires
]

FEATURES_EMPLOI = [
      "P20_ACTOCC15P",   # Actifs occupés
      "P20_CHOM1564",    # Chômeurs
      "C20_ACTOCC15P_TP1",  # Temps partiel
]


def build_dataset():
      """Construit le dataset consolidé de features IRIS."""
      out_path = os.path.join(DATA_DIR, "iris_features.parquet")
      if os.path.exists(out_path):
                logger.info("Dataset déjà construit.")
                return pd.read_parquet(out_path)

      df_filo = download_filosofi()
      df_pop = download_rp("population")
      df_log = download_rp("logement")
      df_emp = download_rp("emploi")

    # Sélection des features disponibles
      filo_cols = ["IRIS"] + [c for c in FEATURES_FILOSOFI if c in df_filo.columns]
    pop_cols = ["IRIS"] + [c for c in FEATURES_POP if c in df_pop.columns]
    log_cols = ["IRIS"] + [c for c in FEATURES_LOG if c in df_log.columns]
    emp_cols = ["IRIS"] + [c for c in FEATURES_EMPLOI if c in df_emp.columns]

    df = (
              df_filo[filo_cols]
              .merge(df_pop[pop_cols], on="IRIS", how="outer")
              .merge(df_log[log_cols], on="IRIS", how="outer")
              .merge(df_emp[emp_cols], on="IRIS", how="outer")
    )

    # Calcul de taux/ratios dérivés
    if "P20_POP" in df.columns and "P20_POP" in df.columns:
              df["PART_JEUNES"] = df.get("P20_POP0014", np.nan) / df["P20_POP"].replace(0, np.nan) * 100
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
  
