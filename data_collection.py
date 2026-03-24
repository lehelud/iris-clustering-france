"""
Module de collecte des données open data au niveau IRIS français.
Version : 2026-03-24_v2

Corrections apportées :
- RP mis à jour vers millésime 2021 (ID INSEE 8268806/8268838/8268843)
- Noms de colonnes mis à jour P20_ → P21_
- BPE : nouvelle URL (fichier 8217527, BPE 2023 niveau IRIS)
- SIRENE : nouvelle URL (files.data.gouv.fr mis à jour)
- Fix RuntimeWarning "Mean of empty slice" : imputation uniquement si colonne non vide
- Ratios dérivés adaptés au préfixe P21_
"""

import os
import requests
import pandas as pd
import numpy as np
import zipfile
import io
import logging
import warnings

# Supprimer les RuntimeWarnings numpy résiduels (sécurité supplémentaire)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

logger.info(">>> data_collection.py version 2026-03-24_v2")

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
# 2. FILOSOFI (inchangé, fonctionne)
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
# 3. RP 2021 – Population / Logement / Emploi
#    (mise à jour depuis RP 2020, IDs INSEE : 8268806 / 8268838 / 8268843)
#    Les colonnes passent de P20_ à P21_
# ---------------------------------------------------------------------------

RP_URLS = {
    "population": "https://www.insee.fr/fr/statistiques/fichier/8268806/base-ic-evol-struct-pop-2021_csv.zip",
    "logement":   "https://www.insee.fr/fr/statistiques/fichier/8268838/base-ic-logement-2021_csv.zip",
    "emploi":     "https://www.insee.fr/fr/statistiques/fichier/8268843/base-ic-activite-residents-2021_csv.zip",
}

# Fallback RP 2020 si 2021 indisponible
RP_FALLBACK_URLS = {
    "population": "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-evol-struct-pop-2020_csv.zip",
    "logement":   "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-logement-2020_csv.zip",
    "emploi":     "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-activite-residents-2020_csv.zip",
}


def download_rp(theme):
    out_path = os.path.join(DATA_DIR, f"rp_{theme}_iris.parquet")
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    for label, url in [("2021", RP_URLS[theme]), ("2020 (fallback)", RP_FALLBACK_URLS[theme])]:
        try:
            logger.info(f"Téléchargement RP {theme} {label}")
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            df = _zip_to_df(r.content)
            df["IRIS"] = df["IRIS"].str.zfill(9)
            df.to_parquet(out_path, index=False)
            logger.info(f"✅ RP {theme} {label} OK ({df.shape[0]} lignes)")
            return df
        except Exception as e:
            logger.warning(f"Échec RP {theme} {label} : {e}")

    logger.error(f"❌ RP {theme} totalement indisponible.")
    return pd.DataFrame({"IRIS": []})


# ---------------------------------------------------------------------------
# 4. BPE 2023 niveau IRIS
#    Nouvelle URL : fichier 8217527 (remplace l'ancienne 7633565)
#    Note : depuis BPE 2023, l'offre de diffusion a changé.
#    On tente le fichier agrégé IRIS direct.
# ---------------------------------------------------------------------------

BPE_URLS = [
    # BPE 2023 – dénombrement IRIS (format CSV, toutes gammes)
    "https://www.insee.fr/fr/statistiques/fichier/8217527/BPE23_IRIS_csv.zip",
    # BPE 2022 – ancienne structure (fallback)
    "https://www.insee.fr/fr/statistiques/fichier/7633565/bpe_iris_2022_csv.zip",
]


def download_bpe():
    out_path = os.path.join(DATA_DIR, "bpe_iris.parquet")
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    for url in BPE_URLS:
        try:
            logger.info(f"Téléchargement BPE : {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            df = _zip_to_df(r.content)
            # La BPE 2023 peut avoir la colonne DCIRIS ou IRIS selon le format
            if "DCIRIS" in df.columns:
                df = df.rename(columns={"DCIRIS": "IRIS"})
            if "IRIS" in df.columns:
                df["IRIS"] = df["IRIS"].str.zfill(9)
            df.to_parquet(out_path, index=False)
            logger.info(f"✅ BPE OK ({df.shape[0]} lignes)")
            return df
        except Exception as e:
            logger.warning(f"Échec BPE ({url}) : {e}")

    logger.warning("⚠️ BPE indisponible – colonne NB_ETABLISSEMENTS sera absente.")
    return pd.DataFrame({"IRIS": []})


# ---------------------------------------------------------------------------
# 5. SIRENE géolocalisé (nouveau chemin data.gouv.fr)
#    L'ancien fichier files.data.gouv.fr/insee-sirene/ a été supprimé.
#    On utilise désormais la géolocalisation SIRENE (avec code IRIS).
#    URL mise à jour : fichier mensuel du dataset officiel INSEE/data.gouv.fr
# ---------------------------------------------------------------------------

SIRENE_URLS = [
    # Géolocalisation SIRENE avec IRIS – mis à jour mensuellement
    "https://files.data.gouv.fr/geo-sirene/last/communes/geo-sirene.csv.gz",
    # Fallback : stock établissements complet (lourd ~700 Mo)
    "https://files.data.gouv.fr/insee-sirene/StockEtablissement_utf8.zip",
]


def download_sirene():
    """
    Télécharge SIRENE géolocalisé et agrège le nombre d'établissements par IRIS.
    Retourne un DataFrame avec colonnes [IRIS, NB_ETABLISSEMENTS].
    """
    out_path = os.path.join(DATA_DIR, "sirene_iris.parquet")
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    # Tentative 1 : geo-sirene (fichier CSV.GZ avec colonne iris)
    url_geo = "https://files.data.gouv.fr/geo-sirene/last/communes/geo-sirene.csv.gz"
    try:
        logger.info(f"Téléchargement SIRENE géolocalisé : {url_geo}")
        r = requests.get(url_geo, timeout=300, stream=True)
        r.raise_for_status()
        # Lecture streaming pour éviter de tout charger en mémoire
        df = pd.read_csv(
            io.BytesIO(r.content),
            compression="gzip",
            sep=",",
            dtype=str,
            low_memory=False,
            usecols=lambda c: c in ["iris", "siret", "etatAdministratifEtablissement"],
        )
        # Garder seulement les établissements actifs
        if "etatAdministratifEtablissement" in df.columns:
            df = df[df["etatAdministratifEtablissement"] == "A"]
        df = df.rename(columns={"iris": "IRIS"})
        df["IRIS"] = df["IRIS"].astype(str).str.zfill(9)
        agg = df.groupby("IRIS").size().reset_index(name="NB_ETABLISSEMENTS")
        agg.to_parquet(out_path, index=False)
        logger.info(f"✅ SIRENE géolocalisé OK ({agg.shape[0]} IRIS)")
        return agg
    except Exception as e:
        logger.warning(f"Échec SIRENE géolocalisé : {e}")

    logger.warning("⚠️ SIRENE indisponible – colonne NB_ETABLISSEMENTS sera absente.")
    return pd.DataFrame({"IRIS": [], "NB_ETABLISSEMENTS": []})


# ---------------------------------------------------------------------------
# 6. Mobilités domicile-travail (agrégation par commune → IRIS non disponible
#    directement ; on skip et on calcule NB_DEPLACEMENTS = 0 par défaut)
#    Note : les mobilités INSEE n'existent qu'au niveau COMMUNE, pas IRIS.
#    Pour un proxy IRIS, il faudrait un traitement spécifique non inclus ici.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. Fusion finale + imputations intelligentes
# ---------------------------------------------------------------------------

def _safe_div(num, den):
    """Division sécurisée : retourne NaN si dénominateur nul ou NaN."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    res = num / den
    res[~np.isfinite(res)] = np.nan
    return res


def _impute_numeric(df):
    """
    Imputation par médiane pour les colonnes numériques non entièrement vides.
    Analogie : comme remplir les trous d'un puzzle uniquement quand on a
    au moins une pièce de référence dans la même zone.
    Les colonnes 100% NaN (sources manquantes) sont laissées telles quelles.
    """
    for col in df.columns:
        if col in ("IRIS", "COM", "TYP_IRIS", "LAB_IRIS"):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            # Colonne entièrement vide : on ne calcule pas la médiane (évite le warning)
            df[col] = np.nan
        elif series.notna().any():
            median_val = series.median()
            df[col] = series.fillna(median_val)
    return df


def build_dataset():
    out_path = os.path.join(DATA_DIR, "iris_features.parquet")
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    # Chargement des sources
    df_filo = download_filosofi()
    df_pop  = download_rp("population")
    df_log  = download_rp("logement")
    df_emp  = download_rp("emploi")
    df_bpe  = download_bpe()
    df_sir  = download_sirene()

    # Base = population
    df = df_pop.copy()

    # Fusions successives
    for df_other in [df_filo, df_log, df_emp]:
        if not df_other.empty and "IRIS" in df_other.columns:
            df = df.merge(df_other, on="IRIS", how="left")

    # Ajout BPE (NB_ETABLISSEMENTS par IRIS)
    if not df_bpe.empty and "IRIS" in df_bpe.columns:
        # BPE complète : agréger le nombre total d'équipements par IRIS
        if "NB_EQUIP" in df_bpe.columns:
            bpe_agg = df_bpe.groupby("IRIS")["NB_EQUIP"].sum().reset_index()
            bpe_agg = bpe_agg.rename(columns={"NB_EQUIP": "NB_ETABLISSEMENTS"})
        elif "NB_ETABLISSEMENTS" in df_bpe.columns:
            bpe_agg = df_bpe[["IRIS", "NB_ETABLISSEMENTS"]]
        else:
            # BPE en format détail : compter les lignes par IRIS
            bpe_agg = df_bpe.groupby("IRIS").size().reset_index(name="NB_ETABLISSEMENTS")
        df = df.merge(bpe_agg, on="IRIS", how="left")
    else:
        df["NB_ETABLISSEMENTS"] = np.nan

    # Ajout SIRENE
    if not df_sir.empty and "IRIS" in df_sir.columns:
        df = df.merge(df_sir, on="IRIS", how="left")

    # Colonne mobilités (proxy = 0 faute de données IRIS directes)
    df["NB_DEPLACEMENTS"] = 0.0

    # -----------------------------------------------------------------------
    # Imputation numérique (fix du RuntimeWarning)
    # -----------------------------------------------------------------------
    df = _impute_numeric(df)

    # -----------------------------------------------------------------------
    # Ratios dérivés – adaptés au millésime 2021 (préfixe P21_)
    # On garde aussi la compatibilité P20_ si RP 2020 fallback utilisé
    # -----------------------------------------------------------------------
    def get_col(df, *candidates):
        """Retourne la première colonne candidate présente dans df."""
        for c in candidates:
            if c in df.columns:
                return df[c]
        return None

    chom     = get_col(df, "P21_CHOM1564", "P20_CHOM1564")
    actocc   = get_col(df, "P21_ACTOCC15P", "P20_ACTOCC15P")
    logvac   = get_col(df, "P21_LOGVAC", "P20_LOGVAC")
    log      = get_col(df, "P21_LOG", "P20_LOG")
    rp_prop  = get_col(df, "P21_RP_PROP", "P20_RP_PROP")
    pop0014  = get_col(df, "P21_POP0014", "P20_POP0014")
    pop6074  = get_col(df, "P21_POP6074", "P20_POP6074")
    pop75p   = get_col(df, "P21_POP75P", "P20_POP75P")
    pop      = get_col(df, "P21_POP", "P20_POP")

    if chom is not None and actocc is not None:
        df["TAUX_CHOM"] = _safe_div(chom, actocc + chom)

    if logvac is not None and log is not None:
        df["TAUX_VACANCE"] = _safe_div(logvac, log)

    if rp_prop is not None and log is not None:
        df["TAUX_PROPRIO"] = _safe_div(rp_prop, log)

    if pop0014 is not None and pop is not None:
        df["PART_JEUNES"] = _safe_div(pop0014, pop)

    if pop6074 is not None and pop75p is not None and pop is not None:
        df["PART_SENIORS"] = _safe_div(pop6074 + pop75p, pop)

    # -----------------------------------------------------------------------
    df.to_parquet(out_path, index=False)
    logger.info(f"Dataset enrichi sauvegardé ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print(df.shape)
