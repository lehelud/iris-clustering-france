"""
Application Streamlit - Rapport de Clustering des IRIS français
Auteur : lehelud
Version enrichie (BPE + Mobilités + SIRENE)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

# ---- Configuration de la page ----
st.set_page_config(
    page_title="Clustering des IRIS français",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = "results"
DATA_DIR = "data"

# Palette couleurs distinctes pour les clusters
CLUSTER_COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly


@st.cache_data(show_spinner=False)
def load_cluster_names():
    path = os.path.join(RESULTS_DIR, "cluster_names.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Convertir les clés en int
    return {int(k): v for k, v in raw.items()}

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Chargement des données de clustering...")
def load_clustered_data():
    path = os.path.join(RESULTS_DIR, "iris_clustered.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

@st.cache_data(show_spinner="Chargement du profil des clusters...")
def load_profile():
    path = os.path.join(RESULTS_DIR, "cluster_profile.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)

@st.cache_data(show_spinner="Chargement des métriques de sélection...")
def load_metrics():
    path = os.path.join(RESULTS_DIR, "metrics_k_selection.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data(show_spinner="Chargement des features brutes...")
def load_features():
    path = os.path.join(DATA_DIR, "iris_features.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def load_population():
    """Retourne un dict IRIS -> population (P20_POP)."""
    path = os.path.join(DATA_DIR, "iris_features.parquet")
    if not os.path.exists(path):
        return {}
    df_feat = pd.read_parquet(path, columns=["IRIS", "P20_POP"])
    return dict(zip(df_feat["IRIS"].astype(str), df_feat["P20_POP"]))

@st.cache_data(show_spinner="Chargement du GeoJSON IRIS...")
def load_geojson():
    """Charge le GeoJSON IRIS une seule fois et le met en cache."""
    from data_collection import ensure_iris_geojson
    path = ensure_iris_geojson()
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_dep(iris_code: str) -> str:
    """Extrait le code département depuis un code IRIS (9 chars)."""
    com = iris_code[:5]
    return com[:3] if com.startswith("97") else com[:2]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📊 INSEE")
    st.title("🗺️ Clustering IRIS")
    st.markdown("---")

    st.markdown("### Sources de données enrichies")
    st.markdown("- INSEE Filosofi 2020 (revenus)")
    st.markdown("- INSEE RP 2020 (population, logement, emploi)")
    st.markdown("- **BPE (Base Permanente des Équipements)**")
    st.markdown("- **Mobilités domicile-travail**")
    st.markdown("- **SIRENE (établissements)**")
    st.markdown("---")
    st.markdown("*Millésime : 2020*")

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

df = load_clustered_data()
profile = load_profile()
metrics = load_metrics()
df_features = load_features()
cluster_names = load_cluster_names()

# Enrichissement population
if df is not None:
    pop_map = load_population()
    df["pop"] = df["IRIS"].astype(str).map(pop_map)
    df["dep"] = df["IRIS"].astype(str).apply(get_dep)

def cluster_label(cid):
    """Retourne le label complet d'un cluster : 'Cluster N — Nom'."""
    name = cluster_names.get(int(cid), "")
    return f"Cluster {cid} — {name}" if name else f"Cluster {cid}"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🗺️ Clustering enrichi des IRIS français")
st.markdown(
    """
    Cette application présente une **segmentation enrichie des IRIS français**, intégrant :
    - données socio-économiques INSEE,
    - **équipements (BPE)**,
    - **mobilités domicile-travail**,
    - **tissu économique (SIRENE)**.

    L’objectif est d’identifier des **profils territoriaux homogènes**.
    """
)

if df is None or profile is None:
    st.warning(
        "⚠️ Les résultats de clustering ne sont pas encore disponibles.\n"
        "Exécutez d'abord :\n\n"
        "```bash\npython clustering.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

n_clusters = df["cluster"].nunique()
n_iris = len(df)
n_features = len([c for c in df.columns if c not in {"IRIS", "cluster", "pca_x", "pca_y", "umap_x", "umap_y"}])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Nombre de clusters", n_clusters)
col2.metric("IRIS analysés", f"{n_iris:,}")
col3.metric("Variables utilisées", n_features)

if metrics is not None:
    best_row = metrics.loc[metrics["silhouette"].idxmax()]
    col4.metric("Silhouette max", f"{best_row['silhouette']:.3f}")

st.markdown("---")

# ===========================================================================
# ONGLETS
# ===========================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Vue d'ensemble",
    "🔍 Profil des clusters",
    "📉 Sélection du K",
    "🗺️ Carte",
    "📋 Données détaillées",
    "📚 Sources enrichies",
    "📖 Méthodologie",
])

# ===========================================================================
# ONGLET 1 — Vue d'ensemble
# ===========================================================================

with tab1:
    st.subheader("Répartition des IRIS par cluster")

    dist = (
        df.groupby("cluster")
        .agg(count=("IRIS", "count"), population=("pop", "sum"))
        .reset_index()
        .sort_values("cluster")
    )
    dist["population"] = dist["population"].fillna(0).astype(int)
    dist["pct_iris"] = (dist["count"] / dist["count"].sum() * 100).round(1)
    dist["pct_pop"] = (dist["population"] / dist["population"].sum() * 100).round(1)
    dist["label"] = dist["cluster"].apply(cluster_label)
    dist["label_court"] = dist["cluster"].apply(lambda c: f"C{c}")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        fig_pie = px.pie(
            dist, values="count", names="label",
            color="label", color_discrete_sequence=CLUSTER_COLORS,
            title="Répartition des IRIS",
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col_b:
        # Barres horizontales : labels lisibles sur l'axe Y
        fig_bar = px.bar(
            dist, y="label_court", x="count",
            color="label_court", color_discrete_sequence=CLUSTER_COLORS,
            text="pct_iris",
            orientation="h",
            title="Nombre d'IRIS par cluster",
            hover_data={"label": True, "count": True, "population": True, "label_court": False},
        )
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(r=60),
            xaxis=dict(range=[0, dist["count"].max() * 1.2]),
            yaxis_title="",
        )
        st.plotly_chart(fig_bar, width="stretch")

    # Tableau récapitulatif avec population
    st.subheader("Résumé par cluster")
    summary_df = dist[["cluster", "label", "count", "pct_iris", "population", "pct_pop"]].copy()
    summary_df.columns = ["Cluster", "Nom", "IRIS", "% IRIS", "Population", "% Population"]
    summary_df["Population"] = summary_df["Population"].apply(lambda x: f"{x:,}")
    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.subheader("Visualisation PCA / UMAP")

    sample = df.sample(min(15000, len(df)), random_state=42).copy()
    sample["label"] = sample["cluster"].apply(cluster_label)

    if "pca_x" in df.columns:
        fig_pca = px.scatter(
            sample, x="pca_x", y="pca_y",
            color="label", color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.5, title="Projection PCA (2D)",
        )
        st.plotly_chart(fig_pca, width="stretch")

    if "umap_x" in df.columns:
        fig_umap = px.scatter(
            sample, x="umap_x", y="umap_y",
            color="label", color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.5, title="Projection UMAP (2D)",
        )
        st.plotly_chart(fig_umap, width="stretch")

# ===========================================================================
# ONGLET 2 — Profil des clusters
# ===========================================================================

with tab2:
    st.subheader("Profil moyen par cluster")

    feat_cols = [c for c in profile.columns if c != "n_iris"]

    # Z-score par feature : montre l'écart à la moyenne inter-clusters
    profile_zscore = profile[feat_cols].copy()
    for col in feat_cols:
        mean, std = profile_zscore[col].mean(), profile_zscore[col].std()
        profile_zscore[col] = (profile_zscore[col] - mean) / std if std > 0 else 0.0

    fig_heat = px.imshow(
        profile_zscore.T,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Heatmap des profils (z-score — rouge = sous la moyenne, vert = au-dessus)",
        zmin=-2, zmax=2,
    )
    fig_heat.update_layout(coloraxis_colorbar_title="z-score")
    st.plotly_chart(fig_heat, width="stretch")

    cluster_options = sorted(df["cluster"].unique())
    cluster_display = {cid: cluster_label(cid) for cid in cluster_options}
    selected_cluster = st.selectbox(
        "Cluster à analyser",
        options=cluster_options,
        format_func=lambda cid: cluster_display[cid],
    )

    row = profile.loc[selected_cluster]
    name = cluster_names.get(int(selected_cluster), "")
    if name:
        st.success(f"**{cluster_label(selected_cluster)}** — {row['n_iris']:,} IRIS")
    else:
        st.info(f"Ce cluster regroupe **{row['n_iris']:,} IRIS**")

    # Radar chart — z-score : 0 = moyenne inter-clusters, +1/-1 = 1 écart-type
    feat_radar = feat_cols[:12]
    vals_zscore = []
    for f in feat_radar:
        mean, std = profile[f].mean(), profile[f].std()
        vals_zscore.append((row[f] - mean) / std if std > 0 else 0.0)
    # Clamp à [-3, 3] pour éviter les valeurs extrêmes qui écrasent le graphique
    vals_zscore = [max(-3.0, min(3.0, v)) for v in vals_zscore]

    fig_radar = go.Figure()
    # Ligne de référence à 0 (= moyenne de tous les clusters)
    fig_radar.add_trace(go.Scatterpolar(
        r=[0] * (len(feat_radar) + 1),
        theta=feat_radar + [feat_radar[0]],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="Moyenne",
        showlegend=True,
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_zscore + [vals_zscore[0]],
        theta=feat_radar + [feat_radar[0]],
        fill="toself",
        line_color=CLUSTER_COLORS[selected_cluster % len(CLUSTER_COLORS)],
        name=cluster_label(selected_cluster),
        showlegend=True,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[-3, 3], tickvals=[-2, -1, 0, 1, 2])),
        title="Profil du cluster (z-score, 0 = moyenne nationale)",
    )
    st.plotly_chart(fig_radar, width="stretch")

# ===========================================================================
# ONGLET 3 — Sélection du K optimal
# ===========================================================================

with tab3:
    st.subheader("Sélection du nombre optimal de clusters")
    if metrics is not None:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_elbow = px.line(metrics, x="k", y="inertia", markers=True, title="Méthode Elbow")
            st.plotly_chart(fig_elbow, width="stretch")
        with col_m2:
            fig_sil = px.bar(metrics, x="k", y="silhouette", color="silhouette", title="Score silhouette")
            st.plotly_chart(fig_sil, width="stretch")

        st.dataframe(metrics.set_index("k").round(4), width="stretch")

# ===========================================================================
# ONGLET 4 — Carte
# ===========================================================================

with tab4:
    st.subheader("Carte des clusters IRIS")

    import folium
    from streamlit_folium import st_folium

    # Sélecteur département — réduit le nombre de polygones de ~50 000 à ~500-2 000
    deps_available = sorted(df["dep"].dropna().unique())
    selected_dep = st.selectbox(
        "Sélectionner un département",
        options=deps_available,
        index=deps_available.index("75") if "75" in deps_available else 0,
        help="Filtrer la carte par département pour alléger le rendu",
    )

    # 1) Données du département sélectionné
    df_map = df[df["dep"] == selected_dep][["IRIS", "cluster"]].dropna().copy()
    df_map["IRIS"] = df_map["IRIS"].astype(str)
    df_map = df_map.drop_duplicates(subset="IRIS")
    df_map["cluster_label"] = df_map["cluster"].apply(cluster_label)

    if df_map.empty:
        st.warning(f"Aucune donnée IRIS disponible pour le département {selected_dep}.")
        st.stop()

    st.caption(f"{len(df_map):,} IRIS dans le département {selected_dep}")

    cluster_by_iris = dict(zip(df_map["IRIS"], df_map["cluster"]))
    label_by_iris = dict(zip(df_map["IRIS"], df_map["cluster_label"]))
    iris_set = set(df_map["IRIS"])

    color_map = {
        c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for i, c in enumerate(sorted(df["cluster"].unique()))
    }

    # 2) GeoJSON chargé en cache (une seule fois pour toute la session)
    geojson_data = load_geojson()

    if geojson_data is None:
        st.error("❌ Impossible d'afficher la carte : fichier GeoJSON indisponible.")
        st.stop()

    # 3) Filtrer par département uniquement
    features_dep = [
        feat for feat in geojson_data.get("features", [])
        if str(feat.get("properties", {}).get("code_iris", "")) in iris_set
    ]

    if not features_dep:
        st.warning("Aucun contour IRIS correspondant pour ce département.")
        st.stop()

    # 4) Centre de la carte
    try:
        lats, lons = [], []
        for feat in features_dep[:50]:  # échantillon pour le centrage
            geom = feat.get("geometry", {})
            rings = []
            if geom.get("type") == "Polygon":
                rings = geom.get("coordinates", [[]])[0]
            elif geom.get("type") == "MultiPolygon":
                rings = geom.get("coordinates", [[[]]])[0][0]
            for pt in rings:
                lons.append(pt[0])
                lats.append(pt[1])
        map_center = [sum(lats) / len(lats), sum(lons) / len(lons)] if lats else [46.5, 2.5]
    except Exception:
        map_center = [46.5, 2.5]

    # 5) Construction de la carte
    for feat in features_dep:
        iris_code = str(feat.get("properties", {}).get("code_iris", ""))
        feat["properties"]["cluster_label"] = label_by_iris.get(iris_code, "Non classifié")

    geojson_filtered = {"type": "FeatureCollection", "features": features_dep}

    def style_fn(feature):
        iris = str(feature["properties"].get("code_iris", ""))
        cluster = cluster_by_iris.get(iris)
        return {
            "fillColor": color_map.get(cluster, "#cccccc"),
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.7 if cluster is not None else 0.2,
        }

    m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(
        geojson_filtered,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["code_iris", "nom_iris", "cluster_label"],
            aliases=["Code IRIS", "Nom IRIS", "Cluster"],
        ),
    ).add_to(m)

    st_folium(m, width=900, height=600)
    
# ===========================================================================
# ONGLET 5 — Données détaillées
# ===========================================================================

with tab5:
    st.subheader("Données détaillées par IRIS")

    clusters_available = sorted(df["cluster"].unique())
    selected_clusters = st.multiselect(
        "Filtrer par cluster",
        options=clusters_available,
        default=clusters_available,
        format_func=cluster_label,
    )

    df_display = df[df["cluster"].isin(selected_clusters)].copy()
    if cluster_names:
        df_display.insert(1, "nom_cluster", df_display["cluster"].apply(lambda c: cluster_names.get(int(c), "")))

    search = st.text_input("Rechercher un code IRIS")
    if search:
        df_display = df_display[df_display["IRIS"].str.contains(search, case=False, na=False)]

    st.write(f"**{len(df_display):,} IRIS affichés**")
    st.dataframe(df_display.head(500), width="stretch", hide_index=True)

# ===========================================================================
# ONGLET 6 — Sources enrichies
# ===========================================================================

with tab6:
    st.subheader("Sources enrichies utilisées dans le clustering")
    st.markdown("""
    ### 📌 Données intégrées automatiquement :
    - **BPE (Base Permanente des Équipements)** : commerces, santé, éducation, sport, services
    - **Mobilités domicile-travail** : flux entrants/sortants par IRIS
    - **SIRENE** : nombre d’établissements actifs par IRIS
    - **INSEE Filosofi 2020** : revenus, pauvreté, inégalités
    - **INSEE RP 2020** : population, logement, emploi

    Ces données permettent un **clustering territorial beaucoup plus riche**, capturant :
    - attractivité locale,
    - accessibilité aux services,
    - dynamisme économique,
    - structure socio-démographique.
    """)

# ===========================================================================
# ONGLET 7 — Méthodologie
# ===========================================================================

with tab7:
    st.subheader("Méthodologie détaillée")

    # ── 1. Objectif ──
    st.markdown("### 1. Objectif")
    st.markdown("""
    Ce projet vise à segmenter les **~48 000 zones IRIS** de France métropolitaine
    (Îlots Regroupés pour l'Information Statistique, INSEE) en groupes homogènes,
    appelés **clusters**, à partir de leurs caractéristiques socio-démographiques,
    économiques et de mobilité.

    L'IRIS est la maille géographique infra-communale la plus fine publiée par l'INSEE,
    regroupant en moyenne **~2 000 habitants**. La segmentation produite permet d'identifier
    des **profils territoriaux** reproductibles à l'échelle nationale.

    Il s'agit d'un **apprentissage non supervisé** : aucune étiquette cible n'est utilisée.
    Le modèle découvre lui-même la structure des données.
    """)

    # ── 2. Sources de données ──
    st.markdown("### 2. Sources de données")

    sources = {
        "Source": [
            "INSEE Filosofi 2021",
            "INSEE RP 2021 — Population",
            "INSEE RP 2021 — Logement",
            "INSEE RP 2021 — Emploi / Mobilités",
            "BPE 2023 (Base Permanente des Équipements)",
            "SIRENE géolocalisé",
            "Contours IRIS (IGN / Géoplateforme)",
        ],
        "Contenu": [
            "Revenus disponibles, pauvreté, inégalités (Gini, déciles) par IRIS",
            "Structure par âge, sexe, nationalité, immigration par IRIS",
            "Type de logement, ancienneté du bâti, statut d'occupation, HLM",
            "Emploi, chômage, modes de transport domicile–travail",
            "Nombre d'équipements par IRIS (commerces, santé, éducation, sport…)",
            "Nombre d'établissements actifs par IRIS",
            "Polygones géographiques IRIS pour la cartographie",
        ],
        "URL source": [
            "insee.fr (fichier 8229323)",
            "insee.fr (fichier 8268806)",
            "insee.fr (fichier 8268838)",
            "insee.fr (fichier 8268843)",
            "insee.fr (fichier 8217527)",
            "files.data.gouv.fr/geo-sirene",
            "data.geopf.fr (IGN CONTOURS-IRIS 3.0)",
        ],
    }
    st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)

    st.markdown("""
    > **Fallback automatique** : si les données RP 2021 sont indisponibles, le pipeline
    > bascule automatiquement sur les données RP 2020 (préfixes de colonnes `P20_`
    > au lieu de `P21_`). Les ratios sont calculés dans les deux cas.
    """)

    # ── 3. Ingénierie de features ──
    st.markdown("### 3. Ingénierie de features")
    st.markdown("""
    Les données brutes INSEE contiennent des **effectifs absolus** (ex. : nombre de cadres).
    Ces comptes ne sont pas comparables entre IRIS de tailles différentes.

    Le pipeline calcule des **ratios normalisés** (% ou taux), ce qui rend le clustering
    invariant à la taille de l'IRIS :
    """)

    feat_table = {
        "Variable calculée": [
            "LOG_POP", "PCT_CS1 → PCT_CS8", "PCT_IMM / PCT_ETR",
            "PCT_ACTIFS", "TAUX_CHOM", "TAUX_HLM", "TAUX_PROPRIO", "TAUX_VACANCE",
            "PART_JEUNES / PART_SENIORS", "PART_MAISONS",
            "PART_BATI_ANCIEN / _TRENTES_GLORIEUSES / _RECENT",
            "TAUX_SANS_VOITURE / TAUX_MULTI_VOITURES",
            "TAUX_TC / TAUX_VOITURE_TRAVAIL / TAUX_MOBILITE_DOUCE",
            "DENSITE_ETAB",
        ],
        "Définition": [
            "log(1 + population) — log pour réduire l'asymétrie des grandes villes",
            "% de la pop 15+ dans chaque catégorie socio-prof. (CS1=agriculteurs … CS8=inactifs)",
            "Part d'immigrés / d'étrangers dans la population totale",
            "Part des 20–64 ans dans la population",
            "Chômeurs / (actifs occupés + chômeurs)",
            "Logements HLM / résidences principales",
            "Propriétaires occupants / logements totaux",
            "Logements vacants / logements totaux",
            "Part des 0–14 ans / Part des 60+ ans dans la population",
            "Maisons individuelles / logements totaux",
            "Part des logements construits avant 1946 / 1946–1990 / après 1990",
            "Ménages sans voiture / Ménages avec ≥2 voitures (sur résidences principales)",
            "Actifs en TC / voiture / vélo+marche domicile–travail (sur actifs occupés)",
            "Nombre d'établissements SIRENE actifs pour 1 000 habitants",
        ],
        "Source brute": [
            "P21_POP", "C21_POP15P_CS[1-8]", "P21_POP_IMM / P21_POP_ETR",
            "P21_POP2064", "P21_CHOM1564 + P21_ACTOCC15P", "P21_RP_LOCHLMV",
            "P21_RP_PROP", "P21_LOGVAC", "P21_POP0014 / P21_POP6074+P21_POP75P",
            "P21_MAISON", "P21_RP_ACH19+P21_RP_ACH45 / P21_RP_ACH70+P21_RP_ACH90 / P21_RP_ACH18",
            "P21_RP - P21_RP_VOIT1P / P21_RP_VOIT2P", "C21_ACTOCC15P_TCOM / _VOIT / _VELO+_MAR",
            "NB_ETABLISSEMENTS (SIRENE) / P21_POP × 1 000",
        ],
    }
    st.dataframe(pd.DataFrame(feat_table), use_container_width=True, hide_index=True)

    st.markdown("""
    Les colonnes Filosofi (revenus, Gini, etc.) sont incluses **en complément** si disponibles.
    Les colonnes avec **100 % de valeurs manquantes** sont automatiquement exclues.
    Les valeurs manquantes restantes sont imputées par la **médiane de la colonne**.

    **IRIS exclus** : les zones de type `A` (activité économique) et `D` (divers/parcs)
    sont retirées avant clustering — leur population est quasi nulle et fausserait la segmentation.
    """)

    # ── 4. Préparation et normalisation ──
    st.markdown("### 4. Préparation et normalisation")
    st.markdown("""
    Avant d'appliquer K-Means, toutes les features sont **standardisées** (z-score via
    `StandardScaler`) : chaque variable a une moyenne 0 et un écart-type 1.

    Cette étape est indispensable car K-Means est sensible à l'échelle : sans normalisation,
    des variables comme `LOG_POP` (valeurs ~5–12) domineraient le calcul de distance
    par rapport à des ratios (valeurs entre 0 et 1).

    **Pipeline scikit-learn :**
    ```
    données brutes → imputation médiane (SimpleImputer) → standardisation (StandardScaler) → KMeans
    ```
    """)

    # ── 5. Sélection du K optimal ──
    st.markdown("### 5. Sélection du nombre de clusters (K)")
    st.markdown("""
    Le nombre de clusters est **déterminé automatiquement** en testant K de 5 à 12.
    Trois métriques sont calculées pour chaque K :

    | Métrique | Interprétation | Optimum |
    |---|---|---|
    | **Silhouette** | Cohésion intra-cluster vs séparation inter-clusters (−1 à +1) | Maximiser |
    | **Davies-Bouldin** | Ratio dispersion intra / distance inter-clusters | Minimiser |
    | **Calinski-Harabasz** | Ratio variance inter / variance intra (F-ratio) | Maximiser |

    Un **score composite** pondéré est calculé :

    ```
    score = 0.60 × silhouette_normalisé
          + 0.25 × (1 − davies_bouldin_normalisé)   ← inversé : plus bas = mieux
          + 0.15 × calinski_harabasz_normalisé
    ```

    Le K avec le score composite le plus élevé est retenu.
    """)

    if metrics is not None:
        best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
        st.info(f"**K retenu pour ce clustering : {best_k}** (silhouette maximale = {metrics['silhouette'].max():.3f})")
        st.dataframe(metrics.set_index("k").round(4), use_container_width=True)

    # ── 6. Algorithme K-Means ──
    st.markdown("### 6. Algorithme K-Means")
    st.markdown("""
    **K-Means** partitionne les IRIS en K groupes en minimisant la **somme des distances
    euclidiennes** entre chaque point et le centroïde de son cluster (inertie intra-cluster).

    Paramètres utilisés :
    - `n_init=20` : 20 initialisations aléatoires indépendantes → le meilleur résultat est conservé
    - `random_state=42` : reproductibilité des résultats
    - Initialisation : k-means++ (valeur par défaut de scikit-learn, meilleure convergence)

    **Limites de K-Means :**
    - Suppose des clusters de forme sphérique (pas adapté aux formes non convexes)
    - Sensible aux outliers (une grande métropole peut créer son propre cluster)
    - Le K doit être fixé à l'avance (d'où la sélection automatique décrite ci-dessus)
    """)

    # ── 7. Réductions de dimension ──
    st.markdown("### 7. Réductions de dimension — PCA et UMAP")

    col_pca, col_umap = st.columns(2)
    with col_pca:
        st.markdown("""
        **PCA (Analyse en Composantes Principales)**
        - Projection linéaire en 2D préservant le maximum de variance
        - Rapide, déterministe, reproductible
        - Utilisation : visualisation dans l'onglet *Vue d'ensemble*
        - `sklearn.decomposition.PCA(n_components=2)`
        """)
    with col_umap:
        st.markdown("""
        **UMAP (Uniform Manifold Approximation and Projection)**
        - Projection non linéaire, préserve mieux la structure locale
        - Révèle des groupes que la PCA ne distingue pas
        - Paramètres : `n_components=2`, `n_neighbors=30`, `random_state=42`
        - Utilisé pour la page *Exploration UMAP* de app2
        """)

    # ── 8. Nommage automatique des clusters ──
    st.markdown("### 8. Nommage automatique des clusters")
    st.markdown("""
    Chaque cluster reçoit un **nom descriptif généré automatiquement** à partir du
    profil moyen de ses IRIS, en 5 dimensions :

    1. **Urbanisation** — z-score de `LOG_POP` :
       - < −0.6 σ → *Rural*  |  > +0.6 σ → *Urbain dense*  |  sinon → *Périurbain / bourg*
    2. **Profil socio-économique** — z-scores des CS, chômage, HLM, immigration
    3. **Structure démographique** — seniors, jeunes, retraités, propriétaires
    4. **Logement** — âge du bâti, maisons individuelles, vacance
    5. **Mobilité** — transports en commun, voiture, mobilité douce

    Un trait est considéré **distinctif** si son z-score inter-clusters dépasse **0.6 σ**.
    Le nom final concatène les traits distinctifs séparés par ` — `.
    """)

    if cluster_names:
        st.markdown("**Clusters identifiés dans ce run :**")
        for cid, name in sorted(cluster_names.items()):
            color = CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)]
            st.markdown(
                f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.85em">C{cid}</span> &nbsp; {name}',
                unsafe_allow_html=True,
            )

    # ── 9. Reproductibilité ──
    st.markdown("### 9. Reproductibilité et mise à jour")
    st.markdown("""
    - Toutes les données sont **téléchargées automatiquement** depuis les sources officielles
      (INSEE, IGN, data.gouv.fr) — aucun fichier manuel requis.
    - Les fichiers intermédiaires sont mis en cache localement dans `data/` et `results/`.
    - Pour relancer un clustering complet depuis zéro : `python clustering.py`
    - Pour mettre à jour uniquement les données : supprimer les `.parquet` dans `data/`
      et relancer.
    - **Millésime actuel** : RP 2021 / Filosofi 2021 / BPE 2023 / SIRENE (mensuel)
    """)


# Footer
st.markdown("---")
st.markdown("**Sources** : INSEE 2021 | BPE 2023 | SIRENE | **Méthode** : K-Means / PCA / UMAP")
