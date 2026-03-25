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

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://www.insee.fr/fr/img/logo-insee.png", width=100)
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Vue d'ensemble",
    "🔍 Profil des clusters",
    "📉 Sélection du K",
    "🗺️ Carte",
    "📋 Données détaillées",
    "📚 Sources enrichies",
])

# ===========================================================================
# ONGLET 1 — Vue d'ensemble
# ===========================================================================

with tab1:
    st.subheader("Répartition des IRIS par cluster")

    dist = df["cluster"].value_counts().sort_index().reset_index()
    dist.columns = ["cluster", "count"]
    dist["pct"] = (dist["count"] / dist["count"].sum() * 100).round(1)
    dist["label"] = dist["cluster"].apply(cluster_label)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        fig_pie = px.pie(
            dist, values="count", names="label",
            color="label", color_discrete_sequence=CLUSTER_COLORS,
            title="Répartition des IRIS",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_bar = px.bar(
            dist, x="label", y="count",
            color="label", color_discrete_sequence=CLUSTER_COLORS,
            text="pct",
            title="Nombre d'IRIS par cluster",
        )
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tableau récapitulatif des noms
    if cluster_names:
        st.subheader("Profil des clusters")
        names_df = pd.DataFrame([
            {"Cluster": cid, "Nom": name, "IRIS": int(dist.loc[dist["cluster"] == cid, "count"].values[0]) if cid in dist["cluster"].values else 0}
            for cid, name in sorted(cluster_names.items())
        ])
        st.dataframe(names_df, use_container_width=True, hide_index=True)

    st.subheader("Visualisation PCA / UMAP")

    sample = df.sample(min(15000, len(df)), random_state=42).copy()
    sample["label"] = sample["cluster"].apply(cluster_label)

    if "pca_x" in df.columns:
        fig_pca = px.scatter(
            sample, x="pca_x", y="pca_y",
            color="label", color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.5, title="Projection PCA (2D)",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    if "umap_x" in df.columns:
        fig_umap = px.scatter(
            sample, x="umap_x", y="umap_y",
            color="label", color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.5, title="Projection UMAP (2D)",
        )
        st.plotly_chart(fig_umap, use_container_width=True)

# ===========================================================================
# ONGLET 2 — Profil des clusters
# ===========================================================================

with tab2:
    st.subheader("Profil moyen par cluster")

    feat_cols = [c for c in profile.columns if c != "n_iris"]

    profile_norm = profile[feat_cols].copy()
    for col in feat_cols:
        col_min, col_max = profile_norm[col].min(), profile_norm[col].max()
        if col_max > col_min:
            profile_norm[col] = (profile_norm[col] - col_min) / (col_max - col_min)

    fig_heat = px.imshow(
        profile_norm.T,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Heatmap des profils (normalisé)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

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

    # Radar chart
    feat_radar = feat_cols[:12]
    vals_norm = []
    for f in feat_radar:
        c_min, c_max = profile[f].min(), profile[f].max()
        vals_norm.append((row[f] - c_min) / (c_max - c_min) if c_max > c_min else 0.5)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_norm + [vals_norm[0]],
        theta=feat_radar + [feat_radar[0]],
        fill="toself",
        line_color=CLUSTER_COLORS[selected_cluster % len(CLUSTER_COLORS)],
    ))
    st.plotly_chart(fig_radar, use_container_width=True)

# ===========================================================================
# ONGLET 3 — Sélection du K optimal
# ===========================================================================

with tab3:
    st.subheader("Sélection du nombre optimal de clusters")
    if metrics is not None:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_elbow = px.line(metrics, x="k", y="inertia", markers=True, title="Méthode Elbow")
            st.plotly_chart(fig_elbow, use_container_width=True)
        with col_m2:
            fig_sil = px.bar(metrics, x="k", y="silhouette", color="silhouette", title="Score silhouette")
            st.plotly_chart(fig_sil, use_container_width=True)

        st.dataframe(metrics.set_index("k").round(4), use_container_width=True)

# ===========================================================================
# ONGLET 4 — Carte
# ===========================================================================

with tab4:
    st.subheader("Carte des clusters IRIS")

    from data_collection import ensure_iris_geojson
    import json
    import folium
    from streamlit_folium import st_folium

    # 1) On prépare les données utiles pour le mapping
    df_map = df[["IRIS", "cluster"]].dropna().copy()
    df_map["IRIS"] = df_map["IRIS"].astype(str)
    df_map = df_map.drop_duplicates(subset="IRIS")
    df_map["cluster_label"] = df_map["cluster"].apply(cluster_label)

    if df_map.empty:
        st.warning("Aucune donnée IRIS disponible pour la carte.")
        st.stop()

    cluster_by_iris = dict(zip(df_map["IRIS"], df_map["cluster"]))
    label_by_iris = dict(zip(df_map["IRIS"], df_map["cluster_label"]))

    color_map = {
        c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for i, c in enumerate(sorted(df_map["cluster"].unique()))
    }

    # 2) On récupère le fichier préparé une seule fois dans data_collection.py
    geojson_path = ensure_iris_geojson()

    if geojson_path is None:
        st.error("❌ Impossible d'afficher la carte : fichier GeoJSON indisponible.")
        st.stop()

    # 3) Lecture du fichier
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier GeoJSON : {e}")
        st.stop()

    # 4) Filtrer uniquement les IRIS présents dans df
    iris_set = set(df_map["IRIS"])

    geojson_data_filtered = {
        "type": "FeatureCollection",
        "features": [
            feature
            for feature in geojson_data.get("features", [])
            if str(feature.get("properties", {}).get("code_iris", "")) in iris_set
        ]
    }

    if not geojson_data_filtered["features"]:
        st.warning("Aucun contour IRIS correspondant aux données à afficher.")
        st.stop()

    # 5) Déterminer le centre de la carte
    try:
        coords = []
        for feat in geojson_data_filtered["features"]:
            geom = feat.get("geometry", {})
            gtype = geom.get("type")

            if gtype == "Polygon":
                for ring in geom.get("coordinates", []):
                    coords.extend(ring)
            elif gtype == "MultiPolygon":
                for polygon in geom.get("coordinates", []):
                    for ring in polygon:
                        coords.extend(ring)

        if coords:
            lon = sum(pt[0] for pt in coords) / len(coords)
            lat = sum(pt[1] for pt in coords) / len(coords)
            map_center = [lat, lon]
        else:
            map_center = [46.5, 2.5]
    except Exception:
        map_center = [46.5, 2.5]

    # 6) Création carte
    m = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB positron")

    # Injecter le label de cluster dans les propriétés GeoJSON pour le tooltip
    for feature in geojson_data_filtered["features"]:
        iris_code = str(feature.get("properties", {}).get("code_iris", ""))
        feature["properties"]["cluster_label"] = label_by_iris.get(iris_code, "Non classifié")

    def style_fn(feature):
        iris = str(feature["properties"].get("code_iris", ""))
        cluster = cluster_by_iris.get(iris)

        return {
            "fillColor": color_map.get(cluster, "#cccccc"),
            "color": "black",
            "weight": 0.2,
            "fillOpacity": 0.7 if cluster is not None else 0.2,
        }

    folium.GeoJson(
        geojson_data_filtered,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["code_iris", "nom_iris", "cluster_label"],
            aliases=["Code IRIS", "Nom IRIS", "Cluster"],
        )
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
    st.dataframe(df_display.head(500), use_container_width=True, hide_index=True)

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

# Footer
st.markdown("---")
st.markdown("**Sources** : INSEE 2020 | BPE | Mobilités | SIRENE | **Méthode** : K-Means / PCA / UMAP")
