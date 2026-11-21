"""
Application Streamlit - Rapport de Clustering des IRIS français
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Clustering des IRIS français",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = "results"
DATA_DIR = "data"

CLUSTER_COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly

# Labels français pour les features
FEATURE_LABELS = {
    'LOG_POP':       'Taille IRIS (log pop)',
    'PCT_CS1':       'Agriculteurs',
    'PCT_CS2':       'Artisans / Commerçants',
    'PCT_CS3':       'Cadres',
    'PCT_CS4':       'Prof. intermédiaires',
    'PCT_CS5':       'Employés',
    'PCT_CS6':       'Ouvriers',
    'PCT_CS7':       'Retraités',
    'PCT_CS8':       'Autres inactifs',
    'PART_JEUNES':   'Jeunes (0-14 ans)',
    'PART_SENIORS':  'Seniors (60+ ans)',
    'PCT_ACTIFS':    'Actifs (20-64 ans)',
    'PCT_IMM':       'Immigrés',
    'PCT_ETR':       'Étrangers',
    'TAUX_CHOM':     'Taux de chômage',
    'TAUX_VACANCE':  'Logements vacants',
    'TAUX_PROPRIO':  'Propriétaires',
    'DENSITE_ETAB':  'Densité établissements',
}

# Features à afficher dans le radar (les plus interprétables)
RADAR_FEATURES = [
    'PCT_CS1', 'PCT_CS2', 'PCT_CS3', 'PCT_CS4',
    'PCT_CS5', 'PCT_CS6', 'PCT_CS7',
    'PART_JEUNES', 'PART_SENIORS', 'PCT_IMM', 'LOG_POP',
]

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Chargement des données…")
def load_clustered_data():
    path = os.path.join(RESULTS_DIR, "iris_clustered.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None

@st.cache_data(show_spinner=False)
def load_profile():
    path = os.path.join(RESULTS_DIR, "cluster_profile.csv")
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None

@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(RESULTS_DIR, "metrics_k_selection.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data(show_spinner=False)
def load_features():
    path = os.path.join(DATA_DIR, "iris_features.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None

@st.cache_data(show_spinner=False)
def load_cluster_names():
    """Charge les noms et descriptions des clusters.
    Format : {cluster_id: {"nom": str, "description": str}}
    Compatible avec l'ancien format chaîne simple.
    """
    path = os.path.join(RESULTS_DIR, "cluster_names.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for k, v in raw.items():
        cid = int(k)
        if isinstance(v, dict):
            result[cid] = v
        else:
            result[cid] = {"nom": str(v), "description": ""}
    return result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cluster_label(cid):
    info = cluster_names.get(int(cid), {})
    nom = info.get("nom", "") if isinstance(info, dict) else str(info)
    return f"Cluster {cid} — {nom}" if nom else f"Cluster {cid}"

def cluster_description(cid):
    info = cluster_names.get(int(cid), {})
    return info.get("description", "") if isinstance(info, dict) else ""

def feat_label(col):
    return FEATURE_LABELS.get(col, col)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://www.insee.fr/fr/img/logo-insee.png", width=100)
    st.title("🗺️ Clustering IRIS")
    st.markdown("---")

    df       = load_clustered_data()
    profile  = load_profile()
    metrics  = load_metrics()
    df_feat  = load_features()
    cluster_names = load_cluster_names()

    if cluster_names:
        st.markdown("### Clusters identifiés")
        for cid, info in sorted(cluster_names.items()):
            nom  = info.get("nom", f"Cluster {cid}") if isinstance(info, dict) else str(info)
            desc = info.get("description", "") if isinstance(info, dict) else ""
            # Premier fragment de description (n_iris + hab)
            short = " · ".join(desc.split(" · ")[:2]) if desc else ""
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            st.markdown(
                f'<div style="border-left:4px solid {color};padding-left:8px;margin-bottom:8px">'
                f'<b>Cluster {cid}</b><br><small>{nom}</small>'
                + (f'<br><small style="color:gray">{short}</small>' if short else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("*Sources : INSEE RP 2020/2021 · Filosofi · BPE · SIRENE*")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🗺️ Clustering enrichi des IRIS français")
st.markdown(
    "Segmentation territoriale basée sur des **ratios sociodémographiques** : "
    "structure professionnelle, démographie, immigration — invariante à la taille de l'IRIS."
)

if df is None or profile is None:
    st.warning(
        "⚠️ Résultats absents. Lancez d'abord :\n```bash\npython clustering.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

n_clusters = df["cluster"].nunique()
n_iris     = len(df)
n_features = len([c for c in df.columns if c not in {"IRIS", "cluster", "pca_x", "pca_y", "umap_x", "umap_y"}])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clusters", n_clusters)
col2.metric("IRIS analysés", f"{n_iris:,}")
col3.metric("Variables", n_features)

if metrics is not None:
    best_row = metrics.loc[metrics["silhouette"].idxmax()]
    col4.metric("Silhouette max", f"{best_row['silhouette']:.3f}", f"k={int(best_row['k'])}")

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
    "📚 Sources",
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
            color_discrete_sequence=CLUSTER_COLORS,
            title="Répartition des IRIS",
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col_b:
        fig_bar = px.bar(
            dist, x="label", y="count",
            color="label", color_discrete_sequence=CLUSTER_COLORS,
            text="pct", title="Nombre d'IRIS par cluster",
        )
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(showlegend=False, xaxis_tickangle=-25, xaxis_title="")
        st.plotly_chart(fig_bar, width="stretch")

    # Tableau récapitulatif
    if cluster_names:
        st.subheader("Synthèse des clusters")
        rows = []
        for _, row in dist.iterrows():
            cid  = row["cluster"]
            info = cluster_names.get(int(cid), {})
            nom  = info.get("nom", "") if isinstance(info, dict) else str(info)
            desc = info.get("description", "") if isinstance(info, dict) else ""
            rows.append({
                "Cluster": int(cid),
                "Nom": nom,
                "IRIS": int(row["count"]),
                "Part (%)": float(row["pct"]),
                "Détails": desc,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Projections dimensionnelles
    st.subheader("Visualisation PCA / UMAP")
    sample = df.sample(min(15000, len(df)), random_state=42).copy()
    sample["label"] = sample["cluster"].apply(cluster_label)

    if "pca_x" in df.columns:
        fig_pca = px.scatter(
            sample, x="pca_x", y="pca_y", color="label",
            color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.5, title="Projection PCA (2D)",
        )
        st.plotly_chart(fig_pca, width="stretch")

    if "umap_x" in df.columns:
        fig_umap = px.scatter(
            sample, x="umap_x", y="umap_y", color="label",
            color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.5, title="Projection UMAP (2D)",
        )
        st.plotly_chart(fig_umap, width="stretch")

# ===========================================================================
# ONGLET 2 — Profil des clusters
# ===========================================================================

with tab2:
    st.subheader("Profil moyen par cluster")

    feat_cols = [c for c in profile.columns if c != "n_iris"]

    # --- Heatmap normalisée ---
    profile_norm = profile[feat_cols].copy()
    for col in feat_cols:
        lo, hi = profile_norm[col].min(), profile_norm[col].max()
        if hi > lo:
            profile_norm[col] = (profile_norm[col] - lo) / (hi - lo)

    # Labels français sur les axes
    renamed_cols = [feat_label(c) for c in feat_cols]
    profile_norm.columns = renamed_cols
    profile_norm.index   = [cluster_label(i) for i in profile_norm.index]

    fig_heat = px.imshow(
        profile_norm.T,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Heatmap des profils (normalisé 0-1 inter-clusters)",
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, width="stretch")

    st.markdown("---")

    # --- Sélecteur de cluster ---
    cluster_options = sorted(df["cluster"].unique())
    selected_cluster = st.selectbox(
        "Cluster à analyser",
        options=cluster_options,
        format_func=cluster_label,
    )

    row = profile.loc[selected_cluster]
    nom  = cluster_names.get(int(selected_cluster), {}).get("nom", "") if cluster_names else ""
    desc = cluster_description(selected_cluster)

    st.success(f"**{cluster_label(selected_cluster)}**  ·  {int(row['n_iris']):,} IRIS")
    if desc:
        st.caption(desc)

    col_r, col_top = st.columns([1, 1])

    # --- Radar chart avec labels français ---
    with col_r:
        radar_available = [c for c in RADAR_FEATURES if c in feat_cols]
        if radar_available:
            vals_norm = []
            for f in radar_available:
                f_min, f_max = profile[f].min(), profile[f].max()
                v = (row[f] - f_min) / (f_max - f_min) if f_max > f_min else 0.5
                vals_norm.append(float(v))

            theta = [feat_label(f) for f in radar_available]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_norm + [vals_norm[0]],
                theta=theta + [theta[0]],
                fill="toself",
                name=f"Cluster {selected_cluster}",
                line_color=CLUSTER_COLORS[selected_cluster % len(CLUSTER_COLORS)],
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"Profil radar — Cluster {selected_cluster}",
                height=380,
            )
            st.plotly_chart(fig_radar, width="stretch")

    # --- Top features distinctives ---
    with col_top:
        # Z-scores pour ce cluster
        z_scores = {}
        for col in feat_cols:
            vals = pd.to_numeric(profile[col], errors='coerce')
            std = vals.std()
            if std > 1e-9:
                z_scores[col] = float((row[col] - vals.mean()) / std)

        top_df = (
            pd.DataFrame({"feature": list(z_scores.keys()), "z": list(z_scores.values())})
            .assign(abs_z=lambda x: x["z"].abs())
            .sort_values("abs_z", ascending=False)
            .head(10)
            .sort_values("z")
        )
        top_df["label"] = top_df["feature"].apply(feat_label)
        top_df["couleur"] = top_df["z"].apply(lambda v: "Au-dessus de la moyenne" if v >= 0 else "En-dessous de la moyenne")

        fig_top = px.bar(
            top_df, x="z", y="label", orientation="h",
            color="couleur",
            color_discrete_map={"Au-dessus de la moyenne": "#2ecc71", "En-dessous de la moyenne": "#e74c3c"},
            title="Top 10 features distinctives (z-score)",
        )
        fig_top.update_layout(
            yaxis_title="", xaxis_title="Z-score inter-clusters",
            legend_title="", height=380,
        )
        fig_top.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_top, width="stretch")

    # --- Comparaison tous clusters sur une feature ---
    st.markdown("---")
    st.subheader("Comparaison inter-clusters")
    feat_choice = st.selectbox(
        "Feature à comparer",
        options=feat_cols,
        format_func=feat_label,
        index=feat_cols.index("PCT_CS3") if "PCT_CS3" in feat_cols else 0,
    )
    comp_df = pd.DataFrame({
        "label": [cluster_label(i) for i in profile.index],
        "valeur": profile[feat_choice].values * (100 if profile[feat_choice].max() <= 1.5 else 1),
    })
    unit = "%" if profile[feat_choice].max() <= 1.5 else ""
    fig_comp = px.bar(
        comp_df, x="label", y="valeur",
        color="label", color_discrete_sequence=CLUSTER_COLORS,
        title=f"{feat_label(feat_choice)} par cluster",
        labels={"valeur": f"{feat_label(feat_choice)} {unit}", "label": ""},
    )
    fig_comp.update_layout(showlegend=False, xaxis_tickangle=-20)
    st.plotly_chart(fig_comp, width="stretch")

# ===========================================================================
# ONGLET 3 — Sélection du K optimal
# ===========================================================================

with tab3:
    st.subheader("Sélection du nombre optimal de clusters")
    if metrics is not None:
        c1, c2 = st.columns(2)
        with c1:
            fig_elbow = px.line(metrics, x="k", y="inertia", markers=True, title="Méthode Elbow (inertie)")
            st.plotly_chart(fig_elbow, width="stretch")
        with c2:
            fig_sil = px.bar(metrics, x="k", y="silhouette", color="silhouette",
                             color_continuous_scale="Viridis", title="Score Silhouette")
            st.plotly_chart(fig_sil, width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            fig_db = px.bar(metrics, x="k", y="davies_bouldin", color="davies_bouldin",
                            color_continuous_scale="Reds_r", title="Davies-Bouldin (↓ mieux)")
            st.plotly_chart(fig_db, width="stretch")
        with c4:
            if "calinski_harabasz" in metrics.columns:
                fig_ch = px.bar(metrics, x="k", y="calinski_harabasz", color="calinski_harabasz",
                                color_continuous_scale="Blues", title="Calinski-Harabasz (↑ mieux)")
                st.plotly_chart(fig_ch, width="stretch")

        st.markdown("**Score composite** = 60% Silhouette + 25% Davies-Bouldin (inversé) + 15% Calinski-Harabasz (normalisés)")
        if "score" in metrics.columns:
            fig_score = px.bar(metrics, x="k", y="score", color="score",
                               color_continuous_scale="Greens", title="Score composite normalisé")
            st.plotly_chart(fig_score, width="stretch")

        st.dataframe(metrics.set_index("k").round(4), width="stretch")

# ===========================================================================
# ONGLET 4 — Carte
# ===========================================================================

with tab4:
    st.subheader("Carte des clusters IRIS")

    from data_collection import ensure_iris_geojson
    import folium
    from streamlit_folium import st_folium

    df_map = df[["IRIS", "cluster"]].dropna().copy()
    df_map["IRIS"] = df_map["IRIS"].astype(str)
    df_map = df_map.drop_duplicates(subset="IRIS")
    df_map["cluster_label"] = df_map["cluster"].apply(cluster_label)

    if df_map.empty:
        st.warning("Aucune donnée IRIS disponible pour la carte.")
        st.stop()

    cluster_by_iris = dict(zip(df_map["IRIS"], df_map["cluster"]))
    label_by_iris   = dict(zip(df_map["IRIS"], df_map["cluster_label"]))

    color_map = {
        c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for i, c in enumerate(sorted(df_map["cluster"].unique()))
    }

    geojson_path = ensure_iris_geojson()

    if geojson_path is None:
        st.error("❌ Fichier GeoJSON indisponible.")
        st.stop()

    try:
        import json as _json
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = _json.load(f)
    except Exception as e:
        st.error(f"❌ Erreur lecture GeoJSON : {e}")
        st.stop()

    iris_set = set(df_map["IRIS"])
    geojson_filtered = {
        "type": "FeatureCollection",
        "features": [
            feat for feat in geojson_data.get("features", [])
            if str(feat.get("properties", {}).get("code_iris", "")) in iris_set
        ],
    }

    if not geojson_filtered["features"]:
        st.warning("Aucun contour IRIS correspondant.")
        st.stop()

    # Injecter le label dans les propriétés GeoJSON pour le tooltip
    for feat in geojson_filtered["features"]:
        iris_code = str(feat.get("properties", {}).get("code_iris", ""))
        feat["properties"]["cluster_label"] = label_by_iris.get(iris_code, "Non classifié")

    # Centre de la carte
    try:
        coords = []
        for feat in geojson_filtered["features"]:
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                for ring in geom.get("coordinates", []):
                    coords.extend(ring)
            elif geom.get("type") == "MultiPolygon":
                for poly in geom.get("coordinates", []):
                    for ring in poly:
                        coords.extend(ring)
        map_center = [sum(p[1] for p in coords) / len(coords), sum(p[0] for p in coords) / len(coords)] if coords else [46.5, 2.5]
    except Exception:
        map_center = [46.5, 2.5]

    m = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB positron")

    folium.GeoJson(
        geojson_filtered,
        style_function=lambda feat: {
            "fillColor": color_map.get(cluster_by_iris.get(str(feat["properties"].get("code_iris", "")), -1), "#cccccc"),
            "color": "black",
            "weight": 0.2,
            "fillOpacity": 0.7,
        },
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
        df_display.insert(1, "nom_cluster", df_display["cluster"].apply(
            lambda c: cluster_names.get(int(c), {}).get("nom", "") if isinstance(cluster_names.get(int(c), {}), dict) else ""
        ))

    search = st.text_input("Rechercher un code IRIS")
    if search:
        df_display = df_display[df_display["IRIS"].str.contains(search, case=False, na=False)]

    st.write(f"**{len(df_display):,} IRIS affichés**")
    st.dataframe(df_display.head(500), width="stretch", hide_index=True)

# ===========================================================================
# ONGLET 6 — Sources
# ===========================================================================

with tab6:
    st.subheader("Sources et méthodologie")
    st.markdown("""
    ### Données utilisées
    | Source | Contenu | Année |
    |--------|---------|-------|
    | **INSEE Filosofi** | Revenus, pauvreté, inégalités | 2020-2021 |
    | **INSEE RP Population** | Structure par âge, nationalité | 2020-2021 |
    | **INSEE RP Logement** | Vacance, propriété | 2020-2021 |
    | **INSEE RP Emploi** | Chômage, CSP | 2020-2021 |
    | **BPE** | Équipements (santé, commerce, sport…) | 2023 |
    | **SIRENE** | Établissements actifs géolocalisés | Continu |

    ### Améliorations v2
    - **Features normalisées** : ratios (%) au lieu de comptes bruts → invariance à la taille de l'IRIS
    - **Écrêtage 1%-99%** : réduction de l'impact des IRIS extrêmes (villages agricoles, cités étudiantes…)
    - **Suppression des redondances** : PCT_ETR exclue (r=0.97 avec PCT_IMM)
    - **RobustScaler** : mise à l'échelle robuste aux valeurs aberrantes résiduelles
    - **Score composite normalisé** : Silhouette (60%) + Davies-Bouldin (25%) + Calinski-Harabasz (15%)
    - **Nommage automatique** via z-scores inter-clusters sur les dimensions clés
    """)

st.markdown("---")
st.markdown("**Sources** : INSEE 2020-2021 · BPE · SIRENE | **Méthode** : K-Means · RobustScaler · PCA · UMAP")
