"""
Application Streamlit - Rapport de Clustering des IRIS français
Auteur : lehelud
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# ---------------------------------------------------------------------------
# Utilitaires de chargement
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Chargement des données de clustering...")
def load_clustered_data(method="kmeans"):
      path = os.path.join(RESULTS_DIR, f"iris_clustered_{method}.parquet")
      if not os.path.exists(path):
                return None
            return pd.read_parquet(path)


@st.cache_data(show_spinner="Chargement du profil des clusters...")
def load_profile(method="kmeans"):
      path = os.path.join(RESULTS_DIR, f"cluster_profile_{method}.csv")
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
# Noms lisibles des clusters (à personnaliser après analyse)
# ---------------------------------------------------------------------------

CLUSTER_NAMES = {
      0: "Cluster 0 – À définir",
      1: "Cluster 1 – À définir",
      2: "Cluster 2 – À définir",
      3: "Cluster 3 – À définir",
      4: "Cluster 4 – À définir",
      5: "Cluster 5 – À définir",
      6: "Cluster 6 – À définir",
      7: "Cluster 7 – À définir",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
      st.image("https://www.insee.fr/fr/img/logo-insee.png", width=100)
    st.title("🗺️ Clustering IRIS")
    st.markdown("---")

    method = st.selectbox(
              "Méthode de clustering",
              options=["kmeans", "hierarchical"],
              format_func=lambda x: {"kmeans": "K-Means", "hierarchical": "Hiérarchique"}[x],
    )

    st.markdown("---")
    st.markdown("**Sources de données**")
    st.markdown("- INSEE Filosofi 2020 (revenus)")
    st.markdown("- INSEE Recensement Population 2020")
    st.markdown("- INSEE RP Logement 2020")
    st.markdown("- INSEE RP Emploi 2020")
    st.markdown("---")
    st.markdown("*Données : millésime 2020*")
    st.markdown("*Périmètre : France métropolitaine*")

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

df = load_clustered_data(method)
profile = load_profile(method)
metrics = load_metrics()
df_features = load_features()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🗺️ Clustering des IRIS français")
st.markdown(
      """
      Ce rapport présente une **segmentation automatique des IRIS français** (Îlots Regroupés pour 
      l'Information Statistique) à partir des données open data de l'INSEE. L'objectif est d'identifier 
      des **groupes d'IRIS homogènes** sur des critères socio-économiques, démographiques et de logement.
      """
)

# ---- Données non disponibles ----
if df is None or profile is None:
      st.warning(
          "⚠️ Les résultats de clustering ne sont pas encore disponibles. "
          "Veuillez d'abord exécuter le pipeline :\n\n"
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
    col4.metric("Silhouette score", f"{best_row['silhouette']:.3f}")

st.markdown("---")

# ===========================================================================
# ONGLETS
# ===========================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
      "📊 Vue d'ensemble",
      "🔍 Profil des clusters",
      "📉 Sélection du K",
      "🗺️ Carte",
      "📋 Données détaillées",
])

# ===========================================================================
# ONGLET 1 — Vue d'ensemble
# ===========================================================================

with tab1:
      st.subheader("Répartition des IRIS par cluster")

    # Distribution
    dist = df["cluster"].value_counts().sort_index().reset_index()
    dist.columns = ["cluster", "count"]
    dist["label"] = dist["cluster"].map(lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"))
    dist["pct"] = (dist["count"] / dist["count"].sum() * 100).round(1)

    col_a, col_b = st.columns([1, 1])

    with col_a:
              fig_pie = px.pie(
                            dist, values="count", names="label",
                            color_discrete_sequence=CLUSTER_COLORS,
                            title="Répartition des IRIS",
              )
              fig_pie.update_traces(textposition="inside", textinfo="percent+label")
              st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
              fig_bar = px.bar(
                            dist, x="label", y="count",
                            color="label", color_discrete_sequence=CLUSTER_COLORS,
                            text="pct",
                            labels={"label": "Cluster", "count": "Nombre d'IRIS"},
                            title="Nombre d'IRIS par cluster",
              )
              fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
              fig_bar.update_layout(showlegend=False)
              st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Visualisation PCA (2D)")
    if "pca_x" in df.columns and "pca_y" in df.columns:
              sample = df.sample(min(10000, len(df)), random_state=42)
              sample["cluster_label"] = sample["cluster"].map(lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"))
              fig_pca = px.scatter(
                  sample, x="pca_x", y="pca_y",
                  color="cluster_label",
                  color_discrete_sequence=CLUSTER_COLORS,
                  opacity=0.5, size_max=4,
                  labels={"pca_x": "PC1", "pca_y": "PC2", "cluster_label": "Cluster"},
                  title="Projection PCA des IRIS colorés par cluster",
              )
              fig_pca.update_traces(marker=dict(size=3))
              st.plotly_chart(fig_pca, use_container_width=True)

    if "umap_x" in df.columns and "umap_y" in df.columns:
              st.subheader("Visualisation UMAP (2D)")
              fig_umap = px.scatter(
                  sample, x="umap_x", y="umap_y",
                  color="cluster_label",
                  color_discrete_sequence=CLUSTER_COLORS,
                  opacity=0.5,
                  labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2", "cluster_label": "Cluster"},
                  title="Projection UMAP des IRIS colorés par cluster",
              )
              fig_umap.update_traces(marker=dict(size=3))
              st.plotly_chart(fig_umap, use_container_width=True)

# ===========================================================================
# ONGLET 2 — Profil des clusters
# ===========================================================================

with tab2:
      st.subheader("Profil moyen par cluster")
    st.markdown(
              "Le tableau ci-dessous présente les **valeurs moyennes** des indicateurs "
              "pour chaque cluster. Les valeurs en vert/rouge indiquent les écarts par rapport "
              "à la moyenne nationale."
    )

    # Heatmap des profils normalisés
    feat_cols = [c for c in profile.columns if c != "n_iris"]

    if len(feat_cols) > 0:
              profile_norm = profile[feat_cols].copy()
              for col in feat_cols:
                            col_min = profile_norm[col].min()
                            col_max = profile_norm[col].max()
                            if col_max > col_min:
                                              profile_norm[col] = (profile_norm[col] - col_min) / (col_max - col_min)

                        fig_heat = px.imshow(
                                      profile_norm.T,
                                      color_continuous_scale="RdYlGn",
                                      aspect="auto",
                                      title="Heatmap des profils de clusters (normalisé 0-1)",
                                      labels={"x": "Cluster", "y": "Indicateur", "color": "Valeur norm."},
                        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Sélection d'un cluster pour analyse détaillée
    selected_cluster = st.selectbox(
              "Sélectionner un cluster pour analyse détaillée",
              options=sorted(df["cluster"].unique()),
              format_func=lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"),
    )

    if selected_cluster in profile.index:
              st.subheader(f"Profil du {CLUSTER_NAMES.get(selected_cluster, f'Cluster {selected_cluster}')}")

        row = profile.loc[selected_cluster]
        n = int(row.get("n_iris", 0))
        st.info(f"Ce cluster regroupe **{n:,} IRIS** ({n/n_iris*100:.1f}% du total).")

        # Radar chart
        feat_radar = [c for c in feat_cols if c in row.index][:12]
        if len(feat_radar) >= 3:
                      # Normalisation pour le radar
                      vals_norm = []
                      for f in feat_radar:
                                        col_min = profile[f].min()
                                        col_max = profile[f].max()
                                        if col_max > col_min:
                                                              vals_norm.append((row[f] - col_min) / (col_max - col_min))
        else:
                    vals_norm.append(0.5)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                              r=vals_norm + [vals_norm[0]],
                              theta=feat_radar + [feat_radar[0]],
                              fill="toself",
                              name=f"Cluster {selected_cluster}",
                              line_color=CLUSTER_COLORS[selected_cluster % len(CLUSTER_COLORS)],
            ))
            fig_radar.update_layout(
                              polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                              title=f"Radar — Cluster {selected_cluster}",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Tableau des valeurs
        df_row = pd.DataFrame({
                      "Indicateur": feat_cols,
                      "Valeur moyenne": [row.get(c, np.nan) for c in feat_cols],
        })
        st.dataframe(df_row, use_container_width=True, hide_index=True)

# ===========================================================================
# ONGLET 3 — Sélection du K optimal
# ===========================================================================

with tab3:
      st.subheader("Sélection du nombre optimal de clusters")
    st.markdown(
              "Deux méthodes sont utilisées pour déterminer le nombre de clusters optimal :\n"
              "- **Méthode Elbow** : on cherche le coude dans la courbe d'inertie\n"
              "- **Silhouette score** : on maximise la cohérence intra-cluster"
    )

    if metrics is not None:
              col_m1, col_m2 = st.columns(2)

        with col_m1:
                      fig_elbow = px.line(
                                        metrics, x="k", y="inertia",
                                        markers=True,
                                        title="Méthode Elbow (inertie)",
                                        labels={"k": "Nombre de clusters", "inertia": "Inertie"},
                      )
                      fig_elbow.update_traces(line_color="#E74C3C")
                      k_opt = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
                      fig_elbow.add_vline(x=k_opt, line_dash="dash", line_color="gray",
                                          annotation_text=f"K optimal = {k_opt}")
                      st.plotly_chart(fig_elbow, use_container_width=True)

        with col_m2:
                      fig_sil = px.bar(
                                        metrics, x="k", y="silhouette",
                                        color="silhouette",
                                        color_continuous_scale="Greens",
                                        title="Score de silhouette par K",
                                        labels={"k": "Nombre de clusters", "silhouette": "Silhouette"},
                      )
                      st.plotly_chart(fig_sil, use_container_width=True)

        col_m3, col_m4 = st.columns(2)
        with col_m3:
                      fig_db = px.line(
                                        metrics, x="k", y="davies_bouldin",
                                        markers=True,
                                        title="Indice Davies-Bouldin (min = meilleur)",
                                        labels={"k": "K", "davies_bouldin": "Davies-Bouldin"},
                      )
                      st.plotly_chart(fig_db, use_container_width=True)
                  with col_m4:
                                fig_ch = px.line(
                                                  metrics, x="k", y="calinski_harabasz",
                                                  markers=True,
                                                  title="Indice Calinski-Harabasz (max = meilleur)",
                                                  labels={"k": "K", "calinski_harabasz": "Calinski-Harabasz"},
                                )
                                st.plotly_chart(fig_ch, use_container_width=True)

        st.dataframe(metrics.set_index("k").round(4), use_container_width=True)
else:
        st.info("Les métriques de sélection du K ne sont pas disponibles.")

# ===========================================================================
# ONGLET 4 — Carte
# ===========================================================================

with tab4:
      st.subheader("Carte des clusters IRIS")

    geojson_path = os.path.join(DATA_DIR, "iris_contours.geojson")
    if not os.path.exists(geojson_path):
              st.warning(
                  "⚠️ Les contours géographiques IRIS ne sont pas disponibles. "
                  "Exécutez `data_collection.download_iris_geojson()` pour les télécharger."
    )
else:
        try:
                      import json
                      import folium
                      from streamlit_folium import st_folium

            with open(geojson_path, "r", encoding="utf-8") as f:
                              geojson_data = json.load(f)

            # Merge cluster data
            df_map = df[["IRIS", "cluster"]].copy()
            df_map["cluster_label"] = df_map["cluster"].map(
                              lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}")
            )

            m = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles="CartoDB positron")

            color_map = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                                                  for i, c in enumerate(sorted(df["cluster"].unique()))}

            def style_function(feature):
                              iris_code = feature["properties"].get("CODE_IRIS", "")
                              row = df_map[df_map["IRIS"] == iris_code]
                              if len(row) == 0:
                                                    return {"fillColor": "#cccccc", "color": "#999", "weight": 0.3, "fillOpacity": 0.5}
                                                cluster = int(row.iloc[0]["cluster"])
                return {
                                      "fillColor": color_map.get(cluster, "#cccccc"),
                                      "color": "#333",
                                      "weight": 0.3,
                                      "fillOpacity": 0.7,
                }

            folium.GeoJson(
                              geojson_data,
                              style_function=style_function,
                              tooltip=folium.GeoJsonTooltip(fields=["CODE_IRIS", "NOM_IRIS"], aliases=["Code IRIS", "Nom"]),
            ).add_to(m)

            st_folium(m, width=900, height=600)

except Exception as e:
            st.error(f"Erreur lors du chargement de la carte : {e}")
            st.info("Carte indisponible. Assurez-vous que folium et streamlit-folium sont installés.")

# ===========================================================================
# ONGLET 5 — Données détaillées
# ===========================================================================

with tab5:
      st.subheader("Données détaillées par IRIS")

    # Filtre par cluster
    clusters_available = sorted(df["cluster"].unique())
    selected_clusters = st.multiselect(
              "Filtrer par cluster",
              options=clusters_available,
              default=clusters_available,
              format_func=lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"),
    )

    df_display = df[df["cluster"].isin(selected_clusters)].copy()
    df_display["cluster_label"] = df_display["cluster"].map(
              lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}")
    )

    # Recherche par code IRIS
    search = st.text_input("Rechercher un code ou nom IRIS", "")
    if search:
              mask = df_display["IRIS"].str.contains(search, case=False, na=False)
        df_display = df_display[mask]

    st.write(f"**{len(df_display):,} IRIS affichés**")

    cols_display = ["IRIS", "cluster_label"] + [
              c for c in df_display.columns
              if c not in {"IRIS", "cluster", "cluster_label", "pca_x", "pca_y", "umap_x", "umap_y"}
    ][:15]

    st.dataframe(
              df_display[cols_display].head(500).reset_index(drop=True),
              use_container_width=True,
              hide_index=True,
    )

    # Export CSV
    csv = df_display[cols_display].to_csv(index=False).encode("utf-8")
    st.download_button(
              label="📥 Télécharger les données filtrées (CSV)",
              data=csv,
              file_name="iris_clustering_export.csv",
              mime="text/csv",
    )

# Footer
st.markdown("---")
st.markdown(
      "**Sources** : INSEE Filosofi 2020 · INSEE Recensement Population 2020 · "
      "Contours IRIS data.gouv.fr | "
      "**Méthode** : K-Means / Clustering Hiérarchique · PCA · UMAP"
)
