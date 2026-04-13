# 🗺️ Clustering des IRIS français (http://37.59.98.254/clustering-iris/)

> Segmentation automatique des IRIS (Îlots Regroupés pour l'Information Statistique) à partir des données open data INSEE — Rapport interactif Streamlit.
>
> ## 🎯 Objectif
>
> Ce projet vise à **identifier des groupes d'IRIS français qui se ressemblent** sur des critères socio-économiques, démographiques et de logement, en exploitant les données open data disponibles au niveau IRIS.
>
> Les résultats sont présentés et commentés dans une **application Streamlit interactive**.
>
> ---
>
> ## 📊 Sources de données
>
> | Source | Description | Millésime |
> |--------|-------------|-----------|
> | [INSEE Filosofi](https://www.insee.fr/fr/statistiques/7233950) | Revenus, pauvreté, inégalités | 2020 |
> | [INSEE RP – Population](https://www.insee.fr/fr/statistiques/7704076) | Structure démographique | 2020 |
> | [INSEE RP – Logement](https://www.insee.fr/fr/statistiques/7704076) | Parc immobilier, occupation | 2020 |
> | [INSEE RP – Emploi](https://www.insee.fr/fr/statistiques/7704076) | Activité, chômage | 2020 |
> | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/contours-iris/) | Contours géographiques IRIS | 2023 |
>
> ---
>
> ## 🧠 Variables utilisées
>
> ### Revenus & pauvreté (Filosofi)
> - Revenu médian disponible par UC (`DISP_MED20`)
> - - 1er et 3e quartile du revenu (`DISP_Q120`, `DISP_Q320`)
>   - - Taux de pauvreté à 60% (`TP6020`)
>     - - Indice de Gini (`DISP_GI20`)
>      
>       - ### Démographie (RP)
>       - - Population totale et structure par âge
>         - - Part des jeunes (0-14 ans), des seniors (60+)
>           - - Part des cadres et des ouvriers
>            
>             - ### Logement (RP)
>             - - Taux de vacance
>               - - Taux de propriétaires
>                 - - Part de résidences secondaires
>                  
>                   - ### Emploi (RP)
>                   - - Taux de chômage
>                     - - Part des actifs en temps partiel
>                      
>                       - ---
>
> ## ⚙️ Méthodes de clustering
>
> - **K-Means** (méthode principale)
> - - **Clustering Hiérarchique Agglomératif** (linkage Ward)
>   - - **DBSCAN** (détection d'outliers)
>    
>     - ### Sélection du nombre optimal de clusters
>     - - Méthode **Elbow** (courbe d'inertie)
>       - - **Silhouette score**
>         - - **Indice Davies-Bouldin**
>           - - **Indice Calinski-Harabasz**
>            
>             - ### Réduction de dimension
>             - - **PCA** (2D) pour visualisation
>               - - **UMAP** (2D) pour visualisation non-linéaire
>                
>                 - ---
>
> ## 🚀 Installation & Lancement
>
> ### 1. Cloner le dépôt
> ```bash
> git clone https://github.com/lehelud/iris-clustering-france.git
> cd iris-clustering-france
> ```
>
> ### 2. Installer les dépendances
> ```bash
> pip install -r requirements.txt
> ```
>
> ### 3. Collecter les données
> ```bash
> python data_collection.py
> ```
>
> ### 4. Lancer le clustering
> ```bash
> python clustering.py
> ```
>
> ### 5. Lancer l'application Streamlit
> ```bash
> streamlit run app.py
> ```
>
> ---
>
> ## 📁 Structure du projet
>
> ```
> iris-clustering-france/
> ├── app.py                  # Application Streamlit (rapport interactif)
> ├── clustering.py           # Pipeline de clustering (KMeans, Hiérarchique, DBSCAN)
> ├── data_collection.py      # Collecte et fusion des données INSEE open data
> ├── requirements.txt        # Dépendances Python
> ├── data/                   # Données téléchargées (gitignored)
> │   ├── iris_features.parquet
> │   ├── filosofi_iris.parquet
> │   └── iris_contours.geojson
> └── results/                # Résultats du clustering (gitignored)
>     ├── iris_clustered_kmeans.parquet
>     ├── cluster_profile_kmeans.csv
>     └── metrics_k_selection.csv
> ```
>
> ---
>
> ## 📱 Rapport Streamlit
>
> L'application propose 5 onglets :
>
> 1. **📊 Vue d'ensemble** — Distribution des clusters, visualisation PCA/UMAP
> 2. 2. **🔍 Profil des clusters** — Heatmap, radar chart, tableau des indicateurs
>    3. 3. **📉 Sélection du K** — Courbes Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz
>       4. 4. **🗺️ Carte** — Carte choroplèthe interactive des clusters par IRIS (Folium)
>          5. 5. **📋 Données détaillées** — Tableau filtrable + export CSV
>            
>             6. ---
>            
>             7. ## 🔧 Technologies
>            
>             8. ![Python](https://img.shields.io/badge/Python-3.10+-blue)
> ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
> ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
> ![Plotly](https://img.shields.io/badge/Plotly-5.15+-purple)
>
> ---
>
> ## 📝 Licence
>
> MIT — Données INSEE sous licence [Etalab Open Licence](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)
> 
