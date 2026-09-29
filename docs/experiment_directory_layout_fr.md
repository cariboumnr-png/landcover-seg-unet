# Structure et Arborescence du Répertoire des Expériences et Artefacts

Ce document définit la structure canonique de répertoires du système de fichiers pour
les jeux de données d'entrée, les artefacts de pipeline et les résultats d'exécution
de session sous un répertoire racine d'expérience (`<exp_root>`).

> [!NOTE]
> **Avis relatif à la génération de données** : Le dépôt ne contient **pas** de jeux de
données géospatiales réels ni d'imagerie satellite propriétaire. Pour alimenter
un environnement de test local avec un ensemble complet de fichiers GeoTIFF synthétiques, exécutez :
> ```bash
> python scripts/generate_dummy_data.py
> ```
> Ce script utilise la fabrique centrale `dummy_geotiff_factory` pour générer l'arborescence
de données factices illustrée dans le dossier `input/` ci-dessous. L'exécution des pipelines
du projet sur cette arborescence d'entrée alimente les sous-répertoires d'artefacts (`artifacts/`)
et de résultats (`results/`) correspondants.

> [!TIP]
> **Découplage et Isolement des Pipelines** :
> - **`data-harmonize` (ETL)** : Lit les fichiers GeoTIFF non harmonisés depuis `input/raw/`
> et produit des rasters virtuels réprojetés dans `artifacts/harmonized_data/`.
> - **Pipelines en aval (`data-ingest`, `data-prepare`, `model-train`)** : Consomment
> les GeoTIFF pré-alignés issus de `input/data/` ou les sorties VRT harmonisées de `artifacts/harmonized_data/`.
> - **Isolement** : L'exécution de `data-harmonize` n'écrase **ni** n'interfère
> avec `input/data/`, ce qui permet d'exécuter et de tester chaque pipeline de manière
> indépendante sans couplage séquentiel strict.

```text
<exp_root>/                                      # Par défaut : ./experiment/
│
├── input/                                       # Généré par scripts/generate_dummy_data.py
│   │
│   ├── raw/                                     # Fichiers GeoTIFF bruts non harmonisés (utilisés par l'ETL)
│   │   ├── sample_sentinel2.tif                 # Composite factice Sentinel-2 à 10 bandes uint16 (EPSG:32618)
│   │   ├── sample_dem.tif                       # Raster d'élévation MNE factice à 1 bande float32 (EPSG:32618)
│   │   └── sample_landcover.tif                 # Masque d'étiquette de couverture du sol à 1 bande uint8 (EPSG:32618)
│   │
│   ├── data/                                    # Rasters d'images et d'étiquettes de développement/test (utilisés par data-ingest)
│   │   ├── sample_dev_image.tif                 # GeoTIFF composite de caractéristiques de développement
│   │   ├── sample_dev_label.tif                 # GeoTIFF de masque d'étiquette de développement
│   │   ├── sample_test_image.tif                # GeoTIFF composite de caractéristiques de test
│   │   └── sample_test_label.tif                # GeoTIFF de masque d'étiquette de test
│   │
│   ├── domain_knowledge/                        # Rasters de connaissances du domaine catégoriels
│   │   ├── sample_domain_1.tif                  # Carte de connaissances du domaine 1
│   │   └── sample_domain_2.tif                  # Carte de connaissances du domaine 2
│   │
│   └── reference_raster/                        # Rasters de référence d'étendue spatiale et AOI
│       ├── sample_extent.tif                    # Raster de référence d'étendue (EPSG:3161)
│       └── sample_test_aoi.tif                  # Raster AOI de test (EPSG:3161)
│
├── artifacts/                                   # Généré par l'exécution des pipelines
│   │
│   ├── world_grids/                             # Produit par le pipeline 'world-grid'
│   │   ├── grid_row_<srow>_<orow>_col_<scol>_<ocol>.json *
│   │   └── grid_report.json                     # Rapport de synthèse d'exécution du quadrillage mondial
│   │
│   ├── harmonized_data/                         # Produit par le pipeline 'data-harmonize'
│   │   ├── harmonization_runs.json              # Registre cumulatif d'harmonisation
│   │   └── run_0001/                            # Répertoire d'exécution ETL sérialisé
│   │       ├── harmonized_<name>.vrt            # Raster virtuel source unique réprojeté
│   │       ├── harmonized_features_STACKED.vrt  # Raster virtuel composite de caractéristiques empilé
│   │       ├── harmonized_labels_STACKED.vrt    # Raster virtuel composite d'étiquettes empilé
│   │       ├── valid_pixel_mask.vrt             # Raster virtuel de masque de pixels valides booléen à 1 bande
│   │       ├── harmonize_report.json            # Rapport de synthèse d'exécution de l'harmonisation
│   │       └── config.json                      # Enregistrement de configuration de l'harmonisation
│   │
│   ├── ingested_data/                           # Produit par le pipeline 'data-ingest'
│   │   ├── ingestion_runs.json                  # Registre cumulatif d'ingestion
│   │   ├── run_0001/                            # Répertoire de lot sérialisé
│   │   │   ├── ingest_report.json               # Rapport de synthèse d'ingestion
│   │   │   ├── collisions.json                  # Registre des collisions entre lots
│   │   │   └── config.json                      # Enregistrement de configuration
│   │   ├── domain_knowledge/                    # Bassin de connaissances de domaine
│   │   │   ├── <domain_name>.json *
│   │   │   └── <domain_name>_tiles_<gid>.npz
│   │   └── data_blocks/                         # Bassin canonique de blocs de données
│   │       ├── blocks/                          # Tableaux bruts non normalisés
│   │       ├── windows/                         # Manifestes d'index des fenêtres spatiales
│   │       │   └── windows_<gid>.json *
│   │       ├── catalog.json                     # Catalogue cumulatif et traçabilité
│   │       └── schema.json                      # Schéma structurel et block_identity
│   │
│   └── prepared_data/                           # Produit par le pipeline 'data-prepare'
│       ├── train_blocks/                        # Bloc de tableaux d'entraînement préparés
│       ├── val_blocks/                          # Bloc de tableaux de validation préparés
│       ├── test_blocks/                         # Bloc de tableaux de test préparés
│       ├── block_splits_source.json
│       ├── block_splits_summary.json
│       ├── block_splits_prepared.json
│       ├── label_stats.json
│       ├── image_stats.json
│       ├── schema.json
│       ├── prep_report.json                     # Rapport de synthèse d'exécution de la préparation
│       └── config.json                          # Enregistrement de configuration de la préparation
│
└── results/                                     # Produit par 'model-train' / 'model-evaluate'
    └── run_0001/                                # Répertoire de session d'expérience sérialisé
        ├── checkpoints/
        │   ├── status.json                      # Suivi du statut de la phase d'entraînement
        │   ├── <name>_best.pt                   # Poids des meilleurs points de contrôle du modèle
        │   └── <name>_last.pt                   # Poids des derniers points de contrôle du modèle
        ├── logs/                                # Journaux d'exécution de la session d'entraînement
        ├── plots/                               # Graphiques de visualisation des métriques et diagnostics
        ├── previews/                            # Aperçus des prédictions de sortie du modèle
        ├── config.json                          # Configuration d'exécution entièrement résolue
        ├── evaluation.json                      # JSON de métriques d'évaluation (si session d'évaluation)
        ├── summary.json                         # JSON de métriques globales de la session
        └── step_results.json                    # JSON de suivi des pertes et métriques par étape

* Remarque : Les fichiers `_meta.json` associés sont générés aux côtés des artefacts JSON de grille, de domaine et de fenêtre.
```
