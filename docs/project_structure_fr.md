## Structure actuelle du projet

Dernière mise à jour: 2026-09-20

Ce document resume l'organisation actuelle du depot et la responsabilite
principale de chaque zone. Il privilegie les frontieres utiles pour ajouter ou
deplacer du code, plutot qu'une liste exhaustive de tous les fichiers.

```text
./
|-- docs/                         Documentation, diagrammes et ADRs
|   |-- ADRs/                     Decisions d'architecture
|   |-- images/                   Images de documentation
|   |-- data_preparation_fr.md    Guide de preparation des donnees
|   |-- workflow_chart_fr.md      Diagramme general du workflow
|   `-- project_structure_fr.md   Ce fichier
|
|-- experiment/                   I/O locale ignoree des experiences
|   |-- artifacts/                Grilles, manifestes, checkpoints, etc. generes
|   |-- input/                    Entrees locales d'experience
|   `-- results/                  Sorties locales de pipelines/sessions
|
|-- notebooks/
|   |-- 01_data_preparation.ipynb Demo d'ingestion, validation et preparation
|   `-- 02_model_train.ipynb      Demo d'entrainement de modele de bout en bout
|
|-- src/landseg/                  Package Python installable
|   |-- adapters/                 Surfaces d'entree externes
|   |-- artifacts/                Chemins, politiques, payloads et checkpoints
|   |-- configs/                  Configs Hydra YAML et schemas structures
|   |-- core/                     Contrats transversaux et types de resultats
|   |-- execution/                Registre de pipelines et dispatch d'execution
|   |-- geopipe/                  Pipelines géospatiaux d'harmonisation, d'ingestion et de préparation
|   |-- models/                   Frames, backbones, tetes, conditionnement, factories
|   |-- session/                  Construction, orchestration et moteurs runtime
|   |-- study/                    Sweeps et analyse post-execution
|   |-- utils/                    Helpers partages de logging et multiprocessing
|   |-- _constants.py             Constantes partagees
|   `-- __init__.py
|
|-- configs/                      Fichiers de configuration utilisateur
|   `-- user.yaml                 Configuration locale principale par pipeline
|
|-- scripts/                      Helper scripts et entry points
|   |-- generate_dummy_data.py    Local dummy dataset generator script
|   `-- run.py                    Bootstrap script for VM/Databricks executions
|
|-- tests/                        Unit and integration test suites
|   |-- conftest.py               Global test fixtures and setups
|   `-- unit/                     Unit tests mirroring package structure
|
|-- pyproject.toml                Package metadata and `landseg` console entry point
|-- README.md                     Project overview
`-- CONTRIBUTING.md               Contribution notes
```

### Organisation Du Package

```text
src/landseg/
|-- adapters/
|   |-- api/
|   |   |-- api.py                Facade d'API programmatique
|   |   `-- configurators/        API helpers pour ingest, prepare et train
|   `-- cli/
|       |-- cli.py                Implementation CLI pilotee par Hydra
|       |-- resolver.py           Helpers de resolution de config CLI
|       `-- translate.py          Couche de traduction de config utilisateur
|
|-- artifacts/
|   |-- checkpoint.py             Sauvegarde/chargement de checkpoints
|   |-- controller.py             Resolve/build/rebuild selon les politiques
|   |-- paths.py                  Chemins canoniques des artefacts
|   |-- payload_io.py             Persistance des payloads et metadonnees
|   `-- policy.py                 Regles de cycle de vie des artefacts
|
|-- configs/
|   |-- hydra/
|   |   |-- config.yaml           Point d'entree de composition Hydra
|   |   |-- dataspecs/            Defaults des specifications de donnees
|   |   |-- data/                 Defaults d'harmonisation, d'ingestion et de préparation
|   |   |-- models/               Defaults des architectures modeles
|   |   |-- pipeline/             Configs harmonize, ingest, prepare, train, eval et study
|   |   |-- session/              Configs runtime, loader, optimiseur, taches, orchestration
|   |   `-- study/                Defaults d'etude et de sweep
|   `-- schema/
|       |-- root.py               Schema structure racine
|       |-- utils.py              Utilitaires de schema
|       `-- sections/             Dataclasses par section: pipeline, session, models, etc.
|
|-- core/
|   |-- data_specs.py             Contrat runtime de specification des donnees
|   |-- model_protocol.py         Protocole de comportement des modeles
|   `-- session_results.py        Sorties structurees des sessions
|
|-- execution/
|   |-- executor.py               Point d'entree unifie d'execution
|   `-- pipelines/
|       |-- _registry.py          Lookup et enregistrement des pipelines
|       |-- world_grid.py         Pipeline de génération de grille monde
|       |-- data_harmonize.py     Pipeline d'harmonisation des données
|       |-- data_ingest.py        Pipeline d'ingestion des données
|       |-- data_prepare.py       Pipeline de préparation des données
|       |-- model_train.py        Pipeline d'entrainement
|       |-- model_evaluate.py     Pipeline d'evaluation
|       |-- diagnose_overfit.py   Diagnostic d'overfit
|       |-- study_sweep.py        Sweep d'hyperparametres
|       `-- study_analysis.py     Analyse des resultats d'etude
|
|-- geopipe/
|   |-- contracts/                Schémas centraux de rapports et de transfert
|   |-- core/                     Abstractions clés, catalogues et grille monde
|   |-- grid/                     Construction, cycle de vie et logger de grille
|   |-- harmonize/                Reprojection, empilement et masques
|   |   |-- manifest/             Schéma de manifeste, normaliseur, compilateur
|   |   |-- rasters/              Empilement, masquage et helpers spatiaux
|   |   |-- taxonomy/             Profilage taxonomique et mapping d'indices
|   |   |-- context.py            Constructeur de contexte amont d'harmonisation
|   |   |-- logger.py             Logger structuré d'étape d'harmonisation
|   |   `-- pipeline.py           Exécuteur d'étape d'harmonisation
|   |-- ingest/                   Cartes de domaine et blocs canoniques
|   |   |-- blocks/               Assemblage de blocs, manifestes et mapping
|   |   |-- domains/              Mapping de raster domaine, PCA et cycle de vie
|   |   |-- context.py            Constructeur de contexte amont d'ingestion
|   |   |-- logger.py             Logger structuré d'étape d'ingestion
|   |   `-- pipeline.py           Exécuteur d'étape d'ingestion
|   |-- prepare/                  Vue en mémoire, partitionnement et écriture
|   |   |-- dataset/              Vue de jeu de données en mémoire et sémantique
|   |   |-- partition/            Intégration AOI spatiale, split et scoring
|   |   |-- materialize/          Stats Welford, piles de cibles et écriture
|   |   |-- context.py            Constructeur de contexte amont de préparation
|   |   |-- logger.py             Logger structuré d'étape de préparation
|   |   `-- pipeline.py           Exécuteur d'étape de préparation
|   |-- factory.py                Factory des DataSpecs
|   `-- utils/                    Contexte raster et helpers de coordonnées
|
|-- models/
|   |-- backbones/
|   |   |-- base.py               Abstractions de base des backbones
|   |   |-- factory.py            Construction des backbones
|   |   `-- unet/                 Corps et composants UNet, UNet++, UNet+++
|   |-- core/
|   |   |-- conditioner/          Conditionnement par concatenation et FiLM
|   |   |-- config.py             Structures de config modele
|   |   |-- domains.py            Helpers modeles lies aux domaines
|   |   |-- heads.py              Composants de tetes de prediction
|   |   `-- safety.py             Validation et garde-fous des modeles
|   |-- frames/                   Modeles complets assemblant backbones et tetes
|   `-- factory.py                Construction modele de haut niveau
|
|-- session/
|   |-- common/                   Alias, evenements et types d'orchestration partages
|   |-- data/                     Adaptateurs Dataset et DataLoader
|   |-- engine/
|   |   |-- builder.py            Construction du moteur
|   |   |-- epoch/                Executor d'epoch et policies train/eval
|   |   `-- runtime/
|   |       |-- builder.py        Construction du runtime
|   |       |-- executor/         Etat batch, objectif et executor runtime
|   |       |-- optim/            Construction optimiseur et logique d'optimisation
|   |       `-- tasks/            Tetes, contraintes, pertes, metriques, regularisation
|   |-- instrumentation/
|   |   |-- callbacks/            Dispatch callbacks, logging et hooks de tracking
|   |   |-- dashboards/           Adaptateurs TensorBoard et MLflow
|   |   `-- formatters/           Rendu et formatage de rapports
|   |-- orchestration/
|   |   |-- builder.py            Construction de l'orchestration de session
|   |   |-- policy/               Policies d'epoch et de phase
|   |   `-- runner/               Runners continuous et curriculum
|   |-- factory.py                Factory de session
|   `-- metadata.py               Metadonnees de session
|
|-- study/
|   |-- analysis/                 Helpers d'analyse de trials/resultats
|   `-- sweep/                    Objectif Optuna, config et optimisation
|
|-- utils/
|   |-- logger.py                 Configuration du logging
|   `-- multip.py                 Helpers de multiprocessing
```

### Notes Sur Les Frontieres Actuelles

- `adapters/` reste volontairement mince: il traduit les entrees API/CLI en
  appels d'execution configures.
- `execution/` possede le dispatch des pipelines nommes; le travail specifique
  est delegue a `geopipe/`, `session/`, `models/`, `artifacts/` et `study/`.
- `configs/hydra/` contient la composition YAML runtime, tandis que
  `configs/schema/` contient les contrats Python structures.
- `geopipe/` possede la preparation geospatiale jusqu'aux `DataSpecs`; le
  chargement des donnees pour l'entrainement commence dans `session/data/`.
- `models/` possede uniquement la construction des reseaux. Les objectifs,
  metriques, optimiseurs et taches runtime vivent dans `session/engine/runtime/`.
- `session/` est la couche runtime principale: l'orchestration choisit phases et
  runners, les policies d'epoch definissent train/eval, et les taches runtime
  calculent tetes, pertes, metriques, contraintes et regularisation.
- `geopipe/` est organisé en trois étapes (`harmonize`, `ingest`, `prepare`),
  offrant des triades symétriques (`context.py`, `logger.py`, `pipeline.py`),
  régies par les contrats centraux dans `contracts/`.
