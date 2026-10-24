## Structure actuelle du projet

Dernière mise à jour: 2026-10-24

Ce document resume l'organisation actuelle du depot et la responsabilite
principale de chaque zone. Il privilegie les frontieres utiles pour ajouter ou
deplacer du code, plutot qu'une liste exhaustive de tous les fichiers.

```text
./
|-- docs/                         Documentation, diagrammes et ADRs
|   |-- ADRs/                     Decisions d'architecture
|   |-- images/                   Images de documentation
|   |-- data_preparation_fr.md    Guide de preparation des donnees
|   |-- preflight_readiness_fr.md Guide de validation preliminaire et diagnostics
|   |-- workflow_chart_fr.md      Diagramme general du workflow
|   `-- project_structure_fr.md   Ce fichier
|
|-- experiment/                   I/O locale ignoree des experiences
|   |-- artifacts/                Grilles, manifestes, checkpoints, etc. generes
|   |-- input/                    Entrees locales d'experience
|   |-- preflight/                Rapports de validation préliminaire exportés
|   `-- results/                  Sorties locales de pipelines/sessions
|
|-- knowledge/                    Connaissances écologiques du domaine et plongements
|   |-- ontario_tree_species_profiles.csv
|   |-- ontario_tree_species_grouped_profiles.csv
|   |-- embeddings/               Tenseurs PyTorch et matrices de similarité précalculés
|   `-- README.md
|
|-- notebooks/
|   |-- 00_raster_mosaic.ipynb    Démo d'assemblage de mosaïque raster virtuelle
|   |-- 01_data_preparation.ipynb Démo d'ingestion, validation et préparation
|   |-- 02_model_train.ipynb      Démo d'entraînement de modèle de bout en bout
|   `-- 03_study_sweep.ipynb      Démo de sweep hyperparamétrique Optuna
|
|-- src/landseg/                  Package Python installable
|   |-- adapters/                 Surfaces d'entree externes
|   |-- artifacts/                Chemins, politiques, payloads et checkpoints
|   |-- configs/                  Configs Hydra YAML et schemas structures
|   |-- core/                     Contrats transversaux et types de resultats
|   |-- execution/                Registre de pipelines et dispatch d'execution
|   |-- geopipe/                  Sous-système rasters, grille, harmonisation, ingestion, préparation
|   |-- knowledge/                Validation de taxonomie, plongements d'espèces, matrices de similarité
|   |-- models/                   Frames, backbones, tetes, conditionnement, factories
|   |-- session/                  Construction, orchestration et moteurs runtime
|   |-- study/                    Sweeps et analyse post-execution
|   |-- utils/                    Helpers partages de logging et multiprocessing
|   |-- py.typed                  Marqueur d'annotations de type PEP 561
|   |-- _constants.py             Constantes partagees
|   `-- __init__.py
|
|-- configs/                      Recettes modulaires par tâche
|   |-- data_harmonize.yaml       Recette d'harmonisation
|   |-- data_ingest.yaml          Recette d'ingestion
|   |-- data_prepare.yaml         Recette de préparation
|   |-- diagnose_overfit.yaml     Recette de diagnostic de surapprentissage
|   |-- end_to_end/               Recettes de workflows composites
|   |-- model_evaluate.yaml       Recette d'évaluation
|   |-- model_train.yaml          Recette d'entraînement
|   |-- study_analysis.yaml       Recette d'analyse d'étude
|   |-- study_sweep.yaml          Recette de sweep d'étude
|   `-- world_grid.yaml           Recette de grille monde
|
|-- scripts/                      Helper scripts et entry points
|   |-- build_raster_mosaic.py    Utilitaire CLI de mosaïque raster virtuelle multi-tuiles / cross-UTM
|   |-- build_species_embeddings.py Script CLI de génération de plongements d'espèces
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
|       `-- resolver.py           Helpers de resolution de config CLI
|
|-- artifacts/
|   |-- checkpoint.py             Sauvegarde/chargement de checkpoints
|   |-- controller.py             Resolve/build/rebuild selon les politiques
|   |-- ledger.py                 Calcul d'empreintes et detection de collisions
|   |-- paths/                    Chemins canoniques des artefacts
|   |-- payload_io.py             Persistance des payloads et metadonnees
|   `-- policy.py                 Regles de cycle de vie des artefacts
|
|-- configs/
|   |-- hydra/
|   |   |-- config.yaml           Point d'entree de composition Hydra
|   |   |-- command/              Configs harmonize, ingest, prepare, train, eval et study
|   |   |-- data/                 Defaults d'harmonisation, d'ingestion et de préparation
|   |   |-- models/               Defaults des architectures modeles
|   |   |-- session/              Configs dataloader, moteur, taches et orchestration
|   |   `-- study/                Defaults d'etude et de sweep
|   `-- schema/
|       |-- root.py               Schema structure racine
|       |-- utils.py              Utilitaires de schema
|       `-- sections/             Dataclasses par section: commands, session, models, data, etc.
|
|-- core/
|   |-- data_specs.py             Contrat runtime de specification des donnees
|   |-- model_protocol.py         Protocole de comportement des modeles
|   `-- session_results.py        Sorties structurees des sessions
|
|-- execution/
|   |-- executor.py               Point d'entree unifie de dispatch des commandes
|   |-- pipelines/                Pipelines d'execution atomiques par etape
|   |   |-- base.py               Gabarit ExecutionPipeline et execution de cycle de vie
|   |   |-- world_grid.py         Pipeline de génération de grille monde
|   |   |-- data_harmonize.py     Pipeline d'harmonisation des données
|   |   |-- data_ingest.py        Pipeline d'ingestion des données
|   |   |-- data_prepare.py       Pipeline de préparation des données
|   |   |-- model_train.py        Pipeline d'entrainement
|   |   `-- model_evaluate.py     Pipeline d'evaluation
|   |-- preflight/                Moteur de sondes et répartiteur d'inspection préliminaire
|   |   |-- probes/               Sondes diagnostiques (matériel, stockage, spatial, registre, etc.)
|   |   |-- engine.py             Moteur d'inspection et répartiteur de cibles
|   |   |-- prerequisites.py      Vérification de lignage et prérequis amont
|   |   |-- reporter.py           Formatage du tableau de bord terminal et export de rapport
|   |   `-- schema.py             Modèles de données des résultats de sondes et préparation
|   `-- workflows/                Workflows d'orchestration composites multi-etapes
|       |-- batch_ingest.py       Ingestion par lots vers le pool de blocs
|       |-- diagnose_overfit.py   Workflow de diagnostic de surapprentissage
|       |-- e2e_intake.py         Workflow d'ingestion continue (harmonisation + ingestion)
|       |-- e2e_experiment.py     Workflow d'experimentation complete (grille -> modele)
|       |-- study_sweep.py        Workflow d'orchestration de sweep Optuna
|       `-- study_analysis.py     Workflow d'analyse des resultats d'etude
|
|-- geopipe/
|   |-- contracts/                Schémas centraux de rapports et de transfert
|   |-- core/                     Abstractions clés, catalogues, coords et grille monde
|   |-- grid/                     Construction, cycle de vie et logger de grille
|   |-- harmonize/                Reprojection, composition du processeur et manifeste
|   |   |-- manifest/             Schéma de manifeste, normaliseur, compilateur
|   |   |-- processor/            Routage de composition, lecteurs de features/labels, processeur
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
|   |-- rasters/                  Sous-système centralisé d'opérations et d'E/S rasters
|   |   |-- _types.py             Types Window, DatasetInfo, Affine et BoundingBox
|   |   |-- geometry.py           Vérification de SCR, résolution et limites de chevauchement
|   |   |-- io.py                 read_raster, read_rasters, read_window, get_info
|   |   |-- mask.py               Génération de masques de pixels valides
|   |   |-- mosaic.py             Assemblage de mosaïques rasters virtuelles
|   |   |-- stack.py              Empilement multi-canaux de rasters
|   |   |-- vrt.py                Écriture de rasters virtuels et analyse de balises VRT
|   |   `-- warp.py               Reprojection, distorsion et transformations de limites
|   `-- factory.py                Factory des DataSpecs
|
|-- knowledge/
|   |-- embeddings.py             Tenseurs de plongements Sentence Transformer et matrices de similarité
|   |-- resolver.py               Résolution de répertoire et découverte taxonomique
|   `-- taxonomy.py               Schéma taxonomique et validation des spécifications d'espèces
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
|   |-- contracts/                Protocoles partages de donnees, observateur et phases
|   |-- data/                     Dataset, collation, memoire et construction des dataloaders
|   |-- engine/
|   |   |-- builder.py            Composition du moteur et construction de l'epoch runner
|   |   |-- batch/                Etat par batch, objectifs et execution
|   |   |-- epoch/                Epoch runner, builder et policies train/evaluation
|   |   |-- optim/                Construction de l'optimiseur et du scheduler
|   |   `-- tasks/                Tetes, contraintes, pertes, metriques et regularisation
|   |-- instrumentation/
|   |   |-- builder.py            Construction du dispatcher de callbacks
|   |   |-- callbacks/            Dispatch callbacks, logging et hooks de tracking
|   |   |-- dashboards/           Adaptateurs TensorBoard et MLflow
|   |   `-- formatters/           Rendu et formatage de rapports
|   |-- orchestration/
|   |   |-- builder.py            Construction de l'orchestration de session
|   |   |-- policy/               Policies d'epoch et de phase
|   |   `-- runner/               Runners continuous et curriculum
|   |-- factory.py                Factory de session
|   `-- logger.py                 Logging propre a la session
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
- `geopipe/rasters/` constitue l'abstraction centralisée et exclusive pour toutes
  les opérations d'E/S rasters, la lecture de fenêtres, la validation de
  géométrie/SCR et la reprojection, éliminant les appels directs à rasterio.
- `knowledge/` fournit des ressources écologiques de domaine découplées, la
  validation taxonomique et les espaces de plongements, configurés via
  `execution.knowledge_dir`.
- `models/` possede uniquement la construction des reseaux. Les objectifs,
  metriques, optimiseurs et taches vivent dans `session/engine/`.
- `session/` est la couche runtime principale. `factory.py` selectionne un
  session runner; `data/` construit les dataloaders, `engine/` compose
  l'execution par batch et par epoch, et `orchestration/` choisit
  l'entrainement continu ou par curriculum.
- `session/contracts/` fournit les protocoles partages de donnees,
  d'observateur et de phase; les exports paresseux des packages sont reserves
  aux consommateurs publics, et non aux dependances internes entre modules.
- La configuration Hydra de session suit ces frontieres: `dataloader` et le
  groupe `engine` imbrique (execution, optimisation, planification et taches)
  sont distincts des parametres de monitoring, curriculum et reprise de
  `orchestration`.
- `geopipe/` est organisé en trois étapes (`harmonize`, `ingest`, `prepare`),
  offrant des triades symétriques (`context.py`, `logger.py`, `pipeline.py`),
  régies par les contrats centraux dans `contracts/`.
- Les recettes orientées tâches sous `configs/*.yaml` se résolvent directement
  dans les contrats Hydra structurés sans couche de traduction intermédiaire.
