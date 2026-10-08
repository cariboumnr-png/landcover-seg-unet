## Workflow actuel

Dernière mise à jour : 2026-10-07

```
[grid/builder]                           (1 Grille globale – définition spatiale canonique)
|
+--> [artifacts/controller]
|        (résolution / build / réutilisation selon politique)
|
+--> [grid/lifecycle]
|        (persistance et validation de l’artefact grille)
|
+--> [geopipe/harmonize]                 (2 Harmonisation – rééchantillonnage des rasters)
|        |
|        +--> [artifacts/controller]
|        |        (résolution / build / réutilisation du lot harmonisé)
|        |
|        +--> [geopipe/ledger]
|                 (enregistrement du lot dans harmonization_runs.json)
|
+--> [geopipe/ingest]                    (3 Ingestion – pool canonique de blocs)
|        |
|        +--> [ingest/blocks/assembler]
|        |        (découpage des tuiles et features de domaine)
|        |
|        +--> [ingest/collision]
|        |        (évaluation de la politique : skip vs overwrite)
|        |
|        +--> [artifacts/controller]
|        |        (persistance des blocs dans le catalogue du pool)
|        |
|        +--> [geopipe/ledger]
|                 (enregistrement du lot dans ingestion_runs.json)
|
+--> [geopipe/prepare]                   (4 Préparation – jeux de données d'expérience)
|        |
|        +--> [prepare/partitioner]
|        |        (partitionnement géographique AOI train/val/test)
|        |
|        +--> [artifacts/controller]
|        |        (persistance des manifestes, stats et DataSpecs)
|
+--> [models/factory]                    (5 Construction et assemblage du modèle)
|
+--> [session/factory]                   (6 Frontière de construction de session)
|        |
|        +--> [session/data]             (dataloaders et adaptateurs de batch)
|        |
|        +--> [session/engine]           (moteurs d’exécution batch + epoch)
|        |
|        +--> [session/instrumentation]  (callbacks, logging, suivi, tableaux de bord)
|        |
|        +--> [session/orchestration]    (gestion du cycle de vie et transitions)
|
+--> [execution/preflight]               (7 Moteur de validation avant vol / preflight)
|        |                               (inspection diagnostique non destructive)
|        +--> [probes/lineage]           (prérequis amont et intégrité des registres)
|        +--> [probes/filesystem]        (droits d'écriture et écrasement d'artefacts)
|        +--> [probes/domain]            (grille spatiale, manifestes bruts, modèle)
|        +--> [probes/hardware]          (détection CUDA/CPU et mémoire VRAM libre)
|        `--> [reporter]                 (tableau de bord 120-col et export JSON)
|
+--> [execution/executor]                (8 Dispatch d'exécution : command=<nom>)
         |
         +--> [execution/pipelines]      (8a Pipelines atomiques – invariant 1:1)
         |        |
         |        +--> [WorldGridGeneration]   (construction de la grille globale)
         |        +--> [DataHarmonization]     (harmonisation d'un lot raster)
         |        +--> [DataIngestion]         (ingestion d'un lot dans le pool)
         |        +--> [DataPreparation]       (partitionnement en DataSpecs)
         |        +--> [ModelTraining]         (session d'entraînement complète)
         |        +--> [ModelEvaluation]       (évaluation autonome de checkpoint)
         |
         `--> [execution/workflows]      (8b Workflows composites multi-runs)
                  |
                  +--> [batch_ingest]          (boucle d'ingestion des lots en attente)
                  +--> [diagnose_overfit]      (diagnostic de surapprentissage)
                  +--> [study_sweep]           (essais d'optimisation Optuna)
                  +--> [study_analysis]        (rapports d'étude et métriques)
                  +--> [default]               (audit d'aptitude avant vol)
```
---

### Notes d'interprétation (mises à jour)

- Toutes les étapes de construction de fondation (grille, harmonisation,
  ingestion, préparation) restent pures, déterministes et sans effets de bord.

- Toutes les décisions de réutilisation, reconstruction, écrasement et
  validation d'artefacts sont centralisées par `artifacts.controller`.

- Les registres de runs (`harmonization_runs.json`, `ingestion_runs.json`)
  tracent l'historique des lots, les empreintes SHA-256 et les politiques de
  collision.

- Les étapes aval opèrent sur des artefacts résolus, jamais sur des intermédiaires
  implicites ou recalculés.

- La construction des `DataSpecs` et des modèles intervient strictement avant la
  frontière de session et produit des objets immutables.

- La construction de la session est centralisée dans `session/factory` et possède :
  - la construction de l'interface de données (dataloaders, échantillonneurs)
  - l'assemblage des composants (liaisons de modèles, pertes, optimiseurs)
  - l'initialisation de l'état runtime
  - la liaison des callbacks et de l'instrumentation
  - l'instanciation des moteurs d'exécution
  - la configuration de l'orchestration du cycle de vie

- L'orchestration du cycle de vie (phases et transitions) est gérée par
  la session, non par les pipelines.

- La validation avant vol (`execution.preflight`) offre un moteur d'inspection
  non destructif capable de valider les prérequis, l'écriture du stockage, les
  contrats de domaine, les registres et le matériel avant tout traitement lourd.

- La couche d'exécution applique une séparation nette entre :
  - **Pipelines atomiques** (`execution.pipelines`) : classes de runner dédiées
    (`WorldGridGeneration`, `DataHarmonization`, `DataIngestion`,
    `DataPreparation`, `ModelTraining`, `ModelEvaluation`) appliquant
    l'invariant 1:1 strict (1 invocation $\rightarrow$ 1 cible $\rightarrow$
    1 répertoire de run $\rightarrow$ 1 rapport).
  - **Workflows composites** (`execution.workflows`) : procédures fonctionnelles
    légères coordonnant des boucles multi-runs (`batch_ingest`), des sweeps
    d'optimisation (`study_sweep`), des diagnostics (`diagnose_overfit`) ou des
    analyses (`study_analysis`).

- Le système maintient un flux unidirectionnel de dépendance :
  `geopipe (ETL) → artifacts → models → session → execution (preflight, pipelines, workflows)`

- Cette structure garantit :
  - la reproductibilité via le suivi explicite des artefacts et registres
  - la sécurité par détection précoce des erreurs (validation avant vol)
  - la séparation stricte des préoccupations build-time et runtime
  - des pipelines modulaires, composables et prévisibles
  - une reconstruction déterministe depuis les configurations et artefacts