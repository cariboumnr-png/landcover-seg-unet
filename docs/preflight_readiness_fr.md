# Guide de validation préliminaire et de diagnostic d'exécution

[English](./preflight_readiness.md) | [Français](./preflight_readiness_fr.md)

Dernière mise à jour : 2026-10-07

---

## Vue d'ensemble

Les pipelines d'apprentissage profond géospatial requièrent des ressources
de calcul considérables : le rééchantillonnage de rasters volumineux, le découpage
de tuiles multi-canaux et l'exécution de réseaux de neurones sur GPU sont des
opérations coûteuses en temps et en mémoire. Lorsque des pipelines rencontrent
des prérequis manquants, des discordances de système de coordonnées (SCR), un
stockage inaccessible en écriture ou des poids manquants en plein milieu d'une
exécution, le calcul est gaspillé et les répertoires partiellement matérialisés
peuvent être laissés dans un état incohérent.

Le **moteur de validation préliminaire** (`landseg.execution.preflight`) fournit
une couche d'audit diagnostique unifiée et non destructive. Il inspecte les
prérequis des pipelines et des workflows en mode simulation (dry-run) avant
l'exécution, affichant des tableaux de bord ASCII normalisés de 120 colonnes dans
le terminal et générant des artefacts de rapport JSON structurés.

---

## Sommaire

- [Invocations en ligne de commande et par API](#invocations-en-ligne-de-commande-et-par-api)
- [Ordre canonique des sondes](#ordre-canonique-des-sondes)
- [Évaluation du statut des sondes](#évaluation-du-statut-des-sondes)
- [Répertoire d'inspection par cible](#répertoire-dinspection-par-cible)
  - [1. world-grid](#1-world-grid)
  - [2. data-harmonize](#2-data-harmonize)
  - [3. data-ingest](#3-data-ingest)
  - [4. batch-ingest](#4-batch-ingest)
  - [5. data-prepare](#5-data-prepare)
  - [6. model-train](#6-model-train)
  - [7. model-evaluate](#7-model-evaluate)
  - [8. diagnose-overfit](#8-diagnose-overfit)
  - [9. all (Audit global du système)](#9-all-audit-global-du-système)
- [Artefacts de rapport de validation](#artefacts-de-rapport-de-validation)

---

## Invocations en ligne de commande et par API

### Interface en ligne de commande (CLI)

Les vérifications préalables sont exécutées via `command=preflight` avec Hydra :

```bash
# Exécuter l'audit de préparation global sur les 8 cibles prises en charge
python scripts/run.py command=preflight

# Inspecter une cible spécifique de pipeline ou de workflow
python scripts/run.py command=preflight command.preflight.target=model-train
python scripts/run.py command=preflight command.preflight.target=batch-ingest

# Activer le mode strict (les avertissements bloquent l'exécution)
python scripts/run.py command=preflight command.preflight.target=data-harmonize command.preflight.strict=true

# Désactiver l'export du rapport JSON (affichage console uniquement)
python scripts/run.py command=preflight command.preflight.export_report=false
```

### API Python programmatique

L'inspection préalable peut également être intégrée dans des notebooks ou des scripts :

```python
import landseg

# Composer ou charger RootConfig
config = landseg.load_config()

# Lancer l'inspection préliminaire
result = landseg.run_preflight(config, target='model-train')

# Évaluer la préparation
if not result.is_ready:
    print(f'Exécution bloquée ! Erreurs : {result.errors}')
```

---

## Ordre canonique des sondes

Les sondes diagnostiques s'exécutent selon une séquence prévisible et strictement définie :

$$\text{Lineage} \longrightarrow \text{Filesystem} \longrightarrow \text{Domain Contracts} \longrightarrow \text{Ledger} \longrightarrow \text{Hardware}$$

1. **`Lineage` (Lignage)** : Dépendances amont des pipelines, rapports terminés et
   artefacts requis sur le disque.
2. **`Filesystem` (Système de fichiers)** : Droits d'écriture du répertoire de
   destination et statut d'écrasement ou reconstruction des fichiers cibles.
3. **`Domain Contracts` (Contrats de domaine)** :
   - **`Spatial`** : Rasters de référence, définitions de SCR, résolution de pixel,
     emprises et spécifications de tuiles.
   - **`Dataset`** : Présence du manifeste des données sources et décompte d'éléments.
   - **`Policy`** : Politiques de résolution de collision de blocs (`skip` vs `overwrite`).
   - **`Model`** : Reconnaissance de l'architecture dans le registre, points de contrôle
     et découpages d'évaluation.
4. **`Ledger` (Registre)** : Manifestes d'historique (`harmonization_runs.json`,
   `ingestion_runs.json`), lots en attente d'ingestion et décompte du pool canonique.
5. **`Hardware` (Matériel)** : Détection de l'accélérateur de calcul (nom de périphérique
   CUDA) et estimation de la réserve VRAM face aux dimensions de tenseurs par lot.

---

## Évaluation du statut des sondes

Chaque sonde est évaluée selon l'un des quatre états canoniques :

| Statut | Signification | Impact sur la cible |
| :--- | :--- | :--- |
| `PASS` | Condition vérifiée et parfaitement saine. | La cible demeure `READY`. |
| `WARN` | Avis consultatif (ex. repli sur CPU, blocs déjà existants, rien en attente). | La cible demeure `READY` en mode standard ; échoue en mode `strict=true`. |
| `FAIL` | Condition bloquante (ex. rapport manquant, chemin protégé, checkpoint absent). | La cible passe au statut `BLOCKED`. |
| `SKIP` | Sonde délibérément omise ou non applicable. | Neutre. |

Une cible d'exécution est considérée comme **`READY`** uniquement lorsque toutes
ses sondes affichent `PASS` ou `WARN` (en mode non strict). Si une seule sonde
affiche `FAIL`, la cible est marquée **`BLOCKED`**.

---

## Répertoire d'inspection par cible

### 1. `world-grid`

Valide les paramètres de tuilage spatial et les sources raster de référence :

```text
========================================================================================================================
                                         PRE-FLIGHT READINESS CHECK: world-grid
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Filesystem     world_grid_output             PASS     Target directory is writable
 Filesystem     world_grid_report             PASS     Target file already exists; force_rebuild: False
 Spatial        world_grid_reference          PASS     Reference raster found at: data/reference/ref.tif
 Spatial        crs                           PASS     Target CRS is defined by the reference raster
 Spatial        pixel_size                    PASS     Pixel size is defined by the reference raster
 Spatial        extent                        PASS     Extent is defined by the reference raster
 Spatial        origin                        PASS     Origin is defined by the reference raster
 Spatial        grid_specs                    PASS     Grid specifications valid
========================================================================================================================
 STATUS: READY (0 errors, 0 warnings)
========================================================================================================================
```

### 2. `data-harmonize`

Valide la grille monde, le manifeste des rasters bruts et les enregistrements passés :

```text
========================================================================================================================
                                       PRE-FLIGHT READINESS CHECK: data-harmonize
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Upstream "world-grid" report verified.
 Filesystem     harmonization_output          PASS     Target directory is writable
 Dataset        source_dataset_manifest       PASS     Found 8 rasters available for harmonization at: input/raw
 Ledger         past_harmonization_runs       PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         harmonized_dataset            PASS     Dataset not yet harmonized
========================================================================================================================
 STATUS: READY (0 errors, 0 warnings)
========================================================================================================================
```

### 3. `data-ingest`

Valide l'ingestion d'un lot unitaire, la politique de collision et l'état du registre :

```text
========================================================================================================================
                                         PRE-FLIGHT READINESS CHECK: data-ingest
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Upstream pipeline "data-harmonize" manifest verified (1 succes...
 Filesystem     ingestion_output              PASS     Target directory is writable
 Policy         collision_policy              PASS     Collision policy 'skip' configured
 Ledger         past_harmonization_runs       PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         past_ingestion_runs           PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         pending_ingestion             WARN     Harmonization ledger up to date; nothing to ingest
 Ledger         ingested_blocks_pool          PASS     Ingestion pool contains 15 existing blocks
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

### 4. `batch-ingest`

Valide la file d'ingestion multi-lots, la résolution de collisions et le pool de blocs :

```text
========================================================================================================================
                                        PRE-FLIGHT READINESS CHECK: batch-ingest
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Filesystem     ingestion_output              PASS     Target directory is writable
 Policy         collision_policy              PASS     Collision policy 'skip' configured
 Ledger         past_harmonization_runs       PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         past_ingestion_runs           PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         pending_ingestion             WARN     Harmonization ledger up to date; nothing to ingest
 Ledger         ingested_blocks_pool          PASS     Ingestion pool contains 15 existing blocks
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

### 5. `data-prepare`

Valide la disponibilité du pool canonique et les blocs préparés pour chaque partition :

```text
========================================================================================================================
                                        PRE-FLIGHT READINESS CHECK: data-prepare
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Required artifacts verified for "data-ingest".
 Lineage        pipeline_prerequisites        PASS     Upstream pipeline "data-ingest" manifest verified (1 successfu...
 Filesystem     preparation_output            PASS     Target directory is writable
 Filesystem     preparation_report            PASS     Target file already exists; force_rebuild: False
 Ledger         ingested_blocks_pool          PASS     Ingestion pool contains 15 existing blocks
 Ledger         prepared_blocks_state         WARN     Found existing prepared blocks; train: 3 | val: 1 | test: 1
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

### 6. `model-train`

Valide la préparation des données, l'architecture modèle, l'état des lots et le GPU :

```text
========================================================================================================================
                                        PRE-FLIGHT READINESS CHECK: model-train
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Upstream "data-prepare" prerequisites verified.
 Filesystem     checkpoint_dir                PASS     Target directory is writable
 Model          model_body                    PASS     Configured architecture: "unetppp" recognized in registry
 Ledger         pending_ingestion             PASS     Harmonization ledger up to date; nothing to ingest
 Ledger         prepared_blocks_state         PASS     Found 3 train / 1 val / 1 test prepared blocks ready
 Hardware       cuda_device                   WARN     CUDA unavailable; compute running on CPU
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

### 7. `model-evaluate`

Valide le fichier de points de contrôle (checkpoint), l'architecture et la partition d'évaluation :

```text
========================================================================================================================
                                       PRE-FLIGHT READINESS CHECK: model-evaluate
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Upstream "data-prepare" prerequisites verified.
 Filesystem     eval_output                   PASS     Target directory is writable
 Model          checkpoint_exists             FAIL     Checkpoint file not found: None
 Model          model_body                    PASS     Configured architecture: "unetppp" recognized in registry
 Model          eval_split_configured         PASS     Evaluation target split 'test' configured
 Hardware       cuda_device                   WARN     CUDA unavailable; compute running on CPU
========================================================================================================================
 STATUS: BLOCKED (1 errors, 1 warnings)
========================================================================================================================
```

### 8. `diagnose-overfit`

Valide le calcul direct/rétrograde rapide et les ressources matérielles :

```text
========================================================================================================================
                                      PRE-FLIGHT READINESS CHECK: diagnose-overfit
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Model          model_body                    PASS     Configured architecture: "unetppp" recognized in registry
 Hardware       cuda_device                   WARN     CUDA unavailable; compute running on CPU
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

### 9. `all` (Audit global du système)

Lorsque `target=all` est appelé, chaque tableau de bord est rendu séquentiellement,
suivi d'une bannière de synthèse globale du système et du chemin d'export du rapport :

```text
========================================================================================================================
                                         PRE-FLIGHT READINESS CHECK: world-grid
... (rapports individuels pour les 8 cibles prises en charge) ...
========================================================================================================================
                                      PRE-FLIGHT READINESS CHECK: diagnose-overfit
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Model          model_body                    PASS     Configured architecture: "unetppp" recognized in registry
 Hardware       cuda_device                   WARN     CUDA unavailable; compute running on CPU
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
 SYSTEM STATUS: 7 READY | 1 BLOCKED
========================================================================================================================
Preflight report saved to: ./experiment/preflight/preflight_report_20261007_200100_a88673.json
```

---

## Artefacts de rapport de validation

Lorsque l'export de rapport est activé (`command.preflight.export_report=true`, par défaut),
les rapports sont enregistrés sous :

```text
<exp_root>/preflight/preflight_report_<timestamp>_<hex>.json
```

### Structure du schéma

```json
{
  "timestamp": "2026-10-07T20:01:00Z",
  "uid": "20261007_200100_a88673",
  "target": "all",
  "status": "BLOCKED",
  "strict": false,
  "is_ready": false,
  "summary": {
    "total_probes": 28,
    "pass": 24,
    "warn": 3,
    "fail": 1,
    "skip": 0
  },
  "targets": [
    {
      "target": "model-train",
      "status": "READY",
      "is_ready": true,
      "probes": [
        {
          "probe_id": "pipeline_prerequisites",
          "category": "Lineage",
          "status": "PASS",
          "message": "Upstream \"data-prepare\" prerequisites verified.",
          "details": {}
        },
        {
          "probe_id": "checkpoint_dir",
          "category": "Filesystem",
          "status": "PASS",
          "message": "Target directory is writable",
          "details": {"target_dir": "experiment/results"}
        },
        {
          "probe_id": "model_body",
          "category": "Model",
          "status": "PASS",
          "message": "Configured architecture: \"unetppp\" recognized in registry",
          "details": {"model_body": "unetppp"}
        },
        {
          "probe_id": "pending_ingestion",
          "category": "Ledger",
          "status": "PASS",
          "message": "Harmonization ledger up to date; nothing to ingest",
          "details": {}
        },
        {
          "probe_id": "prepared_blocks_state",
          "category": "Ledger",
          "status": "PASS",
          "message": "Found 3 train / 1 val / 1 test prepared blocks ready",
          "details": {"train_blocks": 3, "val_blocks": 1, "test_blocks": 1}
        },
        {
          "probe_id": "cuda_device",
          "category": "Hardware",
          "status": "WARN",
          "message": "CUDA unavailable; compute running on CPU",
          "details": {"device_name": "cpu", "cuda": false}
        }
      ],
      "errors": [],
      "warnings": [
        "CUDA unavailable; compute running on CPU"
      ],
      "telemetry": {}
    }
  ]
}
```
