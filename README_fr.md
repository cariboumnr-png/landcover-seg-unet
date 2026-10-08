# Cadre de classification multimodale de l'occupation du sol

[English](README.md) | [Francais](README_fr.md)

> Resume en langage clair:
> Ce projet prepare des donnees raster geospatiales et entraine des modeles
> d'apprentissage profond pour la classification de l'occupation du sol au
> niveau du pixel. Il organise les donnees en artefacts reutilisables, construit
> des modeles de segmentation a partir de la configuration, et execute des
> workflows reproductibles d'entrainement, d'evaluation et d'etude.

`landseg` est un cadre modulaire, oriente artefacts, pour la segmentation de
l'occupation du sol. Il combine l'imagerie satellite, des entrees
topographiques optionnelles et des caracteristiques de domaine optionnelles au
moyen d'un pipeline geospatial deterministe et d'un runtime PyTorch base sur
des sessions.

La pile de modeles actuelle est centree sur des modeles de segmentation
configurables de type U-Net, y compris des corps U-Net, U-Net++ et U-Net+++.
Le runtime prend en charge les sorties multi-tetes, les pertes configurables,
les metriques de segmentation, le cablage des optimiseurs, les callbacks et les
adaptateurs de tableaux de bord. La configuration est separee entre des
parametres racine destines a l'utilisateur et l'arborescence Hydra/schema
structuree fournie avec le package.

## Etat Du Projet

Ce depot est en developpement actif de recherche et d'experimentation. Le
workflow principal de preparation des donnees et d'entrainement des modeles est
utilisable, mais les frontieres de modules, les surfaces de configuration et
les API avancees d'etude peuvent encore evoluer.

Actuellement utilisable:

- Ingestion des donnees et preparation propre a l'experience
- Construction de grilles, domaines, blocs de donnees, manifestes et datasets a partir d'artefacts
- Commandes d'entrainement et d'evaluation autonome des modeles
- Diagnostics de surapprentissage pour valider la chaine de bout en bout
- Moteur de validation d'aptitude avant vol
- Chemins de code pour les adaptateurs TensorBoard et MLflow
- Points d'entree de commande pour le sweep d'etude et l'analyse d'etude orientes Optuna

Encore en maturation:

- Workflows et exemples centres sur les notebooks
- Ergonomie de l'API programmatique publique
- Garanties de configuration pour les etudes/sweeps
- Exports d'evaluation et schemas de rapports standardises
- Garanties de compatibilite a long terme pour les champs de configuration internes

## Documentation

- [Structure du depot](./docs/project_structure_fr.md)
- [Schema du workflow](./docs/workflow_chart_fr.md)
- [Guide de preparation des donnees](./docs/data_preparation_fr.md)
- [Guide de preparation avant vol](./docs/preflight_readiness_fr.md)
- [Decisions d'architecture](./docs/ADRs/)

## Concepts Cles

### Artefacts De Fondation

Les rasters bruts sont transformes en artefacts reutilisables alignes sur une
grille, par exemple des grilles monde, cartes de domaine, blocs de donnees,
manifestes et schemas. Ces artefacts font le lien entre les formats raster
geospatiaux et les entrees d'entrainement orientees tenseurs.

### DataSpecs

Les artefacts prepares sont assembles en `DataSpecs`, qui decrivent les entrees
du modele, les partitions du dataset, la normalisation, la structure des
classes et les autres contrats de donnees utilises par le runtime.

### Modeles

Les modeles sont construits depuis la configuration via `landseg.models`. La
couche modele possede la construction des reseaux neuronaux: backbones, frames,
tetes, helpers de domaine, conditionnement et validation de surete. Les
objectifs d'entrainement et les metriques restent dans le runtime de session,
plutot que dans les definitions de modeles.

### Sessions

Les sessions assemblent la surface runtime pour l'entrainement ou l'evaluation:

- datasets et dataloaders
- liaisons des modeles
- tetes, pertes, metriques, contraintes et taches de regularisation
- optimiseurs
- executors d'epoch et de runtime
- callbacks, tracking, tableaux de bord et formatage de rapports
- politiques d'orchestration et runners

### Couche D'Execution

La couche d'execution dispatche une commande nommee (pipelines d'execution
atomiques ou workflows composites), resout la configuration, coordonne la
resolution des artefacts et delegue le travail aux factories et runners de
session. Les implementations restent volontairement minces.

## Installation

Python 3.12 ou plus recent est requis.

```bash
pip install .
```

Cela installe la commande console `landseg`:

```bash
landseg command=default
```

Pour executer dans des environnements distants (tels que des noeuds de calcul
Databricks ou des machines virtuelles) sans installer le package, vous pouvez
utiliser le script de demarrage :

```bash
python scripts/run.py command=default
```

## Configuration

La plupart des workflows utilisateur devraient commencer avec le fichier
`configs/user.yaml` sous le repertoire `configs/` a la racine. L'arborescence
Hydra fournie sous `src/landseg/configs/hydra/` contient les defaults de
composition internes et doit etre modifiee avec prudence.

Les couches de configuration sont:

- `configs/user.yaml`: entrees locales de donnees et choix de haut niveau
- `src/landseg/configs/hydra/`: defaults de composition Hydra du package
- `src/landseg/configs/schema/`: contrats de configuration Python structures
- Surcharges de developpement: resolues depuis le chemin defini dans
  `execution.dev_cfg` (generalement via la variable d'environnement
  `AUX_SETTINGS_PATH`)

Avant d'executer les pipelines de donnees, lisez le
[guide de preparation des donnees](./docs/data_preparation_fr.md) et organisez
les entrees locales sous la racine d'experience configuree.

## Utilisation Des Commandes

Les noms de commandes sont enregistres dans `landseg.execution.executor`.

### 0. Génération De La Grille Monde

Construit et persiste l'artefact de grille monde canonique pour le tuilage
spatial à partir d'un raster de référence ou de paramètres d'étendue explicites.

```bash
landseg command=world-grid
```

### 1. Harmonisation Des Données

Harmonise, reprojette et rééchantillonne les rasters bruts (features, labels
et masques de domaine) sur le canevas de la grille monde.

```bash
landseg command=data-harmonize
```

### 2. Ingestion Des Donnees

Construit les blocs de données canoniques non partitionnés à partir des
rasters harmonisés et de la grille monde.

```bash
landseg command=data-ingest
```

### 3. Ingestion Par Lots

Ingère séquentiellement plusieurs lots d'harmonisation planifiés dans le pool
canonique de blocs de données.

```bash
landseg command=batch-ingest
```

### 4. Preparation Des Donnees

Construit les artefacts propres a l'experience a partir des blocs de donnees
ingeres, y compris le partitionnement géographique par AOI, les splits, la
normalisation et les schemas.

```bash
landseg command=data-prepare
```

### 5. Entrainement Du Modele

Construit et execute une session complete d'entrainement a partir des artefacts
prepares.

```bash
landseg command=model-train
```

### 6. Evaluation Du Modele

Execute l'evaluation a partir des artefacts prepares et d'un checkpoint entraine.

```bash
landseg command=model-evaluate command.model_evaluate.checkpoint=path/to/checkpoint
```

### 7. Diagnostic De Surapprentissage

Execute un diagnostic contraint de bout en bout sur un petit perimetre pour
valider le cablage du modele, du dataset, des pertes, de l'optimiseur, des
metriques et de l'execution.

```bash
landseg command=diagnose-overfit
```

### 8. Sweep D'Etude

Execute le point d'entree de sweep oriente Optuna.

```bash
landseg command=study-sweep
```

### 9. Analyse D'Etude

Analyse les resultats d'etude via le point d'entree d'analyse.

```bash
landseg command=study-analysis
```

### 10. Validation Avant Vol (Pre-Flight)

Execute des audits non destructifs avant vol pour verifier les dependances,
l'integrite des registres, les contrats spatiaux et les ressources de calcul
avant de lancer des traitements lourds.

```bash
# Auditer l'environnement complet ou des cibles specifiques de pipeline
landseg command=preflight target=all
landseg command=preflight target=model-train strict=true
```

Pour les specifications detaillees des sondes, codes de statut et tableaux
de bord, consultez le [guide de preparation avant vol](./docs/preflight_readiness_fr.md).

## Organisation Des Artefacts Et Des Sorties

L'I/O locale des experiences est normalement placee sous le repertoire
d'experience configure. Dans l'arborescence de travail par defaut, cela
correspond a:

```text
experiment/
|-- input/       Entrees source locales
|-- artifacts/   Artefacts generes reutilisables
`-- results/     Sorties d'execution et de sessions
```

Les artefacts sont destines a servir de source de verite pour la
reproductibilite. Le framework resout, reutilise, reconstruit ou valide les
artefacts via un code centralise de politiques d'artefacts, plutot que de
demander aux utilisateurs de gerer manuellement les fichiers intermediaires.

## Frontieres Du Package

L'organisation actuelle du code source est:

```text
src/landseg/
|-- adapters/        Surfaces d'entree CLI et API programmatique
|-- artifacts/       Chemins, persistance, politiques, checkpoints
|-- configs/         Defaults Hydra YAML et schemas de config structures
|-- core/            Contrats partages et types de resultats
|-- execution/       Dispatch de commandes, pipelines, workflows et moteur avant-vol
|-- geopipe/         Pipeline geospatial de fondation et transformation
|-- models/          Frames, backbones, tetes, conditionnement, factories
|-- session/         Donnees runtime, moteurs, taches, instrumentation, orchestration
|-- study/           Utilitaires de sweep et d'analyse
`-- utils/           Helpers partages de logging et multiprocessing
```

Pour une carte plus complete, consultez
[docs/project_structure_fr.md](./docs/project_structure_fr.md).

## Suivi Et Instrumentation

Les evenements d'entrainement et d'evaluation sont emis via une instrumentation
basee sur des callbacks. Le code actuel inclut:

- dispatch de callbacks et callbacks de logging
- hooks de tracking pour entrainement, validation et inference
- adaptateur de tableau de bord TensorBoard
- adaptateur de tableau de bord MLflow
- helpers de rendu et formatage de rapports

Ces surfaces sont encore affineees, surtout pour la generation standardisee
d'apercus, les exports d'evaluation et les rapports de comparaison.

## Feuille De Route

Recemment complete ou stabilise :

- Orchestration d'execution et validation avant vol : pipelines atomiques,
  workflows composites (`e2e-intake`, `e2e-experiment`) et moteur preflight
  (`command=preflight`, ADR-0060).
- Ingestion incrementale par lots, registres de runs et politiques de
  collision (`skip` vs `overwrite`, ADR-0059).
- Semantique dynamique des donnees et preparation decouplee (`data-prepare`,
  ADR-0057).
- Surfaces d'API programmatiques et integration pour notebooks (ADR-0029).
- Regularisation par similarite ecologique (ADR-0053), tetes multiples et
  metriques d'evaluation etendues (ADR-0042).
- Prereglages de sweeps d'etude Optuna et metriques d'objectifs (ADR-0044).

Objectifs a court et moyen terme :

- Recettes de configuration modulaires sous `configs/recipes/` et simplification
  des surcharges CLI (ADR-0060 Section 6).
- Documentation des workflows de sweep Optuna et publication de tutoriels en
  notebooks.
- Stabilisation des formats de rapports de metriques et outils de comparaison
  inter-runs.

Objectifs a plus long terme :

- Runners de workflow a etat avec orchestration DAG, reprise sur incident et
  rejeu d'etapes (ADR-0060 Section 6.3).
- Architecture d'apprentissage continu et memoire tampon de rejeu (ADR-0052).
- Ajout de nouvelles familles de modeles au-dela de la pile U-Net actuelle
  (ex. transformeurs visuels).
- Chemins d'export stables pour la production (ONNX, TorchScript).
- Prise en charge de flux d'analyse inter-experiences plus riches.

## Contribution

Ce projet reste experimental. Les contributions devraient preserver la
separation actuelle entre preparation geospatiale, cycle de vie des artefacts,
construction des modeles, runtime de session et pipelines d'execution.

Avant les grands changements structurels, consultez les ADR dans
[docs/ADRs/](./docs/ADRs/) et ajoutez ou mettez a jour un ADR lorsqu'une
decision change la responsabilite des modules, les contrats runtime ou le
comportement visible par l'utilisateur.

## Licence

Distribue sous la licence Apache, Version 2.0. Consultez [LICENSE](./LICENSE)
et [NOTICE](./NOTICE) pour plus de details.

Copyright Sa Majeste le Roi du chef de l'Ontario, represente par le ministre
des Richesses naturelles, 2026.

Copyright Imprimeur du Roi pour l'Ontario, 2026.
