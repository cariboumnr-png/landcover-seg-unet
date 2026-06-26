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
- Pipelines d'entrainement et d'evaluation autonome des modeles
- Diagnostics de surapprentissage pour valider la chaine de bout en bout
- Chemins de code pour les adaptateurs TensorBoard et MLflow
- Points d'entree de sweep d'etude et d'analyse d'etude orientes Optuna

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

### Pipelines D'Execution

La couche d'execution selectionne un pipeline nomme, resout la configuration,
coordonne la resolution des artefacts et delegue le travail principal aux
factories et modules runtime. Les implementations de pipelines restent
volontairement minces.

## Installation

Python 3.12 ou plus recent est requis.

```bash
pip install .
```

Cela installe la commande console `landseg`:

```bash
landseg pipeline=default
```

Pour executer dans des environnements distants (tels que des noeuds de calcul
Databricks ou des machines virtuelles) sans installer le package, vous pouvez
utiliser le script de demarrage :

```bash
python scripts/run.py pipeline=default
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

## Utilisation Des Pipelines

Les noms de pipelines sont enregistres dans `landseg.execution.pipelines`.

### 1. Ingestion Des Donnees

Construit les artefacts de fondation a partir des rasters bruts. Cette etape
s'execute generalement une fois par jeu de donnees source, ou chaque fois que
les rasters source ou les parametres de grille changent.

```bash
landseg pipeline=data-ingest
```

### 2. Preparation Des Donnees

Construit les artefacts propres a l'experience a partir des blocs de donnees
ingeres, y compris les partitions, la normalisation/statistiques et les schemas
de dataset.

```bash
landseg pipeline=data-prepare
```

### 3. Entrainement Du Modele

Construit et execute une session complete d'entrainement a partir des artefacts
prepares.

```bash
landseg pipeline=model-train
```

### 4. Evaluation Du Modele

Execute l'evaluation a partir des artefacts prepares et d'un checkpoint entraine.

```bash
landseg pipeline=model-evaluate pipeline.model_evaluate.checkpoint=path/to/checkpoint
```

### 5. Diagnostic De Surapprentissage

Execute un diagnostic contraint de bout en bout sur un petit perimetre pour
valider le cablage du modele, du dataset, des pertes, de l'optimiseur, des
metriques et de l'execution.

```bash
landseg pipeline=diagnose-overfit
```

### 6. Sweep D'Etude

Execute le point d'entree de sweep oriente Optuna.

```bash
landseg pipeline=study-sweep
```

### 7. Analyse D'Etude

Analyse les resultats d'etude via le point d'entree d'analyse.

```bash
landseg pipeline=study-analysis
```

## Organisation Des Artefacts Et Des Sorties

L'I/O locale des experiences est normalement placee sous le repertoire
d'experience configure. Dans l'arborescence de travail par defaut, cela
correspond a:

```text
experiment/
|-- input/       Entrees source locales
|-- artifacts/   Artefacts generes reutilisables
`-- results/     Sorties de pipelines/sessions
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
|-- execution/       Registre de pipelines et dispatch de haut niveau
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

- Surfaces d'API programmatiques pour les environnements interactifs et les
  Jupyter Notebooks (`TrainingSessionConfigurator`, etc.).
- Renforcement des contrats de modeles et limites strictes de validation de
  configuration.
- Mecanismes d'etiquettes multi-tetes, pertes regularisees (pertes de
  coherence) et metriques d'evaluation etendues.
- Prereglages initiaux de sweep d'etude Optuna et integration des metriques
  d'objectifs.

Objectifs a court et moyen terme :

- Mettre a jour les schemas de workflow pour refleter la separation d'execution
  session/runtime actuelle.
- Documenter les guides de workflow Optuna recommandes et publier des tutoriels
  programmatiques.
- Stabiliser les formats de rapports de metriques et les comparaisons entre
  executions.

Objectifs a plus long terme :

- Ajouter d'autres familles de modeles au-dela de la pile actuelle de type U-Net.
- Definir des chemins d'export stables pour les modeles entraines et les
  artefacts d'evaluation.
- Soutenir des workflows plus riches d'analyse inter-experiences.
- Continuer a consolider les frontieres internes a mesure que les ADR se
  stabilisent.

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
