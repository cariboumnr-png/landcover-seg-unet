# Cadre de classification multimodale de l’occupation du sol

[English](README.md) | [Français](README_fr.md)

>***Résumé en langage clair :***<br>
>*Ce projet fournit des outils pour préparer des images satellites et entraîner*
>*des modèles de classification de l’occupation du sol. Il aide les utilisateurs à organiser les données, exécuter des modèles d’apprentissage profond*
>*et reproduire les résultats de manière cohérente.*

Un cadre modulaire d’apprentissage profond, orienté artefacts, pour la cartographie
de l’occupation du sol au niveau du pixel. Le système intègre des **images spectrales Landsat**,
des **métriques topographiques dérivées de modèles numériques d’élévation (DEM)**,
et des **caractéristiques métier** au moyen d’un pipeline structuré de préparation
des données et d’un environnement d’entraînement basé sur des sessions, reposant
sur des **architectures de segmentation de type U-Net** (implémentation PyTorch).
La prise en charge actuelle des modèles comprend des variantes U-Net multi-têtes
configurables, notamment U-Net standard, U-Net++ et des dorsales de type U-Net3+,
avec des goulots d’étranglement convolutionnels, transformeurs ou hybrides en
option. La configuration par défaut demeure un U-Net convolutionnel conservateur
afin d’assurer la stabilité des expériences de référence, tandis que les goulots
d’étranglement fondés sur les transformeurs sont disponibles pour les expériences
nécessitant un contexte spatial plus étendu et des interactions de caractéristiques
à longue portée.

> **État du projet :**
> Ce dépôt est actuellement en mode **recherche / expérimental**. Les limites des modules
> et les API ne sont **pas encore stables**. Certaines interfaces sont stables pour l’utilisation,
> tandis que d’autres (notamment les couches avancées d’orchestration et d’étude) sont encore en développement.

---

# 📖 Vue d’ensemble

Ce dépôt fournit un flux de travail complet pour préparer des jeux de données géospatiaux
et entraîner des modèles de segmentation de l’occupation du sol, avec un fort accent sur la reproductibilité
et une circulation structurée des données.

## Concepts clés

- **Artefacts fondamentaux (préparation des données)**

  Les rasters bruts sont transformés en artefacts structurés alignés sur une grille
  (par exemple : blocs de données, cartes de domaines). Ces artefacts servent de passerelle entre
  les formats géospatiaux (GeoTIFF) et les entrées de modèles basées sur des tenseurs.

- **Définition des expériences (DataSpecs + modèle)**

  Les artefacts préparés sont assemblés dans des `DataSpecs`, qui définissent :

  - Les entrées du modèle
  - Les partitions du jeu de données
  - La normalisation
  - La structure des classes

  Les modèles sont construits à partir de configurations et associés à ces spécifications.

- **Session (système d’exécution)**

  L’entraînement et l’évaluation sont exécutés via une session, qui assemble :

  - Les chargeurs de données (*dataloaders*)
  - Les modèles
  - Les fonctions de perte et optimiseurs
  - Les moteurs d’exécution
  - Les rappels (*callbacks*) et l’instrumentation

- **Pipelines (couche d’exécution)**

  Les pipelines CLI orchestrent le flux de travail. Ils résolvent la configuration et les artefacts,
  puis délèguent la construction au système (ils n’implémentent pas la logique métier principale).

## Fonctionnalités principales

- **Flux de travail orienté artefacts**

  Toutes les données intermédiaires et finales sont stockées comme artefacts versionnés.
  Les utilisateurs configurent le système ; le cycle de vie des artefacts et leur réutilisation sont gérés automatiquement.

- **Préparation déterministe des données**

  L’alignement des grilles et des domaines garantit une structure spatiale cohérente entre les exécutions.

- **Jeux de données pilotés par spécifications (`DataSpecs`)**

  Un unique objet d’exécution définit toutes les entrées du modèle, les partitions et la normalisation.

- **Environnement d’exécution basé sur les sessions**

  La logique d’entraînement et d’évaluation est encapsulée dans un système d’exécution structuré.

- **Suivi découplé (développement préliminaire)**

  La prise en charge de TensorBoard est disponible via une instrumentation basée sur des callbacks
  (sans dépendance à un fournisseur spécifique).

  Des moteurs supplémentaires (par exemple MLflow) sont prévus.

---

## ⚠️ Notes sur la stabilité

- **Stables pour utilisation**

  - Pipelines d’ingestion et de préparation des données
  - Construction de jeux de données basée sur des artefacts
  - Pipelines d’entraînement et d’évaluation
  - Intégration TensorBoard (suivi de base)

- **En développement actif**

  - Flux de travail basés sur des notebooks
    *(appelés à devenir le point d’entrée principal)*
  - Couche d’étude / exploration paramétrique
    *(utilitaires d’expérimentation basés sur Optuna)*
  - API de session et d’exécution
    *(susceptibles d’évoluer dans le temps)*

---

**Documentation plus détaillée disponible ici :**
- [Structure du dépôt](./docs/project_structure.md)
- [Schéma du flux de travail](./docs/workflow_chart.md)

---

## ▶️ Utilisation de l’entrée CLI

Avant d’exécuter une expérience, vous devez préparer vos rasters d’entrée et organiser correctement
le dossier de votre projet. Commencez par consulter le guide de préparation des données :

📄 [**Guide de préparation des données**](./docs/data_preparation.md)

Une fois vos rasters et dossiers prêts, configurez votre projet à l’aide du fichier
`settings.yaml` situé à la racine. Ce fichier fournit un point d’entrée stable pour spécifier
les entrées et les options de traitement sans modifier l’arborescence interne de configuration Hydra.

Installez le framework :

    pip install .

---

### Étapes du pipeline

Ce projet fonctionne à travers des **étapes de pipeline explicites et consécutives**.
Chaque étape produit ou consomme des artefacts bien définis, gouvernés par des politiques explicites
de cycle de vie.

#### 1. Ingestion des données

Traitez les rasters bruts en **blocs de données stables et catalogués** alignés sur une grille mondiale
et persistés sous forme d’artefacts fondamentaux réutilisables :

    landseg pipeline=data-ingest

Cette étape doit généralement être exécutée **une seule fois par jeu de données**,
sauf si les rasters d’entrée ou la configuration de la grille changent.

---

#### 2. Préparation des données propre à l’expérience

Préparez les artefacts spécifiques à l’expérience (partitions du jeu de données, normalisation, statistiques,
schémas) à partir des blocs de données précédemment ingérés :

    landseg pipeline=data-prepare

Cette étape peut être réexécutée avec différentes configurations d’expérience sans
réingérer les données brutes.

---

#### 3. Entraînement du modèle

Exécutez une tâche complète d’entraînement en utilisant les artefacts de jeu de données actuellement préparés :

    landseg pipeline=model-train

Cette étape construit une session complète d’entraînement, incluant l’état d’exécution,
les moteurs d’exécution et un exécuteur piloté par phases, à partir des artefacts préparés.

Cette étape consomme les artefacts préparés mais ne modifie pas les données fondamentales.

---

#### 4. Évaluation du modèle

Exécutez une tâche d’évaluation autonome à l’aide des artefacts de jeu de données actuellement préparés
et d’un point de contrôle entraîné :

    landseg pipeline=model-evaluate \
      pipeline.model_evaluate.checkpoint=path/to/checkpoint

Cette étape construit une session d’évaluation uniquement à partir des artefacts préparés
et du point de contrôle fourni, puis exécute l’inférence et le calcul des métriques sur la partition
d’évaluation configurée (par exemple `val` ou `test`) sans effectuer d’entraînement,
d’optimisation ou de création de point de contrôle. Elle est destinée à l’évaluation post-entraînement
et à la production de rapports, et consomme les artefacts préparés sans modifier les données fondamentales.

---

#### 5. Test de surapprentissage isolé (*overfit silo test*) (optionnel)

Exécutez un test minimal de surapprentissage sur un petit sous-ensemble afin de valider l’ensemble de la chaîne de traitement.
Ce pipeline construit une session **sans exécuteur**, en utilisant directement le moteur d’exécution partagé.
Il ne nécessite ni ingestion ni préparation préalable :

    landseg pipeline=diagnose-overfit


>🔔 Ces commandes exécutent des configurations Hydra depuis `src/landseg/configs/`. Ces
fichiers internes contrôlent le comportement du framework et ne devraient être modifiés que par
des utilisateurs avancés familiers avec Hydra et la structure du projet. Pour la plupart des flux de travail,
toutes les entrées requises devraient être fournies via le fichier `settings.yaml` à la racine.

---

## 📊 Suivi et visualisation

Les métriques et journaux d’entraînement sont émis via une instrumentation basée sur des callbacks.

- TensorBoard est actuellement pris en charge *(utilisation locale)*
- Le suivi est découplé des composants internes du framework
- Une prise en charge future est prévue pour des moteurs supplémentaires *(par exemple MLflow)*

---

## 🧠 Modèle conceptuel

Le système impose une séparation stricte des responsabilités :

- **Couche fondamentale**
  Construction déterministe des données
  *(grille, domaine, blocs)*

- **Couche des artefacts**
  Persistance, validation et politiques de réutilisation

- **Couche des expériences**
  `DataSpecs` et définition des modèles

- **Couche des sessions**
  Exécution à l’exécution et orchestration du cycle de vie

- **Couche d’exécution**
  Sélection et coordination des pipelines

**Flux des dépendances**

    foundation → artifacts → experiment → session → execution

## 📦 Comportement des artefacts (résumé orienté utilisateur)

Les artefacts sont générés et réutilisés automatiquement. Les utilisateurs ne gèrent pas manuellement les politiques de cycle de vie.

Les artefacts de jeux de données préparés sont stockés sous :

    <racine-expérience-définie-par-l’utilisateur>/artifacts

Les sorties de session sont stockées sous :

    <racine-expérience-définie-par-l’utilisateur>/results/run_xxxx/

Les artefacts servent de source de vérité pour la reproductibilité.

---

## 🚀 Feuille de route

### Court terme
- Flux de travail centré sur les notebooks (point d’entrée convivial)
- Amélioration de la visualisation et des rapports d’expériences
- Intégration TensorBoard améliorée (journalisation enrichie, aperçus)

### Moyen terme
- Interface stable d’étude / exploration d’hyperparamètres (Optuna)
- Intégration MLflow pour le suivi des expériences
- Amélioration de la clarté et des garanties de configuration des sessions
- Standardisation des sorties d’évaluation et des schémas de rapports

### Long terme
- Comparaison inter-expériences et flux de travail d’étude
- Architectures de modèles supplémentaires
- Surface d’exécution orientée production
- Utilitaires étendus d’exportation et de déploiement

---

## 🤝 Contribution

Ce projet est dans une phase expérimentale. La structure des modules, la nomenclature et le comportement
CLI peuvent évoluer. Les contributions devraient se concentrer sur l’utilisabilité pour la recherche, sauf
si elles sont alignées avec un *Architecture Decision Record* (ADR) approuvé.

Veuillez consulter les ADR actifs dans `docs/ADRs/` afin de comprendre les décisions de conception actuelles.

---

## 📜 Licence

Distribué sous la **Licence Apache, Version 2.0**.
Consultez les fichiers `LICENSE` et `NOTICE` pour plus de détails.

© Sa Majesté le Roi du chef de l’Ontario,
représenté par le ministre des Richesses naturelles, 2026.
© Imprimeur du Roi pour l’Ontario, 2026.
