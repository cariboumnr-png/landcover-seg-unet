# Cadre de classification de la couverture terrestre multimodale

[Français](README_fr.md) | [English](README.md)

> ***Résumé en langage clair :***<br>
> *Ce projet fournit des outils pour préparer des images satellitaires et entraîner des modèles qui classifient la couverture terrestre. Il aide les utilisateurs à organiser leurs données, exécuter des modèles d’apprentissage profond et reproduire les résultats de manière cohérente.*


Un cadre modulaire et reproductible d’apprentissage profond pour la cartographie de la couverture terrestre au niveau du pixel.
Le système combine **imagerie spectrale Landsat**, **métriques topographiques dérivées du MNE** et **caractéristiques fondées sur l’expertise métier**, dans des artéfacts stables de **grille** et de **domaine**.
Le pipeline utilise des architectures U‑Net PyTorch et un flux de préparation de données entièrement basé sur des spécifications.

> **Statut du projet :**
> Ce dépôt est actuellement en phase **recherche / expérimental**.
> Les limites des modules et les API **ne sont pas encore stables**.
> Une exécution orientée production (`engine/`) est prévue pour de futurs jalons mais **n’est pas encore incluse**.

---

# 📖 Aperçu

Ce dépôt fournit un flux de travail **de bout en bout, piloté par les artefacts**,
pour la préparation des jeux de données et l’entraînement de modèles de
segmentation de l’occupation du sol.

- **Artefacts de grille & de domaine**
  Tuilage déterministe sur une grille mondiale et alignement de rasters de
  domaine sur la grille, persistés sous forme d’artefacts réutilisables et
  protégés par hachage.

- **Pipeline de préparation des données**
  Validation de la géométrie des rasters → cartographie grille/fenêtres →
  mise en cache des blocs bruts → dérivation de caractéristiques spectrales et
  topographiques → construction de hiérarchies d’étiquettes → partitionnement
  et notation du jeu de données → normalisation → génération des schémas.

- **Schémas comme manifeste de référence**
  Les artefacts `schema.json` générés constituent la description canonique de la
  structure du jeu de données, de la provenance, des partitions, des formes de
  tenseurs, de la topologie des classes et de la normalisation. Aucun manifeste
  rédigé par l’utilisateur n’est requis pour les flux standards.

- **Spécifications de jeu de données (`DataSpecs`)**
  Une représentation d’exécution unifiée, dérivée des schémas et catalogues
  persistés, décrivant les entrées du modèle, la structure des classes, les
  partitions, la normalisation et le conditionnement de domaine optionnel.

- **Architectures de modèles**
  Variantes U‑Net multi‑têtes et U‑Net avec conditionnement de domaine aligné sur
  la grille en option.

- **Moteur d’entraînement, d’évaluation et d’inférence**
  Une couche d’exécution pilotée par phases, fondée sur un moteur multi‑têtes
  et des callbacks, prenant en charge l’entraînement, la validation,
  l’inférence, l’ordonnancement curriculaire, les métriques, les pertes, la
  gestion des points de contrôle et la génération d’aperçus. Conçue pour
  évoluer proprement vers un moteur d’évaluation dédié.

- **Reproductibilité par construction**
  L’entraînement et l’inférence ne consomment que des artefacts persistés
  (schémas, points de contrôle), avec hachage strict, chargement validé par
  schéma, état d’exécution explicite et politiques déterministes de
  reconstruction en cas de divergence, garantissant des expériences
  auditables et redémarrables à travers les exécutions et environnements.


Documentation détaillée :
- [Structure du dépôt](./docs/project_structure_fr.md)
- [Diagramme du flux de travail](./docs/workflow_chart_fr.md)

---


## ▶️ Utilisation

Avant de lancer une expérience, vous devez préparer vos rasters d’entrée et
organiser correctement la structure de votre projet. Veuillez commencer par lire
le guide de préparation des données :

📄 ./docs/data_preparation.md

Une fois vos rasters et dossiers prêts, configurez votre projet à l’aide du
fichier `settings.yaml` à la racine. Ce fichier constitue un point d’entrée
stable pour spécifier les entrées et les options de traitement, sans modifier
l’arborescence interne de configuration Hydra.

Installer le framework :

    pip install .

---

### Étapes du pipeline

Ce projet s’exécute selon des **étapes de pipeline explicites et consécutives**.
Chaque étape produit ou consomme des artefacts bien définis, régis par des
politiques de cycle de vie explicites.

#### 1. Ingestion des données

Traiter les rasters bruts pour produire des **blocs de données catalogués et
stables**, alignés sur une grille mondiale et persistés comme artefacts de base
réutilisables :

    experiment_run pipeline=ingest-data

Cette étape doit généralement être exécutée **une seule fois par jeu de données**,
sauf si les rasters d’entrée ou la configuration de la grille changent.

---

#### 2. Préparation des données à l’échelle de l’expérience

Préparer les artefacts spécifiques à une expérience (partitionnement du jeu de
données, normalisation, statistiques, schémas) à partir des blocs de données
préalablement ingérés :

    experiment_run pipeline=prepare-data

Cette étape peut être relancée avec différentes configurations d’expérience sans
ré‑ingérer les données brutes.

---

#### 3. Entraînement du modèle

Lancer un entraînement complet en utilisant les artefacts de données actuellement
préparés :

    experiment_run pipeline=train-model

Cette étape consomme les artefacts préparés mais ne modifie pas les données de base.

---

#### 4. Test « overfit » en silo (optionnel)

Exécuter un test minimal d’overfitting sur un sous‑ensemble restreint afin de
valider la chaîne de traitement de bout en bout. Ce pipeline **ne nécessite pas
d’ingestion ni de préparation préalables** :

    experiment_run pipeline=train-overfit


>🔔 Ces commandes exécutent les configurations Hydra depuis `src/landseg/configs/`. Pour la plupart des utilisateurs, il est recommandé de fournir les paramètres via le fichier racine `settings.yaml`, conçu pour une personnalisation sécurisée sans modifier l’arborescence interne des configurations.

---

## 🚀 Feuille de route

### Court terme
- Amélioration de la documentation et des exemples (en cours)

### Moyen terme
- Manifeste optionnel rédigé par l’utilisateur

### Long terme
- Architectures de modèles supplémentaires
- Outils d’évaluation et d’exportation
- Promotion graduelle des composants stables dans `engine/training`

---

## 🤝 Contribution

Ce projet est en phase expérimentale. La structure des modules, les noms et le comportement de la CLI peuvent évoluer.
Les contributions doivent se concentrer sur l’utilisabilité en recherche, sauf si elles s’alignent sur une Décision Architecturale (ADR) approuvée.

Veuillez consulter les ADR actives dans `docs/ADRs/` pour comprendre les décisions actuelles.

---

## 📜 Licence

Sous licence **Apache License, Version 2.0**.
Voir les fichiers `LICENSE` et `NOTICE` pour plus de détails.

© Sa Majesté le Roi du chef de l’Ontario,
représenté par le ministre des Richesses naturelles, 2026.
© Imprimeur du Roi pour l’Ontario, 2026.
