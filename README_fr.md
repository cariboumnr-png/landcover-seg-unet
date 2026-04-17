# Cadre de classification de la couverture terrestre multimodale

[Français](README_fr.md) | [English](README.md)

> ***Résumé en langage clair :***<br>
> *Ce projet fournit des outils pour préparer des images satellitaires et entraîner des modèles qui classifient la couverture terrestre. Il aide les utilisateurs à organiser leurs données, exécuter des modèles d’apprentissage profond et reproduire les résultats de manière cohérente.*


Un cadre modulaire et reproductible d’apprentissage profond pour la cartographie de la couverture terrestre au niveau du pixel.
Le système combine **imagerie spectrale Landsat**, **métriques topographiques dérivées du MNE** et **caractéristiques fondées sur l’expertise métier**, dans des artéfacts stables de **grille** et de **domaine**.
Le pipeline utilise des architectures U‑Net PyTorch et un flux de préparation de données entièrement basé sur des spécifications.

> **Statut du projet :**
> Ce dépôt est actuellement en **phase de recherche / expérimentale**.
> Les frontières de modules et les API ne sont **pas encore stables**.
> La construction du runtime est désormais **pilotée par la session**, les
> moteurs d’exécution agissant comme des couches de politique au‑dessus d’un
> noyau batch partagé. Les API publiques de session et de moteur sont encore en
> évolution et ne doivent pas être considérées comme stables.
``

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

- **Entraînement, évaluation et inférence**
  La construction du runtime est prise en charge par une couche de session qui
  assemble les composants, l’état runtime, les callbacks et les moteurs
  d’exécution. L’entraînement et l’évaluation sont pilotés par des moteurs
  limités à la politique (policy‑only) au‑dessus d’un exécuteur batch partagé,
  avec un runner optionnel basé sur des phases pour l’orchestration de plus haut
  niveau lorsque nécessaire.

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

Cette étape construit une session d’entraînement complète, incluant l’état
runtime, les moteurs d’exécution et un runner piloté par phases, à partir des
artefacts de données préparés.

Cette étape consomme les artefacts préparés mais ne modifie pas les données de base.

---

#### 4. Test « overfit » en silo (optionnel)

Exécute un test minimal de sur‑apprentissage sur un sous‑ensemble réduit afin de
valider la chaîne de bout en bout. Ce pipeline construit une session **sans
runner**, en exerçant directement le moteur d’exécution partagé. Il ne nécessite
aucune ingestion ou préparation préalable:

    experiment_run pipeline=train-overfit


>🔔 Ces commandes exécutent les configurations Hydra depuis `src/landseg/configs/`. Pour la plupart des utilisateurs, il est recommandé de fournir les paramètres via le fichier racine `settings.yaml`, conçu pour une personnalisation sécurisée sans modifier l’arborescence interne des configurations.

---

## 🚀 Feuille de route

### Court terme
- Mise à jour de la documentation pour refléter l’architecture runtime
  pilotée par la session (en cours)
- Diagrammes de workflow et références explicites aux ADR
- Exemples améliorés pour les pipelines d’entraînement et de sur‑apprentissage

### Moyen terme
- Clarification et formalisation des types de sessions supportées
  (p. ex. entraînement curriculaire, overfit, évaluation seule)
- Manifeste optionnel de tâches / phases défini par l’utilisateur
- Durcissement progressif des API publiques de session et de moteur

### Long terme
- Architectures de modèles supplémentaires
- Outils d’évaluation, d’export et de génération de rapports
- Consolidation des composants runtime stables vers une surface
  d’exécution orientée production (session / engine)

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
