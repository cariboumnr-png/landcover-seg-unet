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

## 📖 Aperçu

Ce dépôt fournit un flux complet pour la préparation des jeux de données et l’entraînement de modèles de segmentation de la couverture terrestre :

- **Artéfacts de grille et de domaine :** Découpage déterministe du globe et alignement des rasters du domaine.
- **Pipeline de préparation des données :** Mappage de fenêtres → mise en cache de blocs raster → dérivation de caractéristiques spectrales/topographiques → hiérarchie d’étiquettes → normalisation → évaluation et division du jeu de données → génération de schéma.
- **Spécifications de jeu de données :** Une représentation unifiée (`DataSpecs`) décrivant les formes, la topologie des classes, les divisions et les paramètres de normalisation.
- **Architectures de modèles :** U‑Net multi‑tête et variantes U‑Net avec conditionnement optionnel basé sur le domaine.
- **Gestionnaire d’entraînement :** Interface unifiée pour l’entraînement, l’inférence, les métriques, les callbacks et la génération d’aperçus.
- **Reproductibilité :** Hachage strict des artéfacts, validation des schémas et reconstruction automatique en cas de divergence.

Documentation détaillée :
- [Structure du dépôt](./docs/project_structure_fr.md)
- [Diagramme du flux de travail](./docs/workflow_chart_fr.md)

---

## ▶️ Utilisation

Avant d’exécuter des expériences, consulter [ce guide](./docs/data_preparation_fr.md) pour les instructions de préparation des rasters d’images et d’étiquettes.

Installer le cadre :

    pip install .

Exécuter une expérience complète :

    experiment_run profile=end_to_end

Exécuter un test d’ajustement excessif :

    experiment_run profile=overfit_test

Ces commandes exécutent les configurations Hydra depuis `src/landseg/configs/`. Pour la plupart des utilisateurs, il est recommandé de fournir les paramètres via le fichier racine `settings.yaml`, conçu pour une personnalisation sécurisée sans modifier l’arborescence interne des configurations.

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
