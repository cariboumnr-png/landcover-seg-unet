## Flux de travail actuel
```
[foundation/world_grids/builder]            (1 Grille mondiale – construction pure)
|
+--> [foundation/world_grids/lifecycle]
|        (persistance et validation des artefacts de grille)
|
+--> [foundation/domain_maps/mapper]        (2 Domaine → Grille, optionnel)
|        |
|        +--> [foundation/domain_maps/builder]
|        |        (calcul pur des features de domaine)
|        |
|        +--> [foundation/domain_maps/lifecycle]
|                 (persistance des artefacts de domaine)
|
+--> [foundation/data_blocks/mapper]        (3 Imagerie/Labels → fenêtres de grille)
|        |
|        +--> [foundation/data_blocks/builder]
|        |        (construction pure des blocs)
|        |
|        +--> [foundation/data_blocks/manifest]
|                 (catalogue et mise à jour du schéma)
|
+--> [geopipe/specification/factory]        (4 Construction des DataSpecs)
|
+--> [models/factory]                       (5.1 Construction du modèle)
|
+--> [trainer/components/]                  (5.2 Têtes / pertes / optimiseur / dataloaders / métriques / callbacks)
|
+--> [trainer/engine/engine]                (6 Moteur d’entraînement)
|
+--> [trainer/runner/]                      (7 Phases / Runner)
|
+--> [cli/pipelines/*]                      (8 Exécution des pipelines)
```

### Notes d’interprétation

- Toutes les étapes de **construction** (grille, domaine, blocs) sont désormais **pures et déterministes**.
- Toute la logique de réutilisation, d’écrasement et de validation passe par :
  **`artifacts.Controller` / `PayloadController`**.
- Le CLI exécute des **étapes de pipeline explicites** plutôt qu’un run implicite end-to-end.
- Le workflow sépare clairement :
  - **artefacts foundation** (ingestion)
  - **artefacts experiment** (transformation)
  - **runtime d’entraînement**