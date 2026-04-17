## Flux de travail actuel
```
[foundation/world_grids/builder]            (1 Grille monde – construction pure)
|
+--> [foundation/world_grids/lifecycle]
|        (persistance et validation des artefacts de grille)
|
+--> [foundation/domain_maps/mapper]        (2 Domaine → Grille, optionnel)
|        |
|        +--> [foundation/domain_maps/builder]
|        |        (calcul pur des caractéristiques du domaine)
|        |
|        +--> [foundation/domain_maps/lifecycle]
|                 (persistance des artefacts de domaine)
|
+--> [foundation/data_blocks/mapper]        (3 Imagerie/Étiquettes → fenêtres de grille)
|        |
|        +--> [foundation/data_blocks/builder]
|        |        (construction pure des blocs)
|        |
|        +--> [foundation/data_blocks/manifest]
|                 (mise à jour du catalogue et du schéma)
|
+--> [geopipe/specification/factory]        (4 Construction des DataSpecs)
|
+--> [models/factory]                       (5 Construction du modèle)
|
+--> [session/factory]                     (6 Frontière de construction de session)
|        |
|        +--> [session/components]          (composants : loaders, têtes, loss, optim)
|        |
|        +--> [session/state]               (initialisation de l’état runtime)
|        |
|        +--> [session/instrumentation]     (callbacks, exporteurs)
|        |
|        +--> [session/engine/batch]        (moteur d’exécution batch partagé)
|        |
|        +--> [session/engine/policy]       (politiques trainer / evaluator)
|        |
|        +--> [session/runner]              (optionnel : phases et runner)
|
+--> [cli/pipelines/*]                      (7 Exécution du pipeline)
``
```

### Notes d’interprétation (mises à jour)

- Toutes les étapes de construction de la fondation (grille, domaine, blocs)
restent pures et déterministes.
- Toute la logique de réutilisation, d’écrasement et de validation passe par
artifacts.Controller / PayloadController.
- La construction des DataSpecs et du modèle a lieu avant la frontière
de session et ceux‑ci sont traités comme des entrées de la construction de
session.
- La construction de session est centralisée dans session/factory et est
responsable de :

  - la construction des composants
  - l’initialisation de l’état runtime
  - le rattachement (binding) des callbacks
  - l’instanciation des moteurs
  - l’assemblage optionnel du runner


- Les moteurs sont limités à la politique (policy‑only) et reçoivent un état et
des composants entièrement initialisés.
- La CLI exécute des étapes de pipeline explicites et demande une session ;
elle n’assemble pas les internals du runtime.
- Le flux sépare clairement :

  - les artefacts de fondation (ingestion et préparation)
  - les artefacts d’expérience (specs, modèle)
  - le runtime d’entraînement (cycle de vie détenu par la session)