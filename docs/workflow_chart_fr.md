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
|
+--> [cli/pipelines/*]                              (7 Exécution des pipelines)
         (orchestration explicite des étapes de commande seulement ;
          résout la configuration, sélectionne un pipeline, puis
          délègue la construction et l’exécution aux couches en aval)

    +--> [cli/pipelines/train_model]               (7a Pipeline d’entraînement)
    |        (exécution complète d’une expérience)
    |        |
    |        +--> valider / résoudre les artefacts d’expérience requis
    |        |        (artefacts de jeu de données préparé, manifestes,
    |        |         schéma)
    |        |
    |        +--> [geopipe/specification/factory]
    |        |        (construire les DataSpecs à partir des artefacts
    |        |         préparés)
    |        |
    |        +--> [models/factory]
    |        |        (construire le modèle d’entraînement)
    |        |
    |        +--> [session/factory]
    |        |        (assembler la session d’entraînement :
    |        |         composants, état, callbacks, moteurs, runner)
    |        |
    |        +--> [session/runner]
    |                 (exécuter le cycle multi-phases
    |                  entraînement / validation)
    |
    +--> [cli/pipelines/evaluate_model]            (7b Pipeline d’évaluation)
    |        (exécution d’une évaluation unique)
    |        |
    |        +--> valider / résoudre les artefacts d’expérience requis
    |        |        (artefacts de jeu de données préparé, manifestes,
    |        |         schéma, entrées d’évaluation / source du modèle)
    |        |
    |        +--> [geopipe/specification/factory]
    |        |        (construire les DataSpecs d’évaluation)
    |        |
    |        +--> [models/factory]
    |        |        (construire le modèle d’évaluation)
    |        |
    |        +--> [session/factory]
    |        |        (assembler la session d’évaluation :
    |        |         composants, état, callbacks, moteur d’évaluation)
    |        |
    |        +--> [session/engine/policy]
    |                 (exécuter un passage unique d’évaluation et
    |                  produire métriques / exports)
    |
    +--> [cli/pipelines/train_overfit]             (7c Pipeline d’overfit)
             (validation de la chaîne complète sur un seul bloc)
             |
             +--> construire / sélectionner un seul bloc valide
             |        (chemin minimal d’acquisition de bloc pour la
             |         validation de débogage)
             |
             +--> construire des DataSpecs minimaux
             |        (spécification à bloc unique / portée réduite)
             |
             +--> [models/factory]
             |        (construire le modèle de débogage / overfit)
             |
             +--> [session/factory]
             |        (assembler une session d’entraînement compacte
             |         pour le test d’overfit)
             |
             +--> [session/engine/policy]
                      (entraîner de manière répétée sur le même bloc
                       jusqu’à atteindre un ajustement quasi parfait /
                       la cible attendue pour le débogage)
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