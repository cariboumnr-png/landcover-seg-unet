## Workflow actuel

Dernière mise à jour : 2026-05-12

```
[foundation/world_grids/builder]            (1 Grille globale – construction pure)
|
+--> [artifacts/controller]
|        (résolution / build / réutilisation selon politique)
|
+--> [foundation/world_grids/lifecycle]
|        (persistance et validation de l’artefact grille)
|
+--> [foundation/domain_maps/mapper]        (2 Domaine → alignement sur grille, optionnel)
|        |
|        +--> [foundation/domain_maps/builder]
|        |        (calcul pur des features de domaine)
|        |
|        +--> [artifacts/controller]
|        |        (résolution / build / réutilisation artefact domaine)
|        |
|        +--> [foundation/domain_maps/lifecycle]
|                 (persistance et validation des artefacts domaine)
|
+--> [foundation/data_blocks/mapper]        (3 Imagerie / labels → fenêtres de grille)
|        |
|        +--> [foundation/data_blocks/builder]
|        |        (construction pure des blocs)
|        |
|        +--> [artifacts/controller]
|        |        (résolution / build / réutilisation des blocs)
|        |
|        +--> [foundation/data_blocks/manifest]
|                 (catalogue, enregistrement schéma et indexation)
|
+--> [geopipe/specification/factory]        (4 Construction des DataSpecs à partir des artefacts)
|
+--> [models/factory]                       (5 Construction et assemblage du modèle)
|
+--> [session/factory]                      (6 Frontière de construction de session)
|        |
|        +--> [session/data]                (dataloaders et adaptateurs de batch)
|        |
|        +--> [session/engine]
|        |        (moteurs d’exécution batch + epoch)
|        |
|        +--> [session/instrumentation]
|        |        (callbacks, logging, suivi, export)
|        |
|        +--> [session/orchestration]
|        |        (gestion du cycle de vie et coordination des phases)
|        |
|        +--> [session/metadata]
|                 (suivi et contexte runtime de la session)
|
|
+--> [execution/executor]                  (7 Point d’entrée d’exécution)
         (résout la config, sélectionne le pipeline, délègue)

    +--> [execution/pipelines/train]       (7a Pipeline d’entraînement)
    |        (cycle complet d’expérimentation)
    |        |
    |        +--> résolution des artefacts via [artifacts/controller]
    |        |        (grille, domaine, blocs, manifests, schéma)
    |        |
    |        +--> [geopipe/specification/factory]
    |        |        (construction des DataSpecs)
    |        |
    |        +--> [models/factory]
    |        |        (construction du modèle d’entraînement)
    |        |
    |        +--> [session/factory]
    |        |        (assemblage de la session d’entraînement)
    |        |
    |        +--> [session/orchestration]
    |                 (exécution train/validate mono ou multi-phase)
    |
    +--> [execution/pipelines/evaluate]    (7b Pipeline d’évaluation)
    |        (évaluation en un passage)
    |        |
    |        +--> résolution des artefacts via [artifacts/controller]
    |        |        (données, manifests, source modèle)
    |        |
    |        +--> [geopipe/specification/factory]
    |        |        (construction DataSpecs d’évaluation)
    |        |
    |        +--> [models/factory]
    |        |        (construction du modèle d’évaluation)
    |        |
    |        +--> [session/factory]
    |        |        (assemblage de la session d’évaluation)
    |        |
    |        +--> [session/engine]
    |                 (exécution et émission métriques / exports)
    |
    +--> [execution/pipelines/overfit]     (7c Pipeline de sur-apprentissage)
             (validation complète sur portée minimale)
             |
             +--> résolution / build d’un bloc minimal via [artifacts/controller]
             |        (acquisition artefact pour debug)
             |
             +--> construction DataSpecs minimal
             |        (spécification mono-bloc / restreinte)
             |
             +--> [models/factory]
             |        (construction modèle debug)
             |
             +--> [session/factory]
             |        (assemblage session compacte)
             |
             +--> [session/engine]
                      (entraînement répété jusqu’à convergence)
```

---

### Notes d’interprétation (mises à jour)

- Toutes les étapes de construction de la couche foundation (grille,
  domaine, blocs) restent pures, déterministes, et sans effets de bord.

- Toutes les décisions de réutilisation, de reconstruction, d’écrasement
  et de validation des artefacts sont centralisées via
  artifacts.controller, imposant une gestion du cycle de vie pilotée
  par des politiques.

- Les builders de la couche foundation ne gèrent pas la persistance ;
  ils produisent des sorties en mémoire qui sont matérialisées
  exclusivement via la couche des artefacts.

- Toutes les étapes en aval opèrent sur des artefacts résolus, et non sur
  des intermédiaires recalculés ou implicites.

- Les DataSpecs et la construction du modèle se produisent strictement
  avant la frontière de session et sont traités comme des entrées
  entièrement résolues et immuables.

- La construction de la session est centralisée dans session/factory et
  prend en charge :

  - la construction des interfaces de données (dataloaders, samplers)
  - l’assemblage des composants (liaisons du modèle, fonctions de perte,
    optimiseurs)
  - l’initialisation de l’état d’exécution
  - le raccordement des callbacks et de l’instrumentation
  - l’instanciation des moteurs d’exécution
  - la mise en place de l’orchestration du cycle de vie

- Les éléments internes de la session sont entièrement configurés avant
  l’exécution ; aucune mutation structurelle ne se produit pendant
  l’exécution.

- Les moteurs d’exécution opèrent sur un état et des composants injectés
  et ne codent pas la configuration ni les décisions de cycle de vie.

- Le contrôle du cycle de vie (phases train/validate et transitions) est
  géré par l’orchestration de session, et non par les pipelines.

- La couche d’exécution (execution.executor + pipelines) est responsable
  uniquement de :

  - la résolution de la configuration
  - la sélection du pipeline
  - la coordination de la résolution des artefacts via
    artifacts.controller
  - la délégation de la construction aux factories (specs, modèle,
    session)

- Les pipelines ne construisent pas les éléments internes à l’exécution
  et agissent strictement comme des coquilles d’orchestration légères.

- Le workflow impose une séparation claire entre les couches :

  - couche foundation
    construction déterministe des données (grille, domaine, blocs)

  - couche artefacts
    persistance, validation, versionnement, et politiques de
    réutilisation

  - couche expérimentation
    spécification des données (DataSpecs) et construction du modèle

  - couche session
    système d’exécution, exécution et orchestration du cycle de vie

  - couche exécution
    orchestration de haut niveau et sélection de pipeline

- Le système maintient un flux de dépendance unidirectionnel :

  foundation → artifacts → experiment → session → execution

- Cette structure garantit :

  - la reproductibilité via un contrôle explicite des artefacts
  - une séparation stricte entre les responsabilités de build et runtime
  - des pipelines composables et prévisibles
  - une reconstruction déterministe à partir des configs et des artefacts