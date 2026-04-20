## 📁 Structure actuelle du projet (affichant deux niveaux)

Last updated: 2026-04-07

```
./src/landseg/
├── artifacts/                    [Cycle de vie et persistance des artefacts unifiés]
│   ├── checkpoint.py             # utilitaires de sauvegarde des modèles (checkpoints)
│   ├── controller.py             # logique de chargement/assemblage/reconstruction pilotée par politiques
│   ├── payload_io.py             # E/S de payload structuré et métadonnées
│   ├── paths.py                  # structure canonique des chemins d’artefacts
│   └── policy.py                 # politiques de cycle de vie des artefacts
|
├── cli/                          [Entrée CLI et dispatch des pipelines]
│   ├── main.py                   <- entrée CLI basée sur Hydra
│   └── pipelines/                # étapes explicites de pipeline
|
├── configs/                      [Arbre de configuration Hydra]
│   ├── dataspecs/                # spécification des données
│   ├── foundation/               # grille, domaine, blocs de données
│   ├── models/                   # configuration des modèles
│   ├── pipeline/                 # pipelines d’ingestion / préparation / entraînement
│   ├── session/                  # configuration de session
│   ├── transform/                # configurations de transformation des données
│   ├── config.yaml               # point d’entrée de l’arbre de configuration Hydra
│   └── schema.py                 # validation des dataclasses de configuration
|
├── core/                         [Contrats au niveau du projet]
│   ├── data_specs.py             # spécification de dataset à l’exécution
│   └── model_protocol.py         # contrats d’interface des modèles
|
├── geopipe/                      [Pipeline de préparation des données géospatiales]
│   ├── core/                     # structures de données immuables
│   ├── foundation/               # préparation de grille, domaine et blocs
│   ├── transform/                # transformations liées aux expériences
│   ├── specification/            # construction des DataSpecs
│   └── utils/                    # utilitaires au niveau du module
|
├── models/                       [Architectures de modèles]
│   ├── backbones/                # UNet, UNet++ etc.
│   ├── multihead/                # cadre de modèle multi-têtes
│   └── factory.py                # construction des modèles
|
├── session/                      [système d’exécution]
|   ├── common/                   # types partagés dans le module
│   ├── components/               # pertes, métriques, dataloaders
│   ├── engine/                   # moteurs de batch et politiques
│   ├── instrumentation/          # callbacks, utilitaire de prévisualisation
│   ├── orchestration/            # orchestration des sessions
│   ├── factory.py                # constructeurs de sessions
│   └── metadata.py               # métadonnées de session
|
├── utils/                        [Shared utilities]
|   ├── logger.py                 # journalisation du projet
|   └── multip.py                 # outils de multiprocessus
|
└── _constants.py                 # constantes globales du projet
  ```