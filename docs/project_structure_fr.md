## 📁 Structure actuelle du projet (affichant deux niveaux)

```
./src/landseg/
├── artifacts/                    [Cycle de vie et persistance des artefacts unifiés]
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
│   ├── foundation/               # grille, domaine, blocs de données
│   ├── pipeline/                 # pipelines d’ingestion / préparation / entraînement
│   ├── models/                   # configuration des modèles
│   ├── trainer/                  # configuration d’exécution d’entraînement
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
│   └── specification/            # construction des DataSpecs
|
├── models/                       [Architectures de modèles]
│   ├── backbones/                # UNet, UNet++ etc.
│   ├── multihead/                # têtes conditionnées par domaine
│   └── factory.py                # construction des modèles
|
├── trainer/                      [Système d’entraînement]
│   ├── components/               # pertes, métriques, dataloaders
│   ├── engine/                   # boucle d’entraînement principale
│   └── runner/                  # orchestration de l’entraînement
|
└── utils/                        [Utilitaires partagés]
    ├── logger.py                 # journalisation du projet
    ├── funcs.py                  # hachage, JSON, fonctions utilitaires
    └── multip.py                 # outils de multiprocessus
  ```