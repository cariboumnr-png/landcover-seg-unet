## 📁 Structure actuelle du projet (affichant deux niveaux)

Dernière mise à jour: 2026-05-12

```
./notebooks/
│   ├── 01_data_preparation.ipynb # ingestion, validation et préparation des données
│   └── 02_model_train.ipynb      # démo : session complète d'entraînement continu
|
./src/landseg/
├── adapters/                     [Interfaces d'entrée externes]
│   ├── api.py                    <- point d'entrée API programmatique
│   └── cli.py                    <- interface CLI pilotée par Hydra
│
├── artifacts/                    [Gestion du cycle de vie des artefacts & persistance]
│   ├── checkpoint.py             # utilitaires de sauvegarde/chargement de checkpoints
│   ├── controller.py             # résolution/build/rebuild selon politiques
│   ├── paths.py                  # définition des chemins canoniques des artefacts
│   ├── payload_io.py             # sérialisation payload structuré + métadonnées
│   └── policy.py                 # règles et stratégies de cycle de vie
|
├── configs/                      [Arborescence de configuration Hydra]
│   ├── _schema_/                 # dataclasses de configuration (par section)
│   ├── dataspecs/                # configurations de spécification des datasets
│   ├── foundation/               # configs grille / domaine / blocs
│   ├── models/                   # configurations des architectures modèles
│   ├── pipeline/                 # configs des pipelines (ingest / prepare / train)
│   ├── session/                  # configuration des sessions d'exécution
│   ├── transform/                # définitions des transformations de données
│   ├── schema.py                 # schéma de configuration composite
│   └── config.yaml               # point d'entrée Hydra
|
├── core/                         [Contrats et abstractions du projet]
│   ├── data_specs.py             # contrat de spécification des datasets à runtime
│   ├── model_protocol.py         # interface/contrat des modèles
│   └── session_results.py        # structures de sortie des sessions
|
├── execution/                    [Couche d'orchestration d'exécution]
│   ├── pipelines/                # implémentations explicites des étapes pipeline
│   ├── executor.py               # point d'entrée unifié d'exécution
│   └── resolver.py               # résolution de configuration et injection dépendances
|
├── geopipe/                      [Pipeline de préparation des données géospatiales]
│   ├── core/                     # structures de données géospatiales immuables
│   ├── foundation/               # construction grille, domaine et blocs
│   ├── specification/            # construction et assemblage des DataSpecs
│   ├── transform/                # transformations spécifiques à l'expérience
│   └── utils/                    # utilitaires du geopipeline
|
├── models/                       [Architectures et composition des modèles]
│   ├── backbones/                # réseaux backbone (ex : UNet, UNet++)
│   ├── multihead/                # framework de modèles multi-têtes
│   └── factory.py                # construction et assemblage des modèles
|
├── session/                      [Système d'exécution runtime]
│   ├── commmon/                  # types partagés et contrats internes
│   ├── data/                     # adaptateurs dataloader / batching
│   ├── engine/                   # moteurs d'exécution (batch et epoch)
│   ├── instrumentation/          # callbacks, logging, suivi, preview
│   ├── orchestration/            # orchestration du cycle de session
│   ├── factory.py                # construction des sessions
│   └── metadata.py               # métadonnées et traçabilité des sessions
|
├── study/                        [Couche expérimentation & étude]
│   ├── analysis/                 # analyse post-sweep des modèles
│   └── sweep/                    # exploration hyperparamétrique (Optuna)
|
├── utils/                        [Utilitaires partagés]
│   ├── logger.py                 # utilitaires de logging du projet
│   └── multip.py                 # helpers de multiprocessing
|
└── _constants.py                 # constantes globales du projet
  ```