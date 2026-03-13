## 📁 Structure actuelle du projet (affichant deux niveaux)

```
/src/landseg/
├── cli/                          [Scripts CLI]
│   ├── main.py                   <- point d’entrée CLI
│   ├── end_to_end.py             # profil par défaut
│   └── overfit_test.py           # profil overfit
|
├── configs/                      [Arbo Hydra]
│   ├── inputs/
│   ├── models/
│   ├── prep/
│   ├── profile/
│   ├── runner/
│   ├── trainer/
│   ├── config.yaml               # entrée des configs Hydra
│   └── schema.py                 # schéma dataclass + validation runtime
|
├── core/                         [Contrats du projet]
│   ├── ingest_protocols/         # protocoles schéma/domaine/specs
│   ├── trainer_protocols/        # protocoles trainer/engine
│   ├── alias.py                  # alias de types
│   ├── grid_protocol.py          # protocole pour la grille
│   └── model_protocol.py         # protocole pour modèle multi‑têtes
|
├── grid_generator/               [Grilles globales immuables]
│   ├── generator.py              <- interface publique
│   ├── io.py                     # sauvegarde/chargement des artefacts grille
│   └── layout.py                 # définition de la grille
|
├── ingest_dataset/               [Ingestion rasters images/labels]
│   ├── blockbuilder/             # construction blocs (.npz)
│   ├── mapper/                   # projection rasters → grille
│   ├── normalizer/               # normalisation des blocs
│   ├── schema/                   # schéma JSON du dataset
│   ├── splitter/                 # division train/val/etc.
│   ├── config.py                 # config runtime module
│   └── pipeline.py               <- interface publique
|
├── ingest_domain/                [Ingestion rasters de domaine]
│   ├── io.py                     # sauvegarde/chargement artefacts domaine
│   ├── mapper.py                 # projection domaine → grille
│   ├── tilemap.py                # définition carte domaine
│   └── transform.py              # PCA vecteurs domaine
|
├── ingest_specs/                 [Specs pour dataloader du trainer]
│   ├── builder.py                # construction des specs
│   ├── factory.py                <- interface publique
│   └── validate.py               # validation schéma/ingest requis
|
├── models/                       [Architectures de modèles]
│   ├── backbones/                # UNet, UNet++ implémentés
│   ├── multihead/                # framework multi‑têtes
│   └── factory.py                <- interface publique
|
├── trainer_components/           [Composants runtime du trainer]
│   ├── callback/                 # système de callbacks
│   ├── dataloading/              # obtention dataloaders train/val/test
│   ├── heads/                    # specs pour chaque tête
│   ├── loss/                     # config des pertes par tête
│   ├── metrics/                  # config des métriques par tête
│   ├── optimization/             # config optimiseur + scheduler
│   └── factory.py                <- interface publique
|
├── trainer_engine/               [Boucle de train/stages]
│   ├── utils/                    # utilitaires ex: checkpoints
│   ├── engine_config.py/         # config runtime
│   ├── engine_state.py/          # état runtime
│   └── engine.py                 <- interface publique
|
├── trainer_runner/               [Orchestrateur d’entraînement]
│   ├── phase.py/                 # phases/curriculum
│   └── runner.py                 <- interface publique
|
└── utils/                        [Utilitaires globaux]
    ├── context.py                # contexte rasterio (open raster)
    ├── funcs.py                  # commodités diverses
    ├── logger.py                 # logger projet
    └── multip.py                 # wrapper multiprocessing
  ```