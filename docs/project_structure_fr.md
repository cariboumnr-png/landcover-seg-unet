## 📁 Structure actuelle du projet (priorité à la source — réduite aux composants principaux)

```
root/src/landseg
│
├── core/               # contrats au niveau du projet
│   └── protocols.py
|
├── grid/               # générer une grille mondiale stable
│   ├── builder.py      <- API du module
│   ├── io.py
│   └── layout.py
|
├── domain/             # associer les rasters du domaine à la grille mondiale
│   ├── io.py
│   ├── mapper.py       <- API du module
│   ├── tilemap.py
│   └── transform.py
|
├── dataprep/           # transformer les rasters bruts en artéfacts stables
│   ├── blockbuilder/
│   ├── mapper/
│   ├── normalizer/
│   ├── splitter/
│   ├── utils/
│   ├── pipeline.py     <- API du module
│   └── schema.py
│
├── dataset/            # consommer le schéma pour l’entraînement et le dataloading
│   ├── specs.py
│   ├── loader.py       <- API du module
│   └── validate.py
│
├── models/             # définir les structures de modèles (ex.: UNet, UNet++)
│   ├── backbones/
│   ├── multihead/
│   └── factory.py      <- API du module
│
├── training/           # entraîneur et ses composants
│   ├── callback/
│   ├── common/
│   ├── dataloading/
│   ├── heads/
│   ├── loss/
│   ├── metrics/
│   ├── optim/
│   ├── trainer/
│   └── factory.py      <- API du module
│
├── controller/         # construire le contrôleur (exécution des expériences)
│   ├── builder.py      <- API du module
│   ├── controller.py
│   └── phases.py
│
├── utils/              # utilitaires à l’échelle du projet
│
├── configs/            # arborescence de configuration Hydra fournie avec le package
│
└── cli/                # scripts CLI
    ├── main.py         <- API du module
    ├── end_to_end.py
    └── overfit_test.py
  ```