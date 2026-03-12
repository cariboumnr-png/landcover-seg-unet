## 📁 Structure actuelle du projet (priorité à la source — réduite aux composants principaux)

```
root/src/landseg
│
├── core/                      # contrats et protocoles au niveau du projet
│   ├── dataprep_schema.py
│   ├── dataset_specs.py
│   ├── grid_protocol.py
│   └── model_protocol.py
│
├── prep_grid/                 # fabrique : génère la grille mondiale stable
│   └── builder.py             # ← API du module
│
├── prep_domain/               # fabrique : prépare les rasters domaine alignés sur la grille
│   └── mapper.py              # ← API du module
│
├── prep_raster/               # fabrique : prépare les rasters image + labels
│   └── pipeline.py            # ← API du module
│
├── data_schema/               # fabrique : assemble et valide le schéma final d’entraînement
│   └── builder.py             # ← API du module
│
├── dataset/                   # runtime : charge et valide les données via le schéma produit
│   └── loader.py              # ← API du module
│
├── models/                    # définitions des modèles et fabrique de modèles
│   └── factory.py             # ← API du module
│
├── training/                  # moteur d’entraînement et composants runtime
│   └── factory.py             # ← API du module
│
├── controller/                # orchestration des expériences et phases du contrôleur
│   └── builder.py             # ← API du module
│
├── utils/                     # utilitaires généraux du projet
│   ├── contxt.py
│   ├── funcs.py
│   ├── logger.py
│   ├── multip.py
│   ├── pca.py
│   └── preview.py
│
├── configs/                   # arborescence de configuration Hydra
│
└── cli/                       # points d’entrée CLI
    └── main.py                # ← API du module
  ```