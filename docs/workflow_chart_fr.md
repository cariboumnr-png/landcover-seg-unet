## Flux de travail actuel
```
[configs/]
└─> grid/builder.py                    (1 Grille mondiale)
    ├─> domain/mapper.py               (2 DK → grille, optionnel)
    └─> dataprep/pipeline.py           (3 Fit/Test → grille)
        └─> dataprep/schema.py         (4 Schéma de données)
            ├─> models/factory.py      (5.1 Modèle)
            ├─> dataset/builder.py     (5.2 Chargeurs de données)
            ├─> training/heads         (5.3 Influencé par les données)
            ├─> training/loss          (5.4 Spécifié par la tête)
            ├─> training/metrics       (5.5 Spécifié par la tête)
            └─> training/optim|callback|... (5.6 Autres)
                └─> training/factory.py → Trainer      (6 Entraîneur 🗸)
                    └─> controller/builder.py + phases (7 Contrôleur 🗸)
                        └─> cli/main.py                (8 Démarrage ➝)
```