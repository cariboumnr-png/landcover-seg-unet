## Flux de travail actuel
```
[grid_generator/generator]                  (1 Grille globale)
        |
        +--> [ingest_domain/mapper]         (2 Domaine → Grille, optionnel)
        |
        +--> [ingest_dataset/pipeline]      (3 Imagerie/Labels → Blocs)
                |
                +--> [ingest_specs/factory] (4 Génération des specs)
                        |
                        +--> [models/factory]            (5.1 Modèle)
                        +--> [trainer_components/*]      (5.2 Têtes/Perte/Optim)
                        |        (dataloaders, métriques, callbacks, ...)
                        |
                        +--> [trainer_engine/engine]     (6 Moteur d’entraînement)
                                |
                                +--> [trainer_runner/*]  (7 Phases/Runner)
                                        |
                                        +--> [cli/main]  (8 Exécution)
```