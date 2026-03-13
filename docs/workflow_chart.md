## Current workflow
```
[grid_generator/generator]                  (1 World Grid)
        |
        +--> [ingest_domain/mapper]         (2 Domain → Grid, optional)
        |
        +--> [ingest_dataset/pipeline]      (3 Imagery/Labels → Blocks)
                |
                +--> [ingest_specs/factory] (4 Build Data Specs)
                        |
                        +--> [models/factory]            (5.1 Model Build)
                        +--> [trainer_components/*]      (5.2 Heads/Loss/Optim)
                        |        (dataloaders, metrics, callbacks, ...)
                        |
                        +--> [trainer_engine/engine]     (6 Trainer Engine)
                                |
                                +--> [trainer_runner/*]  (7 Phases/Runner)
                                        |
                                        +--> [cli/main]  (8 Run Profile)
```                                        