## Current workflow
```
[configs/]
└─> grid/builder.py                    (1 World Grid)
    ├─> domain/mapper.py                   (2 DK → grid, optional)
    └─> dataprep/pipeline.py               (3 Fit/Test → grid)
        └─> dataprep/schema.py                 (4 Data Scheme)
            ├─> models/factory.py                  (5.1 Model)
            ├─> dataset/builder.py                 (5.2 Dataloaders)
            ├─> training/heads                     (5.3 Data-influenced)
            ├─> training/loss                      (5.4 Head-specified)
            ├─. training/metrics                   (5.5 Head-specified)
            └─> training/optim|callback|...        (5.6 Other)
                └─> training/factory.py → Trainer      (6 Trainer 🗸)
                    └─> controller/builder.py + phases     (7 Controller 🗸)
                        └─> cli/main.py                    (8 Start ➝)
```