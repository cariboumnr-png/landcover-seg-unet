## 📁 Current Project Structure (Source‑first - collapsed to first class)

```
root/src/landseg
│
├── core/                      # project-level protocols and shared contracts
│   ├── dataprep_schema.py
│   ├── dataset_specs.py
│   ├── grid_protocol.py
│   └── model_protocol.py
│
├── prep_grid/                 # factory: construct the stable world grid
│   └── builder.py             # ← module API
│
├── prep_domain/               # factory: process domain rasters aligned to grid
│   └── mapper.py              # ← module API
│
├── prep_raster/               # factory: process imagery + label rasters into artifacts
│   └── pipeline.py            # ← module API
│
├── data_schema/               # factory: assemble final training schema
│   └── builder.py             # ← module API
│
├── models/                    # model architectures and model factory
│   └── factory.py             # ← module API
│
├── training/                  # training engine and runtime components
│   └── factory.py             # ← module API
│
├── controller/                # experiment orchestration and controller phases
│   └── builder.py             # ← module API
│
├── utils/                     # project-wide utilities
│   ├── contxt.py
│   ├── funcs.py
│   ├── logger.py
│   ├── multip.py
│   ├── pca.py
│   └── preview.py
│
├── configs/                   # hydra configuration tree
│
└── cli/                       # CLI entry points
    └── main.py                # ← module API
  ```