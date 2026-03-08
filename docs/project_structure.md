## 📁 Current Project Structure (Source‑first - collapsed to first class)

```
root/src/landseg
│
├── core/               # project-level contracts
│   └── protocols.py
|
├── grid/               # generate stable world grid
│   ├── builder.py      <- module API
│   ├── io.py
│   └── layout.py
|
├── domain/             # map domain rasters to world grid
│   ├── io.py
│   ├── mapper.py       <- module API
│   ├── tilemap.py
│   └── transform.py
|
├── dataprep/           # process raw rasters to stable artifacts
│   ├── blockbuilder/
│   ├── mapper/
│   ├── normalizer/
│   ├── splitter/
│   ├── utils/
│   ├── pipeline.py     <- module API
│   └── schema.py
│
├── dataset/            # consume data schema for traininig.dataloading
│   ├── specs.py
│   ├── loader.py       <- module API
│   └── validate.py
│
├── models/             # defines model structure (current: UNet, UNet++)
│   ├── backbones/
│   ├── multihead/
│   └── factory.py      <- module API
│
├── training/           # trainer and its components
│   ├── callback/
│   ├── common/
│   ├── dataloading/
│   ├── heads/
│   ├── loss/
│   ├── metrics/
│   ├── optim/
│   ├── trainer/
│   └── factory.py      <- module API
│
├── controller/         # build controller (experiment run from this)
│   ├── builder.py      <- module API
│   ├── controller.py
│   └── phases.py
│
├── utils/              # project-wide utilities
│
├── configs/            # hydra config tree shipped with package
│
└── cli/                # CLI scripts
    ├── main.py         <- module API
    ├── end_to_end.py
    └── overfit_test.py
  ```