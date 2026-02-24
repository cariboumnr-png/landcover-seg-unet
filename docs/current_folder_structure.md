```
root/src/landseg
├── grid/               # generate stable world grid
│   ├── builder.py      <- module API
│   ├── io.py
│   └── layout.py
├── domain/             # mapp domain rasters to world grid
│   ├── io.py
│   ├── mapper.py       <- module API
│   ├── tilemap.py
│   └── transform.py
├── dataprep/           # process raw rasters to stable artifacts
│   ├── blockbuilder/
│   ├── mapper/
│   ├── normalizer/
│   ├── splitter/
│   ├── utils/
│   ├── pipeline.py     <- module API
│   └── schema.py
├── dataset/            # wire data schema to trainer dataloading
│   ├── builder.py      <- module API
│   ├── load.py
│   └── validate.py
├── models/             # defines model structure (current: UNet, UNet++)
│   ├── backbones/
│   ├── multihead/
│   └── factory.py      <- module API
├── training/           # build trainer
│   ├── callback/
│   ├── common/
│   ├── dataloading/
│   ├── heads/
│   ├── loss/
│   ├── metrics/
│   ├── optim/
│   ├── trainer/
│   └── factory.py      <- module API
├── controller/         # build controller (experiment run from it)
│   ├── builder.py      <- module API
│   ├── controller.py
│   └── phases.py
├── utils/              # project utilities
├── configs/            # hydra config tree shipped with package
└── cli/
    └── end_to_end.py   <- previously root/main.py
  ```