```
root/src/landseg
│
├── grid/               # generate stable world grid
│   ├── builder.py          <- module API
│   ├── io.py               # grid artifacts I/O
│   └── layout.py           # grid dictionary definition
│
├── domain/             # map domain rasters to world grid
│   ├── io.py               # domain artifacts I/O
│   ├── mapper.py           <- module API
│   ├── tilemap.py          # domain dictionary definition
│   └── transform.py        # PCA utilities to transform domain to vectors
│
├── dataprep/           # process raw rasters to stable artifacts
│   ├── blockbuilder/       # data block building
|   │   ├── block.py            # data block definition
|   │   ├── builder.py          <- module API
|   │   └── cache.py            # block builder pipeline
│   ├── mapper/
│   ├── normalizer/
│   ├── splitter/
│   ├── utils/
│   ├── pipeline.py         <- module API
│   └── schema.py
│
├── dataset/            # wire data schema to trainer dataloading
│   ├── builder.py      <- module API
│   ├── load.py
│   └── validate.py
│
├── models/             # defines model structure (current: UNet, UNet)
│   ├── backbones/
│   ├── multihead/
│   └── factory.py      <- module API
│
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
│
├── controller/         # build controller (experiment run from it)
│   ├── builder.py      <- module API
│   ├── controller.py
│   └── phases.py
│
├── utils/              # project utilities
│
├── configs/            # hydra config tree shipped with package
│
└── cli/                # CLI scripts
    └── end_to_end.py   <- primary entrypoint for `experiment_run`
  ```