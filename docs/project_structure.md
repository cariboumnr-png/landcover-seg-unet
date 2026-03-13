## 📁 Current Project Structure (showing two levels)

```
./src/landseg/
├── cli/                          [CLI scripts]
│   ├── main.py                   <- CLI entry point
│   ├── end_to_end.py             # default profile
│   └── overfit_test.py           # overfit profile
|
├── configs/                      [Hydra-based config tree]
│   ├── inputs/
│   ├── models/
│   ├── prep/
│   ├── profile/
│   ├── runner/
│   ├── trainer/
│   ├── config.yaml               # Hydra configs entry point
│   └── schema.py                 # dataclass schema with runtim validation
|
├── core/                         [Project level contracts]
│   ├── ingest_protocols/         # protocols for data schema/domain/data specs
│   ├── trainer_protocols/        # protocols for trainer componenents/engine
│   ├── alias.py                  # type aliases
│   ├── grid_protocol.py          # protocol for grid layout
│   └── model_protocol.py         # protocol for multihead model framework
|
├── grid_generator/               [Generates immutable world grids]
│   ├── generator.py              <- public module interface
│   ├── io.py                     # grid artifacts save/load
│   └── layout.py                 # grid definition
|
├── ingest_dataset/               [Ingest image/label rasters]
│   ├── blockbuilder/             # build data blocks artifacts (.npz files)
│   ├── mapper/                   # map input raster to world grid
│   ├── normalizer/               # normalize image arrays across data blocks
│   ├── schema/                   # dataset schema JSON definition
│   ├── splitter/                 # split data blocks into train/val etc.
│   ├── config.py                 # module runtime config
│   └── pipeline.py               <- public module interface
|
├── ingest_domain/                [Ingest domain knowledge rasters]
│   ├── io.py                     # domain artifacts save/load
│   ├── mapper.py                 # map domain rasters to world grid
│   ├── tilemap.py                # domain map definition
│   └── transform.py              # domain vector PCA transform utilities
|
├── ingest_specs/                 [Generates data specs for trainer dataloader]
│   ├── builder.py                # data spec building functions
│   ├── factory.py                <- public module interface
│   └── validate.py               # validate present schema/prompt to run ingest
|
├── models/                       [Model architectures]
│   ├── backbones/                # currently implemented: UNet, UNet++
│   ├── multihead/                # multihead model framework
│   └── factory.py                <- public module interface
|
├── trainer_components/           [Build trainier runtime components]
│   ├── callback/                 # callback system
│   ├── dataloading/              # get dataloaders, e.g., train/val/test
│   ├── heads/                    # get specs for each training head
│   ├── loss/                     # config loss compute for each head
│   ├── metrics/                  # config metrics compute for each head
│   ├── optimization/             # config optimizer & scheduler
│   └── factory.py                <- public module interface
|
├── trainer_engine/               [Core training loop/stage definitions]
│   ├── utils/                    # utilities, e.g., checkpointing
│   ├── engine_config.py/         # runtime configuration
│   ├── engine_state.py/          # runtime state
│   └── engine.py                 <- public module interface
|
├── trainer_runner/               [Training orchestrator]
│   ├── phase.py/                 # curriculum phases
│   └── runner.py                 <- public module interface
|
└── utils/                        [Project-wide utilities]
    ├── context.py                # rasterio open raster context
    ├── funcs.py                  # useful conveniences
    ├── logger.py                 # custom project logger
    └── multip.py                 # multiprocessing wrapper
  ```