## 📁 Current Project Structure (showing two levels)

Last updated : 2026-05-12

```

./notebooks/
│   ├── 01_data_preparation.ipynb # data ingest, validation, and preparation pipelines
│   └── 02_model_train.ipynb      # demo: end-to-end continuous training session
|
./src/landseg/
├── adapters/                     [External entry interfaces]
│   ├── api.py                    <- programmatic API entry point
│   └── cli.py                    <- Hydra-driven CLI entry point
│
├── artifacts/                    [Artifact lifecycle management & persistence]
│   ├── checkpoint.py             # model checkpoint save/load utilities
│   ├── controller.py             # policy-driven artifact resolve/build/rebuild
│   ├── paths.py                  # canonical artifact path definitions
│   ├── payload_io.py             # structured payload + metadata serialization
│   └── policy.py                 # artifact lifecycle rules and policies
|
├── configs/                      [Hydra configuration tree]
│   ├── _schema_/                 # structured config dataclasses (sectional)
│   ├── dataspecs/                # dataset specification configs
│   ├── foundation/               # grid/domain/block configuration
│   ├── models/                   # model architecture configs
│   ├── pipeline/                 # ingest / prepare / train pipeline configs
│   ├── session/                  # runtime session configuration
│   ├── transform/                # data transform definitions
│   ├── schema.py                 # composite config schema
│   └── config.yaml               # Hydra config entry point
|
├── core/                         [Project-wide contracts & abstractions]
│   ├── data_specs.py             # runtime dataset specification contract
│   ├── model_protocol.py         # model interface / behavioral contract
│   └── session_results.py        # structured session output/results dataclasses
|
├── execution/                    [Execution orchestration layer]
│   ├── pipelines/                # explicit, stage-based pipeline implementations
│   ├── executor.py               # unified execution entry point
│   └── resolver.py               # config resolution and dependency binding
|
├── geopipe/                      [Geospatial data preparation pipeline]
│   ├── core/                     # immutable core geospatial data structures
│   ├── foundation/               # grid, domain, and block construction
│   ├── specification/            # DataSpecs construction and assembly
│   ├── transform/                # experiment-scoped data transformations
│   └── utils/                    # geopipeline-level utilities
|
├── models/                       [Model architectures & composition]
│   ├── backbones/                # backbone networks (e.g., UNet, UNet++)
│   ├── multihead/                # multi-head modeling framework
│   └── factory.py                # model construction and wiring
|
├── session/                      [Runtime execution system]
│   ├── commmon/                  # shared types and internal contracts
│   ├── data/                     # dataloader and batching adapters
│   ├── engine/                   # batch- and epoch-level execution engines
│   ├── instrumentation/          # callbacks, logging, tracking, preview tools
│   ├── orchestration/            # session lifecycle orchestration
│   ├── factory.py                # session construction/builders
│   └── metadata.py               # session metadata and tracking
|
├── study/                        [Experimentation & study layer]
│   ├── analysis/                 # post-sweep evaluation and analysis
│   └── sweep/                    # Optuna-based hyperparameter sweeping
|
├── utils/                        [Shared utilities]
│   ├── logger.py                 # project-level logging utilities
│   └── multip.py                 # multiprocessing helpers
|
└── _constants.py                 # global constants and shared definitions

```