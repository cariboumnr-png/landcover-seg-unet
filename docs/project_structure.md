## 📁 Current Project Structure (showing two levels)

Dernière mise à jour : 2026-04-07

```
./src/landseg/
├── artifacts/                    [Unified artifact lifecycle & persistence]
│   ├── checkpoint.py             # model checkpointing utilities
│   ├── controller.py             # policy-driven load/build/rebuild logic
│   ├── payload_io.py             # structured payload + metadata I/O
│   ├── paths.py                  # canonical artifact path layout
│   └── policy.py                 # artifact lifecycle policies
|
├── cli/                          [CLI entry point and pipeline dispatch]
│   ├── main.py                   <- Hydra-based CLI entry
│   └── pipelines/                # explicit pipeline stages
|
├── configs/                      [Hydra configuration tree]
│   ├── dataspecs/                # data specification
│   ├── foundation/               # grid, domain, data-block configs
│   ├── models/                   # model configuration
│   ├── pipeline/                 # ingest / prepare / train pipelines
│   ├── session/                  # session configuration
│   ├── transform/                # data transform configs
│   ├── config.yaml               # hydra config tree entry point
│   └── schema.py                 # config dataclass validation
|
├── core/                         [Project-level contracts]
│   ├── data_specs.py             # runtime dataset specification
│   └── model_protocol.py         # model interface contracts
|
├── geopipe/                      [Geospatial data preparation pipeline]
│   ├── core/                     # immutable core data structures
│   ├── foundation/               # grid, domain, block preparation
│   ├── transform/                # experiment-scoped data transforms
│   ├── specification/            # DataSpecs construction
│   └── utils/                    # module level utilities
|
├── models/                       [Model architectures]
│   ├── backbones/                # UNet, UNet++, etc.
│   ├── multihead/                # multihead model framework
│   └── factory.py                # model construction
|
├── session/                      [runtime system]
│   ├── commmon/                  # shared types in the module
│   ├── components/               # losses, metrics, dataloaders
│   ├── engine/                   # batch and policy engines
│   ├── instrumentation/          # callbacks, preview utility
│   ├── orchestration/            # session orchestration
│   ├── factory.py                # session builders
│   └── metadata.py               # session metadata
|
├── utils/                        [Shared utilities]
|   ├── logger.py                 # project logging
|   └── multip.py                 # multiprocessing helpers
|
└── _constants.py                 # project wide constants
  ```