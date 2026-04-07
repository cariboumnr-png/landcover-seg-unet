## 📁 Current Project Structure (showing two levels)

Dernière mise à jour : 2026-04-07

```
./src/landseg/
├── artifacts/                    [Unified artifact lifecycle & persistence]
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
│   ├── foundation/               # grid, domain, data-block configs
│   ├── pipeline/                 # ingest / prepare / train pipelines
│   ├── models/                   # model configuration
│   ├── trainer/                  # training runtime configuration
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
│   └── specification/            # DataSpecs construction
|
├── models/                       [Model architectures]
│   ├── backbones/                # UNet, UNet++, etc.
│   ├── multihead/                # multihead model framework
│   └── factory.py                # model construction
|
├── trainer/                      [Training system]
│   ├── components/               # losses, metrics, dataloaders
│   ├── engine/                   # core training loop
│   └── runner/                   # training orchestration
|
└── utils/                        [Shared utilities]
    ├── logger.py                 # project logging
    ├── funcs.py                  # hashing, JSON, and time helpers
    └── multip.py                 # multiprocessing helpers
  ```