src/
  core/                       # foundation: types, config, logging, errors
    __init__.py
    types.py                  # shared dataclasses (TileID, BBox, Flags)
    config.py                 # load/merge runtime config (OmegaConf bridge)
    logging.py
    errors.py

  specs/                      # first-class, code-level contracts (loaders/validators)
    __init__.py
    grid_spec.py              # GridSpec (YAML → dataclass; validate)
    domain_schema.py          # DomainSchema (YAML → dataclass; validate)
    task_manifest.py          # TaskManifest (YAML → dataclass; validate)

  grid/                       # pure geometry & tiling (no labels, no training)
    __init__.py
    builder.py                # tiles from GridSpec (row/col, bbox, geom)
    ids.py                    # stable tile_id naming from (row, col, version)
    index.py                  # AOI → tile index selection; edge rules
    io.py                     # read/write grid products (GeoParquet/GeoJSON)

  domain/                     # domain extraction & transforms (PCA), no labels
    __init__.py
    resolver.py               # compute per-tile domain features from rasters
    transforms.py             # PCA projection, normalization (no fitting here)
    artifacts.py              # load PCA artifacts (mu, components, stats)
    ood.py                    # OOD checks (range, Mahalanobis), flags

  tasks/                      # binding layer (no training required for v1)
    __init__.py
    compose.py                # grid + domain orchestration for domain prep
    registry.py               # optional: discover available grids/domains

  data_prep/                  # imagery/labels materialization (optional now)
    __init__.py
    imagery.py                # resample/stack imagery onto tiles
    labels.py                 # (later) rasterize/collect labels by tile
    cache.py                  # caching by tile_id (COG/Zarr/NPZ/Parquet)

  models/                     # keep current structure
    __init__.py
    backbones/
    multihead/
    factory.py

  training/                   # keep current structure
    __init__.py
    dataloading/
    callback/
    metrics/
    loss/
    trainer/
    factory.py

  adapters/                   # bridges to legacy paths (non-breaking migration)
    __init__.py
    dataset_compat.py         # legacy dataset API → calls grid/domain/tasks
    domain_compat.py          # fallback to compute domain inline if no cache

  evaluation/                 # (optional) metrics/plots for domain/grid sanity
    __init__.py
    domain_reports.py
    grid_reports.py

  cli/                        # small CLIs to run contracts without notebooks
    __init__.py
    grid_cli.py               # “build grid from spec”
    domain_cli.py             # “compute domain from task”
    task_cli.py               # “inspect/validate task”

  utils/                      # keep general utilities
    __init__.py
    contxt.py
    logger.py
    multip.py
    preview.py
    pca.py                    # (thin): consider delegating to domain/transforms

  # tests/                    # recommend top-level tests next to src/, not inside it