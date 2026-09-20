# Raster Data Preparation Guide

[English](./data_preparation.md) | [Français](./data_preparation_fr.md)

Last updated: 2026-08-19

---

## Overview

Effective geospatial machine-learning workflows begin with careful and
consistent data preparation. The `landseg` framework uses a deterministic,
artifact-driven pipeline architecture consisting of four sequential data
stages:

1. **`world-grid`**: Generates and persists the canonical spatial tiling
   layout (CRS, origin, resolution, tile dimensions, and stride).
2. **`data-harmonize`**: Warps, reprojects, and stacks raw multi-source
   rasters into unified VRT canvases aligned to the world grid.
3. **`data-ingest`**: Tiles harmonized rasters into unpartitioned canonical
   `.npz` data blocks and materializes domain maps.
4. **`data-prepare`**: Partitions data blocks into train/val/test splits
   (optionally constrained by geographic AOI masks) and calculates band
   normalization statistics.

Users can supply their own raw GeoTIFFs along with a dataset manifest, or use
the provided synthetic data generation script to explore the framework.

---

## Contents

- [Synthetic Sample Data Generator](#synthetic-sample-data-generator)
- [World Grid & Extent Reference](#world-grid--extent-reference)
- [Dataset Manifest & Metadata Configs](#dataset-manifest--metadata-configs)
  - [Root Dataset Manifest (`manifest.json`)](#root-dataset-manifest-manifestjson)
  - [Feature Rasters (Optical / DEM)](#feature-rasters-optical--dem)
  - [Label Rasters (Classification Heads)](#label-rasters-classification-heads)
  - [Domain Rasters (Optional Context)](#domain-rasters-optional-context)
  - [Geographic AOI Rasters (Optional Splits)](#geographic-aoi-rasters-optional-splits)
- [Automated vs. Manual Alignment](#automated-vs-manual-alignment)
- [Tutorial - Create Reference Rasters in QGIS](#tutorial---create-reference-rasters-in-qgis)

---

## Synthetic Sample Data Generator

To test the entire pipeline locally without custom data, or to inspect the
exact file naming conventions and JSON schemas, generate synthetic sample
rasters using:

```bash
python scripts/generate_dummy_data.py --output_dir ./experiment/input -y
```

This generates a self-contained dataset under `./experiment/input/`:

```text
experiment/input/
├── reference_raster/
│   ├── sample_extent.tif        # Project extent defining spatial canvas
│   └── sample_test_aoi.tif      # Optional geographic test holdout mask
└── raw_data/
    ├── manifest.json            # Central dataset manifest
    ├── sample_dev_sentinel2.tif # Multi-band optical imagery (10m)
    ├── sample_dev_sentinel2.json
    ├── sample_dev_dem.tif       # Elevation feature raster (10m)
    ├── sample_dev_dem.json
    ├── sample_dev_landcover.tif # Land-cover segmentation ground truth
    ├── sample_dev_landcover.json
    ├── sample_dev_leadspc.tif   # Tree species segmentation ground truth
    ├── sample_dev_leadspc.json
    ├── sample_domain_1.tif      # Categorical domain context mask
    ├── sample_domain_1.json
    └── ...
```

---

## World Grid & Extent Reference

The world grid establishes the spatial anchor for the entire project. It must
be defined in a projected coordinate reference system (e.g. `EPSG:3161`) so
that tile coordinates and pixel dimensions correspond to linear units (metres).

The grid is defined via a **reference raster** (e.g., `sample_extent.tif`):
- Supplies the CRS, pixel size, bounding extent, and top-left origin.
- Serves as the spatial canvas for raster harmonization and block tiling.

Run the world grid pipeline:

```bash
landseg pipeline=world-grid
```

This outputs a canonical grid artifact under:
`experiment/artifacts/world_grids/grid_row_<size>_<stride>_col_<size>_<stride>.json`

---

## Dataset Manifest & Metadata Configs

The data harmonization pipeline (`data-harmonize`) reads a central
`manifest.json` file. Each raw raster referenced in the manifest is
accompanied by an individual raster manifest JSON file (sidecar).

### Root Dataset Manifest (`manifest.json`)

The manifest is an array of objects mapping each raw raster to its raster
manifest:

```json
[
  {
    "name": "sentinel2",
    "path": "./experiment/input/raw_data/sample_dev_sentinel2.tif",
    "manifest": "./experiment/input/raw_data/sample_dev_sentinel2.json"
  },
  {
    "name": "dem",
    "path": "./experiment/input/raw_data/sample_dev_dem.tif",
    "manifest": "./experiment/input/raw_data/sample_dev_dem.json"
  },
  {
    "name": "landcover",
    "path": "./experiment/input/raw_data/sample_dev_landcover.tif",
    "manifest": "./experiment/input/raw_data/sample_dev_landcover.json"
  },
  {
    "name": "domain_1",
    "path": "./experiment/input/raw_data/sample_domain_1.tif",
    "manifest": "./experiment/input/raw_data/sample_domain_1.json"
  }
]
```

---

### Feature Rasters (Optical / DEM)

Feature rasters represent continuous inputs (satellite bands, elevation,
radar). Their JSON config defines 1-based band mappings and optional named
feature schemes:

**Example: Multi-Band Optical (`sample_dev_sentinel2.json`)**
```json
{
  "name": "sentinel2",
  "path": "./experiment/input/raw_data/sample_dev_sentinel2.tif",
  "category": "features",
  "band_mapping": {
    "1": "blue",
    "2": "green",
    "3": "red",
    "4": "red_edge1",
    "5": "red_edge2",
    "6": "red_edge3",
    "7": "nir",
    "8": "narrow_nir",
    "9": "swir1",
    "10": "swir2"
  },
  "categorical_specs": null,
  "schemes": {
    "rgb": ["blue", "green", "red"],
    "rgb_nir": ["blue", "green", "red", "nir"],
    "surface": ["blue", "green", "red", "nir", "swir1", "swir2"]
  }
}
```

**Example: Digital Elevation Model (`sample_dev_dem.json`)**
```json
{
  "name": "dem",
  "path": "./experiment/input/raw_data/sample_dev_dem.tif",
  "category": "features",
  "band_mapping": {
    "1": "dem"
  },
  "categorical_specs": null,
  "schemes": null
}
```

---

### Label Rasters (Classification Heads)

Label rasters contain integer class IDs for model supervision. Multiple label
rasters can be supplied to train multi-task networks (e.g. land cover and
leading tree species).

**Example: Land-Cover Target (`sample_dev_landcover.json`)**
```json
{
  "name": "landcover",
  "path": "./experiment/input/raw_data/sample_dev_landcover.tif",
  "category": "labels",
  "band_mapping": {
    "1": "landcover"
  },
  "categorical_specs": {
    "index_base": 1,
    "num_cls": 3,
    "ignore_cls": [255],
    "class_name": {
      "1": "coniferous",
      "2": "deciduous",
      "3": "water"
    },
    "color_map": {
      "1": [34, 139, 34],
      "2": [218, 165, 32],
      "3": [0, 0, 255]
    }
  },
  "schemes": {
    "binary": {
      "reclass": {
        "1": [1, 2],
        "2": [3]
      },
      "reclass_name": {
        "1": "forest",
        "2": "water"
      }
    }
  }
}
```

#### `categorical_specs` Schema Fields
- **`index_base`** (*required, `int`*): Base index of raw raster classes (typically `0` or `1`).
- **`num_cls`** (*required, `int`*): Total number of active prediction classes
  (excluding ignore indices).
- **`ignore_cls`** (*required, `list[int]`*): List of raw pixel values treated
  as background/ignore during loss computation and metric evaluation.
- **`class_name`** (*optional, `dict[str, str]`*): Mapping from stringified
  class ID to human-readable class name (e.g. `{"1": "coniferous"}`).
- **`color_map`** (*optional, `dict[str, list[int]]`*): Mapping from
  stringified class ID to RGB color triples `[R, G, B]` (values 0–255). This
  color map is carried through ingestion into the dataset schema and used by
  session tracking callbacks for rendering visual prediction overlays.
- **`taxonomy`** (*optional, `dict[str, str]`*): Optional mapping linking
  local class names to standard ecological taxonomy keys.

#### `schemes` Schema Fields
- **Features**: A dictionary mapping scheme names (e.g. `"rgb"`, `"rgb_nir"`)
  to lists of band names. All band names must match `band_mapping.values()`.
- **Labels**: A dictionary mapping scheme names (e.g. `"binary"`, `"coarse"`)
  to reclassification dictionaries containing `reclass` (parent-to-child ID
  groupings) and `reclass_name` (human-readable parent group labels).
- **Domains**: Must be `null` (or empty).

---

### Domain Rasters (Optional Context)

Domain rasters provide categorical contextual masks (e.g., eco-districts,
administrative boundaries, watershed zones) used for model conditioning or
stratification.

**Example: Domain Context (`sample_domain_1.json`)**
```json
{
  "name": "domain_1",
  "path": "./experiment/input/raw_data/sample_domain_1.tif",
  "category": "domains",
  "band_mapping": {
    "1": "domain_1"
  },
  "categorical_specs": {
    "index_base": 1,
    "num_cls": 4,
    "ignore_cls": [255],
    "class_name": {
      "1": "EcoDistrict_1",
      "2": "EcoDistrict_2",
      "3": "EcoDistrict_3",
      "4": "EcoDistrict_4"
    }
  },
  "schemes": null
}
```

---

### Preparation Stage Configuration (`configs/user.yaml`)

During `data-prepare`, users can select specific feature channels and target
reclassification hierarchies:

```yaml
data-prepare:
  output_dpath: ./experiment/artifacts/prepared_data

  # 1. Feature selection by dataset name (null uses all bands)
  features:
    sentinel2: rgb_nir          # Reference named scheme from sidecar
    dem: all                    # Use all bands from DEM
    # Or inline:
    # sentinel2: [blue, green, red, nir]

  # 2. Target reclassification by label name (null uses canonical base classes)
  targets:
    landcover: binary           # Reference named scheme from sidecar
    leadspc: raw                # Use raw/base classes
    # Or inline:
    # landcover:
    #   reclass:
    #     "1": [1, 2]
    #   reclass_name:
    #     "1": "forest"

  # 3. Spatial AOI and split ratios
  val_ratio: 0.20
  test_ratio: 0.00
  buffer_step: 0
  rebuild: true
```

---

### Geographic AOI Rasters (Optional Splits)

In addition to manifest-listed inputs, users can provide external Area of
Interest (AOI) rasters to deterministically isolate evaluation regions:

- `sample_test_aoi.tif`: Blocks overlapping this raster are allocated to the
  `test` split.
- Configured in `configs/user.yaml` under `data-prepare: test_aoi: ...`

---

## Automated vs. Manual Alignment

### Automated Pipeline Alignment (Recommended)
You do not need to pre-align or resample your raw GeoTIFFs to a common CRS or
pixel grid. The `data-harmonize` pipeline automatically:
1. Re-projects raw rasters to the target world grid CRS (e.g. `EPSG:3161`).
2. Resamples continuous features using `bilinear` and categorical labels using
   `nearest`.
3. Stacks aligned features and labels into canonical `.vrt` files.
4. Generates an active pixel mask (`valid_pixel_mask.vrt`).

Execute harmonization:
```bash
landseg pipeline=data-harmonize
```

---

## Tutorial - Create Reference Rasters in QGIS

If you are creating custom extent reference rasters or AOI masks manually:

### Step 1 — Select Projected CRS
1. Open QGIS.
2. In the bottom-right corner, click the CRS button.
3. Select your local projected coordinate system (e.g., `EPSG:3161`).

### Step 2 — Create Bounding Extent Layer
1. Go to **Layer → Create Layer → New Shapefile Layer**.
2. Set Geometry Type to **Polygon**.
3. Draw a polygon bounding the full study area (training + prediction zones).
4. Save as `project_extent.shp`.

### Step 3 — Rasterize the Extent
1. Go to **Raster → Conversion → Rasterize (Vector to Raster)**.
2. Select `project_extent.shp` as input.
3. Set fixed pixel value to `1`.
4. Set output resolution (e.g., `20` metres).
5. Save output as `sample_extent.tif`.