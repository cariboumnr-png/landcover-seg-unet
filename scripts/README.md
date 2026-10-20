# Project Executable Scripts

This directory contains executable entrypoint scripts for embedding generation, dummy testing data creation, and Databricks/VM pipeline execution.

---

## 📜 Script Index & Usage Examples

### 1. `build_species_embeddings.py`
**Description:**
Encodes tree species ecological profiles from CSV knowledge bases using Sentence Transformers (default: `BAAI/bge-base-en-v1.5`). Generates PyTorch L2-normalized embedding tensors (`[N, D]`), cosine similarity matrices (`[N, N]`), class metadata JSONs, and human-readable similarity CSVs into subfolders named after the source CSV under `knowledge/embeddings/`.

**Usage Examples:**

```powershell
# 1. Generate embeddings for 29 Ecological Groups (Default)
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py

# 2. Generate embeddings for Full 138 FRI Species List
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py `
    --csv-path knowledge/ontario_tree_species_profiles.csv

# 3. Specify custom HuggingFace model or output directory
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py `
    --csv-path knowledge/ontario_tree_species_grouped_profiles.csv `
    --output-dir knowledge/embeddings `
    --model-name BAAI/bge-base-en-v1.5
```

---

### 2. `generate_dummy_data.py`
**Description:**
Generates synthetic geospatial raster datasets (GeoTIFFs representing multi-spectral imagery, DEM, DSM, and segmentation masks) in `./experiment/input` for testing UNet training pipelines locally without needing full satellite rasters.

**Usage Example:**

```powershell
.\.venv\Scripts\python.exe scripts/generate_dummy_data.py
```

---

### 3. `run.py`
**Description:**
Bootstrapping runner script designed for Databricks job clusters, cloud VMs, and containerized pipeline execution. Dynamically resolves the workspace `src/` directory and invokes the `landseg` command-line interface (CLI).

**Usage Example:**

```powershell
# Run the pipeline CLI via the runner script
.\.venv\Scripts\python.exe scripts/run.py --help
```

---

### 4. `build_raster_mosaic.py`
**Description:**
Harmonizes and assembles multiple disjoint or adjacent raster tiles (such as Ontario Crown Forest Management Units across differing UTM zones) into a single virtual mosaic VRT (`.vrt`) aligned to provincial standard CRS (`EPSG:3161`) without materializing intermediate imagery to disk.

**Usage Examples:**

```powershell
# 1. Mosaic adjacent GeoTIFF tiles into a single EPSG:3161 Virtual Raster
.\.venv\Scripts\python.exe scripts/build_raster_mosaic.py `
    data/fmu_tiles/*.tif `
    --output experiment/input/feature/mosaic_sentinel2.vrt

# 2. Mosaic categorical classification masks with nearest-neighbor resampling
.\.venv\Scripts\python.exe scripts/build_raster_mosaic.py `
    data/fmu_labels/*.tif `
    --output experiment/input/label/mosaic_labels.vrt `
    --categorical `
    --relative-paths

# 3. Specify target resolution (e.g. 20m) and custom resampling algorithm
.\.venv\Scripts\python.exe scripts/build_raster_mosaic.py `
    data/dem_tiles/*.tif `
    --output experiment/input/feature/mosaic_dem.vrt `
    --target-resolution 20.0 `
    --resampling cubic
```

