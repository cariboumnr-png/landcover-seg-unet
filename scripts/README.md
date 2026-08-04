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
