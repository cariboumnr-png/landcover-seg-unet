# ADR-0065: Ergonomic Dataset Manifest Compilation Interface

**Status:** Accepted — Implemented<br>
**Date:** 2026-11-02 (Implemented: 2026-11-03)

---

## 1. Context

To run data harmonization (`data-harmonize`), users must provide a dataset manifest describing input geospatial rasters. Under the legacy implementation in `landseg.geopipe.harmonize.manifest`, compiling this manifest required assembling a root `manifest.json` indexing separate per-raster sidecar JSON files (e.g., `sample_sentinel2.json`, `sample_landcover.json`).

In practice, manually authoring and maintaining these dual-file JSON manifests introduced frequent user errors:

1. **Dual-File Redundancy & Desynchronization Traps**:
   - The user had to maintain both the root manifest array and individual sidecar JSON documents.
   - Raster `name` and `path` were duplicated in both files. Any minor mismatch between the root entry and sidecar metadata caused `compiler.py` to abort with `DatasetManifestError('Mismatch manifest name/data file path')`.
   - Managing $N$ input rasters required manually editing and keeping synchronized $N + 1$ JSON documents.
2. **Tedious, Rigid Syntax Requirements**:
   - `band_mapping`: Required 1-based, contiguous integer keys serialized as strings (`"1": "blue"`, `"2": "green"`). Hand-indexing 10+ bands for satellite imagery (e.g., Sentinel-2) was tedious and error-prone.
   - **Asymmetric Category Rules**:
     - `domain`: `schemes` had to be omitted or `null` (defining any scheme raised `ValueError`), while `categorical_specs` was mandatory.
     - `feature`: `categorical_specs` had to be `null`, while `schemes` strictly mapped names to band strings declared in `band_mapping`.
     - `label`: Required complex nested mappings (`index_base`, `num_cls`, `ignore_cls`, `reclass`, `reclass_name`) where reclass class IDs strictly fell within `[index_base, index_base + num_cls - 1]`.
3. **No Automatic Raster Inspection**:
   - Manifest authoring was completely detached from the rasters on disk. Users had to manually open external GIS tools to check band counts, nodata values, and bounds, with no automated verification against the physical GeoTIFF before harmonization runs.
4. **Machine-Specific Path Fragility**:
   - Manifests frequently encoded hardcoded absolute paths, preventing datasets from being shared across workstations, remote compute nodes, or CI/CD environments.

---

## 2. Decision

We have introduced a focused, ergonomic dataset manifest compilation interface that establishes a single source of truth for raw dataset manifests, eliminates dual-file synchronization hazards, and validates raster contracts upfront.

The solution consists of four primary components:
1. Complete retirement of legacy dual-file sidecars in favor of a **Unified Single-Document Manifest Specification** (JSON and YAML).
2. A **Fluent Python Builder API** (`DatasetManifestBuilder`).
3. **Native YAML Support** in `landseg.artifacts.Controller` with automatic relative path resolution.
4. A **User-Oriented Preprocessing Helper Notebook** (`notebooks/00_dataset_manifest.ipynb`).

### 2.1. Retirement of Dual-File Sidecars

We have completely removed the legacy dual-file sidecar pattern and retired `compiler.py`. The single file `manifest.json` (or human-readable `manifest.yaml`) directly contains all raster specifications. Managing $N$ input rasters now requires exactly 1 document.

Downstream harmonization validators (`harmonize.validators.manifest_contract`) and context builders (`build_harmonization_context`) consume the compiled manifest directly from `DatasetManifestBuilder.from_file(manifest_path).compile()`.

### 2.2. Fluent Python Builder API (`DatasetManifestBuilder`)

We have introduced `DatasetManifestBuilder` under `landseg.geopipe.harmonize.manifest.builder` to provide a type-safe, programmatic builder with IDE autocompletion and category-specific configuration methods:

- **Simplified Band Mapping**: Users provide an intuitive sequence of band names (`bands=['blue', 'green', 'red']`), which the builder automatically indexes into canonical 1-based dictionaries (`{1: 'blue', 2: 'green', 3: 'red'}`). Canonical presets such as `SensorPreset.SENTINEL_2` provide out-of-the-box standard band names.
- **Category-Specialized Methods**:
  - `add_feature(name, path, bands, schemes=None)`: Automatically sets `categorical_specs=None` and validates scheme references against declared bands.
  - `add_domain(name, path, num_cls, band_name=None, index_base=1, ignore_cls=None)`: Enforces `schemes=None` and constructs valid domain categorical specs.
  - `add_label(name, path, num_cls, band_name=None, index_base=1, ignore_cls=None, classes=None, color_map=None, taxonomy_profile=None, schemes=None)`: Simplifies reclassification syntax, automatically formatting matching `reclass` and `reclass_name` mappings, color maps, and taxonomy profile lookups.
- **In-Memory Contract Validation**: Calling `.compile()` validates all entries against `normalizer.py` schemas, checks physical file presence on disk, audits class index bounds, and guards against duplicate layer configurations.
- **Dual JSON/YAML Materialization with Relative Paths**: Calling `.materialize(save_dir, name)` exports both `manifest.json` and `manifest.yaml` while converting paths relative to `save_dir`, guaranteeing complete portability across environments.

### 2.3. Unified Single-Document Dataset Specification

The manifest supports both YAML and JSON single-document representations. It accepts either a top-level list of raster entries or a top-level mapping with a `rasters` sequence and optional `base_dir`:

```yaml
version: "1.0"
base_dir: "."

rasters:
  - name: sentinel2
    category: features
    path: rasters/sentinel2.tif
    band_mapping:
      1: blue
      2: green
      3: red
      4: red_edge1
      5: red_edge2
      6: red_edge3
      7: nir
      8: narrow_nir
      9: swir1
      10: swir2
    schemes:
      rgb: [blue, green, red]
      rgb_nir: [blue, green, red, nir]

  - name: geology
    category: domains
    path: rasters/geology.tif
    band_mapping:
      1: geology
    categorical_specs:
      index_base: 1
      num_cls: 4
      ignore_cls: [255]

  - name: landcover
    category: labels
    path: rasters/landcover.tif
    band_mapping:
      1: landcover
    categorical_specs:
      index_base: 1
      num_cls: 2
      ignore_cls: [255]
      class_name:
        1: coniferous
        2: deciduous
      color_map:
        1: [34, 139, 34]
        2: [218, 165, 32]
    schemes:
      binary:
        reclass:
          1: [1, 2]
        reclass_name:
          1: forest
```

`DatasetManifestBuilder.from_file(path)` parses this unified document, resolves relative paths against the manifest file location, and populates the builder instance.

### 2.4. Native YAML Support in Artifact Controller

We have extended `landseg.artifacts.Controller` to natively support reading and writing YAML files (`.yaml` and `.yml`):
- Uses `yaml.safe_load` for secure deserialization.
- Employs an isolated `_Dumper(yaml.SafeDumper)` in `_yaml_write` with multi-representer support for `pathlib.PurePath` instances, matching `json.dump(..., default=str)` semantics.
- Generates SHA256 integrity hash tracking records in `_hash.json` automatically.

### 2.5. User-Oriented Preprocessing Helper Notebook (`notebooks/00_dataset_manifest.ipynb`)

On par with [`notebooks/00_raster_mosaic.ipynb`](../../notebooks/00_raster_mosaic.ipynb), we have provided an interactive helper notebook for dataset manifest assembly:
- **Guided Raster Discovery**: Scans local directories or remote cloud mounts (e.g., Databricks Unity Catalog Volumes or DBFS) for raw candidate GeoTIFFs.
- **Visual Inspection & Band Verification**: Leverages `rasterio` to inspect raster headers (CRS, dimensions, band count, data type, nodata, resolution) before registering layers.
- **Interactive Builder Configuration**: Walks users through constructing their `DatasetManifestBuilder` configuration, selecting band schemes, and configuring label reclass tables.
- **Validation & Dual Materialization**: Compiles and writes `manifest.json` and human-readable `manifest.yaml`, establishing a clean, verified entrypoint for [`01_data_preparation.ipynb`](../../notebooks/01_data_preparation.ipynb) and harmonization pipelines.

---

## 3. Consequences

### Positive
- **Zero Synchronization Errors**: Dual-file sidecars are completely eliminated; the manifest document is the single source of truth.
- **Improved Ergonomics**: Simplifies 1-based band mappings and reclassification schemes into intuitive lists and dictionaries with IDE autocompletion.
- **Fail-Fast Upstream Validation**: Catches raster missing-file errors, band count mismatches, and index bounds violations before launching long ETL harmonization jobs.
- **Single-Document Portability**: Relative path serialization allows dataset directories to be copied across local workstations, remote nodes, and CI runners without path breakage.
- **Dual Representation**: Supports both machine-optimized JSON and version-controllable, human-readable YAML.
- **Interactive Visual Feedback**: The companion helper notebook provides instant visual confirmation without the overhead of a dedicated GUI application.

### Neutral / Trade-Offs
- Adds `pyyaml` dependency (standard in Python data science environments).
- Legacy projects using sidecar JSONs must migrate to single-document manifests (supported directly by `DatasetManifestBuilder.from_file()`).
