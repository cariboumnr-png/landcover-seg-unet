# ADR-0065: Ergonomic Dataset Manifest Compilation Interface

**Status:** Proposed<br>
**Date:** 2026-11-02

---

## 1. Context

To run data harmonization (`data-harmonize`), users must provide a dataset manifest describing input geospatial rasters. Under the current implementation in `landseg.geopipe.harmonize.manifest`, compiling this manifest requires assembling a root `manifest.json` indexing separate per-raster sidecar JSON files (e.g., `sample_sentinel2.json`, `sample_landcover.json`).

In practice, manually authoring and maintaining these JSON manifests introduces frequent user errors:

1. **Dual-File Redundancy & Desynchronization Traps**:
   - The user must maintain both the root manifest array and individual sidecar JSON documents.
   - Raster `name` and `path` are duplicated in both files. Any minor mismatch between the root entry and sidecar metadata causes `compiler.py` to immediately abort with `DatasetManifestError('Mismatch manifest name/data file path')`.
   - Managing $N$ input rasters requires manually editing $N + 1$ JSON documents.
2. **Tedious, Rigid Syntax Requirements**:
   - `band_mapping`: Requires 1-based, contiguous, integer keys serialized as strings (`"1": "blue"`, `"2": "green"`). Hand-indexing 10+ bands for satellite imagery (e.g., Sentinel-2) is tedious and error-prone.
   - **Asymmetric Category Rules**:
     - `domain`: `schemes` must be omitted or `null` (defining any scheme raises `ValueError`), but `categorical_specs` is mandatory.
     - `feature`: `categorical_specs` must be `null`, while `schemes` must strictly map names to band strings declared in `band_mapping`.
     - `label`: Requires complex nested mappings (`index_base`, `num_cls`, `ignore_cls`, `reclass`, `reclass_name`) where reclass class IDs must strictly fall within `[index_base, index_base + num_cls - 1]`.
3. **No Automatic Raster Inspection**:
   - Manifest authoring is detached from the rasters on disk. Users must manually open external GIS tools to check band counts, nodata values, and bounds, with no automated verification against the GeoTIFF before harmonization runs.
4. **Machine-Specific Path Fragility**:
   - Manifests frequently encode hardcoded absolute paths, preventing datasets from being shared across workstations, remote compute nodes, or CI/CD environments.

---

## 2. Decision

We will introduce a focused, ergonomic dataset manifest compilation interface that provides a single source of truth for raw dataset manifests, eliminates dual-file synchronization hazards, and validates raster contracts upfront.

The solution will consist of four focused components:
1. A **Fluent Python Builder API** (`DatasetManifestBuilder`).
2. A **Unified Single-Document Manifest Specification** (YAML/JSON).
3. Focused **CLI Diagnostic and Compilation Utilities** (`landseg manifest`).
4. A **User-Oriented Preprocessing Helper Notebook** (`notebooks/00_dataset_manifest.ipynb`).

### 2.1. Fluent Python Builder API (`DatasetManifestBuilder`)

We will introduce `DatasetManifestBuilder` under `landseg.geopipe.harmonize.manifest.builder` to provide a type-safe, programmatic builder with IDE autocompletion:

- **Simplified Band Mapping**: Users will provide a simple list of band names (`bands=['blue', 'green', 'red']`), which the builder will automatically index into canonical 1-based dictionaries (`{1: 'blue', 2: 'green', 3: 'red'}`).
- **Category-Specialized Methods**:
  - `add_feature(name, path, bands, schemes=None)`: Automatically sets `categorical_specs=None` and validates scheme references against declared bands.
  - `add_domain(name, path, band_name, num_cls, index_base=1, ignore_cls=None)`: Enforces `schemes=None` and builds valid domain categorical specs.
  - `add_label(name, path, band_name, num_cls, index_base=1, ignore_cls=None, classes=None, color_map=None, schemes=None)`: Simplifies reclassification syntax, automatically generating matching `reclass` and `reclass_name` mappings.
- **One-Call Synchronized Compilation**:
  Calling `.compile_bundle(output_dir)` will validate all entries and write both the root `manifest.json` and all sidecar JSONs simultaneously, guaranteeing 100% path and name synchronization.
- **Relative Path Resolution**: Paths will be resolved relative to a configurable base directory, ensuring portability.

### 2.2. Unified Single-Document Dataset Specification

We will define a single-file document schema (supporting both YAML and JSON) that consolidates all raster entries into a single human-readable file:

```yaml
version: "1.0"
base_dir: "."

rasters:
  - name: sentinel2
    category: feature
    path: rasters/sentinel2.tif
    bands: [blue, green, red, red_edge1, red_edge2, red_edge3, nir, narrow_nir, swir1, swir2]
    schemes:
      rgb: [blue, green, red]
      rgb_nir: [blue, green, red, nir]

  - name: geology
    category: domain
    path: rasters/geology.tif
    band_name: geology
    categorical:
      index_base: 1
      num_cls: 4
      ignore_cls: [255]

  - name: landcover
    category: label
    path: rasters/landcover.tif
    band_name: landcover
    categorical:
      index_base: 1
      num_cls: 2
      ignore_cls: [255]
      classes:
        1: coniferous
        2: deciduous
    schemes:
      binary:
        forest: [1, 2]
```

A dedicated compiler function `compile_single_manifest(spec_path, output_dir)` will parse this single document and generate the canonical bundle expected by the harmonization pipeline.

### 2.3. Raster-Aware Pre-Validation & Inspection

We will integrate lightweight `rasterio` checks into the builder and compiler:
- Verify that referenced files exist on disk before generating manifest files.
- Verify that the declared band count matches the number of bands physically present in the GeoTIFF.
- Inspect raster metadata (CRS, data type, nodata) to alert users to potential discrepancies early.

### 2.4. Focused CLI Commands (`landseg manifest`)

We will expose targeted subcommands under the CLI entrypoint:
- `landseg manifest validate <manifest_path>`: Audits a manifest file, validating paths, schema rules, and raster band counts.
- `landseg manifest inspect <raster.tif>`: Probes a GeoTIFF and outputs a recommended manifest entry snippet.
- `landseg manifest compile <spec.yaml> -o <output_dir>`: Converts a unified single-document YAML/JSON specification into the harmonization bundle.

### 2.5. User-Oriented Preprocessing Helper Notebook (`notebooks/00_dataset_manifest.ipynb`)

On par with [`notebooks/00_raster_mosaic.ipynb`](../../notebooks/00_raster_mosaic.ipynb), we will provide an interactive, user-facing helper notebook for dataset manifest assembly:
- **Guided Raster Discovery**: Scans local directories or remote cloud storage mounts (e.g., Databricks Unity Catalog Volumes or DBFS) for raw candidate GeoTIFFs.
- **Visual Inspection & Band Verification**: Leverages `rasterio` and `matplotlib` to render quick RGB false-color band previews, check CRS/transform alignments, and inspect categorical value distributions.
- **Interactive Builder Configuration**: Walks users through constructing their `DatasetManifestBuilder` configuration, selecting band schemes, and configuring label reclass tables.
- **Preflight Validation & One-Click Compilation**: Validates the manifest against actual rasters on disk and compiles the final root `manifest.json` and sidecars, establishing a clean, verified entrypoint for [`01_data_preparation.ipynb`](../../notebooks/01_data_preparation.ipynb) and harmonization pipelines.

---

## 3. Consequences

### Positive
- **Zero Synchronization Errors**: Eliminates manual copy-pasting of names and paths between root manifests and sidecars.
- **Improved Ergonomics**: Simplifies 1-based band mappings and reclassification schemes into intuitive lists and dictionaries.
- **Fail-Fast Validation**: Catches raster missing-file errors, band count mismatches, and index bounds violations before launching harmonization.
- **Interactive Visual Feedback**: The companion helper notebook provides instant visual confirmation (band previews, class palettes) without the complexity or maintenance overhead of a dedicated GUI application.
- **Full Backward Compatibility**: Downstream harmonization ETL (`run_data_harmonization`) and `compile_dataset_manifest` will consume the exact same canonical `ManifestEntry` structures without modification.

### Neutral / Trade-Offs
- Adds a small set of builder utilities and CLI subcommands in the harmonization manifest package.
- YAML parsing will require `pyyaml` (already standard in Python data science environments).

