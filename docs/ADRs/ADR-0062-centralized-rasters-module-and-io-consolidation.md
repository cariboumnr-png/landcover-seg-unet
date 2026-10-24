# ADR-0062: Centralized Rasters Subsystem and I/O Consolidation

**Status:** Accepted — Implemented<br>
**Date:** 2026-10-23

---

## 1. Context

Following PR #80 (*Package Portability and Subsystem Decoupling*), the raster
processing utilities previously located in `landseg.geopipe.harmonize.rasters`
were elevated to a first-class subsystem at `landseg.geopipe.rasters`. This
migration cleanly decoupled general raster algorithms (`mask.py`, `mosaic.py`,
`stack.py`, `vrt.py`, and `warp.py`) from pipeline orchestration logic.

However, an audit of `landseg.geopipe` revealed that `rasterio` imports and
low-level raster operations remained fragmented across downstream modules:

1. **Vestigial and Underused Context Manager**:
   `landseg.geopipe.utils.raster_context` provided a single context manager
   `open_rasters(*rasters)`. Because it required tuple unpacking even for a
   single raster (`with open_rasters(p) as (src,):`), developers routinely
   bypassed it and wrote raw `with rasterio.open(...)` blocks across
   `prepare`, `harmonize`, and `assembler/io.py`. Leaving this utility in
   `geopipe.utils` also broke the encapsulation of the `rasters` subsystem.
2. **Read/Write Asymmetry in Raster Metadata and Tags**:
   While `rasters.vrt` provided utilities to *write* VRT metadata tags
   (`add_tag_to_vrt`) and band descriptions (`add_band_description_to_vrt`),
   the logic to *read* and decode these tags was duplicated across
   `geopipe.ingest.blocks.assembler.io` and
   `geopipe.harmonize.processor.composition`.
3. **Repeated Spatial Reference Discovery**:
   Extracting fundamental raster spatial metadata (CRS, Affine transform,
   pixel resolution, bounds, nodata values) was reimplemented independently
   in `prepare.context`, `grid.builder`, `stack.py`, and `mosaic.py`.
4. **Scattered Spatial Geometry and Warping Bounds**:
   Spatial alignment checks (`_check_raster_proj`, `_check_raster_pixels`,
   `_compute_overlap_extent`) lived in `ingest.blocks.mapper.geometry`, while
   bounds transformation was invoked directly inside
   `prepare.partition.operations.aoi`.
5. **Caller Leaks of `rasterio.DatasetReader`**:
   Functions such as `rasters.warp.write_warped_vrt` strictly required an
   already-opened `rasterio.DatasetReader`, forcing callers (such as
   `harmonize.processor.processor`) to manage raw `rasterio.open` context
   lifecycles rather than passing file paths directly.

---

## 2. Decision

We centralized all `rasterio` interactions, virtual raster manipulations,
and generic spatial geometry checks into `landseg.geopipe.rasters`. The
`rasters` subsystem establishes a strictly unidirectional dependency: it
depends solely on Python standard libraries and third-party geospatial
primitives (`numpy`, `rasterio`), and is imported by downstream `geopipe`
subsystems (`grid`, `prepare`, `harmonize`, `ingest`).

### 2.1. Subsystem Layout

The `landseg.geopipe.rasters` module is structured as follows:

```text
src/landseg/geopipe/rasters/
├── __init__.py          # Public exports (lazy-loaded via __getattr__)
├── _types.py            # Normalized rasterio type aliases (Affine, BoundingBox, CRS, etc.)
├── geometry.py          # Generic spatial geometry checks, bounds, and reprojection
├── io.py                # Context managers, DatasetInfo extraction, windowed reads
├── mask.py              # Valid-pixel mask unification
├── mosaic.py            # Warped VRT mosaicking
├── stack.py             # Multi-raster virtual band stacking
├── vrt.py               # GDAL VRT XML construction, tag mutation, and parse_vrt_tag
└── warp.py              # Virtual warping, resampling enums, and CRS equality
```

### 2.2. Consolidation of I/O Primitives (`rasters/io.py`)

We created `landseg.geopipe.rasters.io` to absorb and retire
`geopipe.utils.raster_context`:

1. **Ergonomic Context Managers**:
   - `read_raster(path)`: Context manager yielding a single dataset reader
     without tuple unpacking.
   - `read_rasters(*paths)`: Context manager leveraging `contextlib.ExitStack`
     yielding a tuple of optional readers.
2. **One-Shot Spatial Metadata Reader (`DatasetInfo`)**:
   We introduced a lightweight frozen dataclass `DatasetInfo` and helper
   `get_info(source)` returning comprehensive metadata (CRS, Affine transform,
   dimensions, bounding box, resolution, nodata value, count, dtypes, band
   descriptions, and GDAL tags) in a single call without holding open handles.
   It also exposes a `band_map` property mapping normalized descriptions to
   zero-based band indices.
3. **Windowed Array Extraction (`read_window`)**:
   Standardized helper for reading windowed NumPy arrays with boundless
   padding or fill value substitution.

### 2.3. Domain Boundary Purity in Tag Parsing

During implementation, we explicitly preserved the architectural purity of
`rasters` as a low-level engine:

- We rejected introducing domain-specific schemas (`CategoricalSpec`, feature
  schemes) into `rasters`.
- Low-level tag deserialization was centralized in `rasters.vrt` as
  `parse_vrt_tag(value)`.
- Band mapping was attached directly to `DatasetInfo.band_map`.
- Domain metadata extraction (`read_label_specs`, `read_schemes`) remains
  in consumer pipeline modules (`assembler/io.py`, `composition.py`),
  consuming `rasters.get_info()` and `rasters.parse_vrt_tag()`.

### 2.4. Generic Spatial Geometry Primitives (`rasters/geometry.py`)

We extracted generic spatial geometry checks into `landseg.geopipe.rasters.geometry`:

- `check_raster_crs(crs_a, crs_b=None) -> str`: Verify coordinate reference
  system presence and alignment.
- `check_raster_resolution(transform_a, transform_b=None) -> tuple[float, float]`:
  Verify pixel resolution alignment within floating-point tolerance.
- `compute_overlap_bounds(bounds_a, bounds_b=None) -> BoundingBox`:
  Calculate bounding box intersection or raise on disjoint extents.
- `reproject_bounds(bounds, src_crs, dst_crs) -> BoundingBox`:
  Reproject structured bounding box coordinates to target CRS.

The thin parameter-splatting `transform_bounds` wrapper in `warp.py` was
retired in favor of structured `reproject_bounds`.

### 2.5. Warping API Path-Flexibility (`rasters/warp.py`)

We updated `write_warped_vrt` to accept `src: pathlib.Path | str | DatasetReader`.
When given a file path, `write_warped_vrt` manages the reader context
internally, eliminating boilerplate `with rasterio.open(...)` contexts in
downstream processors.

### 2.6. Retirement of `geopipe.utils.raster_context`

1. Deprecated and deleted `src/landseg/geopipe/utils/raster_context.py`.
2. Removed `open_rasters` from `src/landseg/geopipe/utils/__init__.py`.
3. Migrated unit tests to `tests/unit/geopipe/rasters/test_io.py`.

---

## 3. Consequences

### Positive
- **Strict Quarantine**: Direct `rasterio` imports across all production
  pipelines are 100% isolated inside `landseg.geopipe.rasters`.
- **Domain Purity**: Low-level raster mechanics remain strictly decoupled
  from domain schemas (`CategoricalSpec`, label schemes).
- **Ergonomics**: Single-raster opening requires no tuple unpacking; one-shot
  metadata queries require no open contexts.
- **Single Source of Truth**: Geometry overlap checks, tag parsing, and
  spatial metadata inspection are unified across the codebase.
- **Test Isolation**: All raster I/O, geometry, and GDAL XML logic are
  exercised in dedicated unit tests under `tests/unit/geopipe/rasters/`.

### Neutral / Trade-offs
- Downstream callers import from `landseg.geopipe.rasters` rather than
  calling raw geospatial packages directly.

---

## 4. Implementation Summary

1. **Phase 1: Subsystem Modules**:
   - Implemented `rasters.io` (`read_raster`, `read_rasters`, `get_info`,
     `read_window`, `DatasetInfo`).
   - Implemented `rasters.geometry` (`check_raster_crs`,
     `check_raster_resolution`, `compute_overlap_bounds`, `reproject_bounds`).
   - Enhanced `rasters.vrt` with `parse_vrt_tag`.
   - Updated `rasters.warp.write_warped_vrt` with path-flexibility.
2. **Phase 2: Call Site Migrations**:
   - Refactored `prepare.context` and `grid.builder` to use `rasters.get_info`.
   - Refactored `prepare.partition.operations.aoi` to use `reproject_bounds`.
   - Refactored `harmonize.processor` and `composition.py` to use `io.get_info`
     and path-based `write_warped_vrt`.
   - Refactored `ingest.blocks.assembler.io` to use `vrt.parse_vrt_tag`,
     and dissolved `mapper.geometry` directly into `mapper.py`.
   - Refactored `rasters.stack` and `rasters.mosaic` to eliminate ad-hoc
     spatial reference containers in favor of `io.DatasetInfo`.
3. **Phase 3: Retirement of `raster_context`**:
   - Deleted `src/landseg/geopipe/utils/raster_context.py`.
4. **Phase 4: Validation**:
   - Unit test suite added in `tests/unit/geopipe/rasters/` (`test_io.py`,
     `test_geometry.py`, `test_vrt.py`, `test_mosaic.py`, `test_warp.py`,
     `test_raster_ops.py`), achieving 100% pass rate.

