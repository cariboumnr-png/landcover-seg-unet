# ADR-0062: Centralized Rasters Subsystem and I/O Consolidation

**Status:** Proposed<br>
**Date:** 2026-10-23

---

## 1. Context

Following PR #80 (*Package Portability and Subsystem Decoupling*), the raster
processing utilities previously located in `landseg.geopipe.harmonize.rasters`
were elevated to a first-class subsystem at `landseg.geopipe.rasters`. This
migration cleanly decoupled general raster algorithms (`mask.py`, `mosaic.py`,
`stack.py`, `vrt.py`, and `warp.py`) from pipeline orchestration logic.

However, an audit of `landseg.geopipe` reveals that `rasterio` imports and
low-level raster operations remain fragmented across downstream modules:

1. **Vestigial and Underused Context Manager**:
   `landseg.geopipe.utils.raster_context` provides a single context manager
   `open_rasters(*rasters)`. Because it requires tuple unpacking even for a
   single raster (`with open_rasters(p) as (src,):`), developers routinely
   bypassed it and wrote raw `with rasterio.open(...)` blocks across
   `prepare`, `harmonize`, and `assembler/io.py`. Leaving this utility in
   `geopipe.utils` also breaks the encapsulation of the `rasters` subsystem.
2. **Read/Write Asymmetry in Raster Metadata and Tags**:
   While `rasters.vrt` provides utilities to *write* VRT metadata tags
   (`add_tag_to_vrt`) and band descriptions (`add_band_description_to_vrt`),
   the logic to *read* and decode these tags (`read_band_map`, `read_schemes`,
   `read_label_specs`, `_collect_feature_metadata`,
   `_collect_label_band_metadata`) is duplicated across
   `geopipe.ingest.blocks.assembler.io` and
   `geopipe.harmonize.processor.composition`.
3. **Repeated Spatial Reference Discovery**:
   Extracting fundamental raster spatial metadata (CRS, Affine transform,
   pixel resolution, bounds, nodata values) is reimplemented independently
   in `prepare.context`, `grid.builder`, `stack.py`, and `mosaic.py`.
4. **Scattered Spatial Geometry and Warping Bounds**:
   Spatial alignment checks (`_check_raster_proj`, `_check_raster_pixels`,
   `_compute_overlap_extent`) live in `ingest.blocks.mapper.geometry`, while
   bounds transformation (`rasterio.warp.transform_bounds`) is invoked directly
   inside `prepare.partition.operations.aoi`.
5. **Caller Leaks of `rasterio.DatasetReader`**:
   Functions such as `rasters.warp.write_warped_vrt` strictly require an
   already-opened `rasterio.DatasetReader`, forcing callers (such as
   `harmonize.processor.processor`) to manage raw `rasterio.open` context
   lifecycles rather than passing file paths directly.

---

## 2. Decision

We will centralize all `rasterio` interactions, virtual raster manipulations,
and generic spatial geometry checks into `landseg.geopipe.rasters`. The
`rasters` subsystem will establish a strictly unidirectional dependency: it
will depend solely on Python standard libraries and third-party geospatial
primitives (`numpy`, `rasterio`), and will be imported by downstream `geopipe`
subsystems (`grid`, `prepare`, `harmonize`, `ingest`).

### 2.1. Subsystem Layout

The `landseg.geopipe.rasters` module will be structured as follows:

```text
src/landseg/geopipe/rasters/
├── __init__.py          # Public exports (lazy-loaded via __getattr__)
├── geometry.py          # Generic spatial geometry checks, bounds, and overlaps
├── io.py                # Context managers, raster info extraction, windowed reads
├── mask.py              # Valid-pixel mask unification (existing)
├── metadata.py          # Band mappings, label specs, and scheme tag decoders
├── mosaic.py            # Warped VRT mosaicking (existing)
├── stack.py             # Multi-raster virtual band stacking (existing)
├── vrt.py               # GDAL VRT XML construction and tag mutation (existing)
└── warp.py              # Virtual warping, bounds reprojection, resampling enums
```

### 2.2. Consolidation of I/O Primitives (`rasters/io.py`)

We will create `landseg.geopipe.rasters.io` to absorb and retire
`geopipe.utils.raster_context`:

1. **Ergonomic Context Managers**:
   - `open_raster(path, mode='r')`: Context manager yielding a single
     `rasterio.DatasetReader` without tuple unpacking.
   - `open_rasters(*paths, mode='r')`: Context manager leveraging
     `contextlib.ExitStack` yielding a tuple of optional readers.
2. **One-Shot Spatial Metadata Reader (`RasterInfo`)**:
   We will introduce a lightweight frozen dataclass `RasterInfo`:
   ```python
   @dataclasses.dataclass(frozen=True)
   class RasterInfo:
       crs: str
       transform: rasterio.transform.Affine
       width: int
       height: int
       bounds: tuple[float, float, float, float]
       res: tuple[float, float]
       nodata: float | int | None
       count: int
       dtypes: tuple[str, ...]
       descriptions: tuple[str, ...]
   ```
   A new public function `get_raster_info(path: pathlib.Path) -> RasterInfo`
   will allow callers (`prepare.context`, `grid.builder`, `stack.py`,
   `mosaic.py`) to retrieve spatial parameters in a single call without
   holding open reader handles or importing `rasterio`.
3. **Windowed Array Extraction (`read_raster_window`)**:
   Provide a standardized helper for reading bounded or boundless windowed
   NumPy arrays with homogeneous type casting.

### 2.3. Symmetrical Metadata and Tag Inspection (`rasters/metadata.py`)

We will extract VRT/raster tag and description parsing from
`geopipe.ingest.blocks.assembler.io` and
`geopipe.harmonize.processor.composition` into `rasters.metadata`:

- `read_band_map(path: pathlib.Path) -> dict[str, int]`: Parse raster
  band descriptions into normalized zero-based band indices.
- `read_label_specs(path: pathlib.Path | None) -> dict[str, CategoricalSpec]`:
  Extract categorical class specifications from dataset and band tags.
- `read_schemes(path: pathlib.Path | None) -> dict[str, typing.Any]`:
  Extract embedded feature schemes from dataset and band metadata.
- `read_raster_tags(path: pathlib.Path, band: int | None = None) -> dict[str, str]`:
  Read raw GDAL metadata tags from the dataset or a specific band.

This establishes full symmetry with `rasters.vrt` (`add_tag_to_vrt`,
`add_band_description_to_vrt`).

### 2.4. Generic Spatial Geometry Primitives (`rasters/geometry.py`)

We will extract raster-agnostic spatial checks out of
`geopipe.ingest.blocks.mapper.geometry` and `prepare.partition.operations.aoi`:

- `check_raster_crs(crs_a: str, crs_b: str | None) -> str`: Verify that
  two coordinate reference systems match.
- `check_raster_resolution(transform_a, transform_b) -> tuple[float, float]`:
  Verify pixel resolution alignment within floating-point tolerance.
- `compute_overlap_bounds(bounds_a, bounds_b) -> tuple[float, float, float, float]`:
  Calculate bounding box intersection or raise on disjoint extents.
- `reproject_bounds(bounds, src_crs, dst_crs) -> tuple[float, float, float, float]`:
  Wrap `rasterio.warp.transform_bounds` to prevent downstream callers from
  needing `rasterio.warp` directly.

### 2.5. Warping API Path-Flexibility (`rasters/warp.py`)

We will update `write_warped_vrt` to accept `src: pathlib.Path | rasterio.DatasetReader`.
If a `Path` is provided, `write_warped_vrt` will manage the reader context
internally. This will eliminate boilerplate `with rasterio.open(...)` contexts
in `harmonize.processor.processor`.

### 2.6. Retirement of `geopipe.utils.raster_context`

1. We will deprecate and delete `src/landseg/geopipe/utils/raster_context.py`.
2. We will remove `open_rasters` from `src/landseg/geopipe/utils/__init__.py`.
3. Existing callers will be updated to import from `landseg.geopipe.rasters`.
4. Unit tests in `tests/unit/geopipe/utils/test_raster_context.py` will be
   migrated to `tests/unit/geopipe/rasters/test_io.py`.

---

## 3. Consequences

### Positive
- **Strict Boundary & Encapsulation**: `rasterio` will be quarantined inside
  `landseg.geopipe.rasters`. Downstream pipelines will consume high-level
  primitives (`RasterInfo`, `open_raster`, `read_schemes`, `reproject_bounds`)
  rather than manipulating raw GDAL handles.
- **Ergonomics**: Single-raster opening will no longer require tuple unpacking;
  one-shot metadata queries will no longer require context managers.
- **Elimination of Duplicate Logic**: Spatial reference discovery, tag parsing,
  and geometry overlap checks will have a single canonical implementation.
- **Test Isolation**: All raster I/O and GDAL XML logic will be thoroughly
  exercised in fast, dedicated unit tests under `tests/unit/geopipe/rasters/`.

### Neutral / Trade-offs
- Call sites across `prepare`, `grid`, `harmonize`, and `ingest` will require
  minor refactoring to import from `rasters`.
- `geopipe.core.grid_layout` can continue defining or re-exporting
  `RasterReader` and `RasterWindow` typing aliases for contract backward
  compatibility, but will source them consistently.

---

## 4. Implementation Plan

1. **Phase 1: Implement New `rasters` Modules**:
   - Create `src/landseg/geopipe/rasters/io.py` (`open_raster`, `open_rasters`,
     `get_raster_info`, `read_raster_window`).
   - Create `src/landseg/geopipe/rasters/metadata.py` (`read_band_map`,
     `read_label_specs`, `read_schemes`, `read_raster_tags`).
   - Create `src/landseg/geopipe/rasters/geometry.py` (`check_raster_crs`,
     `check_raster_resolution`, `compute_overlap_bounds`, `reproject_bounds`).
   - Update `rasters/warp.py` to accept `Path` in `write_warped_vrt`.
   - Expose new APIs in `src/landseg/geopipe/rasters/__init__.py`.
2. **Phase 2: Migrate Call Sites**:
   - Refactor `prepare.context` and `grid.builder` to use `get_raster_info`.
   - Refactor `prepare.partition.operations.aoi` to use `reproject_bounds`.
   - Refactor `harmonize.processor` and `composition.py` to use `metadata.py`
     and path-based `write_warped_vrt`.
   - Refactor `ingest.blocks.assembler.io` and `mapper.geometry` to use
     `rasters.metadata` and `rasters.geometry`.
3. **Phase 3: Retire `raster_context`**:
   - Delete `src/landseg/geopipe/utils/raster_context.py` and update
     `src/landseg/geopipe/utils/__init__.py`.
4. **Phase 4: Test Suite Migration & Validation**:
   - Move `test_raster_context.py` to `tests/unit/geopipe/rasters/test_io.py`.
   - Add unit tests for `rasters.metadata` and `rasters.geometry`.
   - Execute targeted test suites across `geopipe`.
