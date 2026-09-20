# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Catalog adapter utilities for dataset preparation.

Provides helpers to load and filter canonical blocks catalogs and schemas,
extracting class counts and spatial coordinates needed for downstream
sampling, partitioning, and analysis.

Public APIs:
    - DataBlocksView: high-level view of data blocks for partitioning.
    - read_catalog: load and adapt canonical blocks into a structured view.
'''

# standard imports
import dataclasses
import typing
# third-party imports
import rasterio
import rasterio.errors
import rasterio.transform
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core


# ----- typing aliases
field = dataclasses.field
CatalogDictCtrl = (artifacts.Controller[dict[str, geo_core.DatasetBlockMeta]])


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class DataBlocksView:
    '''High-level view of data blocks for partitioning.'''
    valid_blocks: dict[tuple[int, int], str]
    external_test_blocks: list[str] | None
    crs: str
    transform: rasterio.transform.Affine
    raw_class_counts: dict[tuple[int, int], dict[str, list[int]]] = field(default_factory=dict)
    valid_class_counts: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    base_class_counts: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    focal_head: str = ''


# ----- public functions
def read_catalog(
    catalog_fpath: str,
    dataset_schema: geo_core.DatasetSchema,
    *,
    valid_pxs: typing.Mapping[str, float] | None = None,
    focal_target: str | None = None,
    test_catalog: str | None = None,
    non_overlapping_test_grid: bool = True,
    canvas_crs: str | None = None,
    canvas_transform: rasterio.transform.Affine | None = None,
) -> DataBlocksView:
    '''
    Load and adapt canonical blocks into a structured view.

    Filters blocks based on a minimum valid-pixel threshold, derives
    class counts, and optionally incorporates external holdout test
    blocks.

    Args:
        catalog_fpath:
            path to canonical blocks catalog JSON.
        dataset_schema:
            ingested dataset schema instance.
        valid_pxs:
            optional mapping of target head to valid pixel threshold.
        focal_target:
            optional target head name for partition stratification.
        test_catalog:
            optional path to external holdout test catalog JSON.
        non_overlapping_test_grid:
            whether to restrict external test blocks to non-overlapping.
        canvas_crs:
            optional pre-resolved CRS string of the dataset canvas.
        canvas_transform:
            optional pre-resolved Affine transform of the canvas.

    Returns:
        DataBlocksView:
            filtered metadata and block mappings for partitioning.
    '''
    # retrieve image paths and shape from schema
    image_paths = dataset_schema['dataset']['data_source']['image_paths']
    image_shape = dataset_schema['tensor_shapes']['image']

    # resolve canvas crs and transform from image source if not provided
    if canvas_crs is None or canvas_transform is None:
        try:
            with rasterio.open(image_paths[0]) as src:
                if src.crs is None:
                    raise ValueError(f'Raster has no CRS: {image_paths[0]}')
                canvas_crs = src.crs.to_string()
                canvas_transform = src.transform
        except rasterio.errors.RasterioError as e:
            raise ValueError(f'Error reading image: {image_paths[0]}') from e

    assert canvas_crs is not None
    assert canvas_transform is not None

    # valid blocks filtered by pixel thresholds
    valid_blocks = _filter_blocks(catalog_fpath, valid_pxs)

    # blocks on base grid (no stride/overlap)
    row_size, col_size = image_shape['H'], image_shape['W']
    base_coords = [
        k for k, v in valid_blocks.items()
        if v['row_col'][0] % row_size == 0 and v['row_col'][1] % col_size == 0
    ]

    # parse external test data catalog if provided
    if test_catalog is not None:
        test_blocks = _filter_blocks(test_catalog, valid_pxs)
        if non_overlapping_test_grid:
            test_blocks = list(
                v['file_path'] for k, v in test_blocks.items()
                if k in base_coords
            )
        else:
            test_blocks = list(
                v['file_path'] for v in test_blocks.values()
            )
    else:
        test_blocks = None

    # raw class counts from catalog entries
    raw_counts = {k: v['class_count'] for k, v in valid_blocks.items()}

    # preliminary focal head derivation from config or catalog entry
    focal_head = focal_target or ''
    if not focal_head and valid_blocks: # fallback to 1st available head
        first_entry = next(iter(valid_blocks.values()))
        if 'class_count' in first_entry and first_entry['class_count']:
            focal_head = next(iter(first_entry['class_count'].keys()))

    valid_counts: dict[tuple[int, int], list[int]] = {}
    base_counts: dict[tuple[int, int], list[int]] = {}
    if focal_head:
        valid_counts = {
            k: v['class_count'][focal_head] for k, v in valid_blocks.items()
            if focal_head in v.get('class_count', {})
        }
        base_counts = {
            k: v['class_count'][focal_head] for k, v in valid_blocks.items()
            if k in base_coords and focal_head in v.get('class_count', {})
        }

    return DataBlocksView(
        valid_blocks={k: v['file_path'] for k, v in valid_blocks.items()},
        external_test_blocks=test_blocks,
        crs=canvas_crs,
        transform=canvas_transform,
        raw_class_counts=raw_counts,
        valid_class_counts=valid_counts,
        base_class_counts=base_counts,
        focal_head=focal_head,
    )


# ----- private helpers
def _filter_blocks(
    fpath: str,
    valid_px_thresholds: typing.Mapping[str, float] | None = None,
) -> dict[tuple[int, int], geo_core.DatasetBlockMeta]:
    '''Parse catalog JSON into filtered class counts and file paths.'''
    thresholds = valid_px_thresholds or {}

    def _is_valid_block(
        valid_thresholds: typing.Mapping[str, float],
        valid_ratios: dict[str, float]
    ) -> bool:
        for k, v in valid_ratios.items():
            threshold = valid_thresholds.get(k)
            if threshold and v < threshold:
                return False
        return True

    catalog_dict = CatalogDictCtrl.load_json_or_fail(fpath).fetch()
    catalog = geo_core.DatasetCatalog.from_dict(catalog_dict)

    valid_catalog = {
        k: v for k, v in catalog.items()
        if _is_valid_block(thresholds, v['valid_px_ratios'])
    }
    return valid_catalog
