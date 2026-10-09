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
Data blocks normalizer utilities for dataset preparation.

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
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core


# ----- typing aliases
field = dataclasses.field
CatalogDictCtrl = (artifacts.Controller[dict[str, geo_core.DatasetBlockMeta]])
Coord = tuple[int, int]


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class DataBlocksManifestInputs:
    '''Bundled datablocks manifest artifacts.'''
    schema: geo_core.DatasetSchema
    catalog: dict[str, geo_core.DatasetBlockMeta]
    test_catalog: dict[str, geo_core.DatasetBlockMeta] | None = None


@dataclasses.dataclass(frozen=True)
class NormalizedDataBlocksView:
    '''High-level view of data blocks for partitioning.'''
    valid_blocks: dict[Coord, str]
    external_test_blocks: list[str] | None
    raw_class_counts: dict[Coord, dict[str, list[int]]] = field(default_factory=dict)
    valid_class_counts: dict[Coord, list[int]] = field(default_factory=dict)
    base_class_counts: dict[Coord, list[int]] = field(default_factory=dict)


# ----- public functions
def read_dataset_manifest(
    inputs: DataBlocksManifestInputs,
    *,
    head_pxs_thres: typing.Mapping[str, float] | None = None,
    focal_head: str | None = None,
    non_overlapping_test_grid: bool = True,
) -> NormalizedDataBlocksView:
    '''
    Load and adapt canonical blocks into a structured view.

    Filters blocks based on a minimum valid-pixel threshold, derives
    class counts, and optionally incorporates external holdout test
    blocks.

    Args:
        schema:
            ingested dataset schema instance.
        catalog:
            ingested canonical blocks catalog.
        test_catalog:
            optional path to external holdout test catalog JSON.
        head_pxs_thres:
            optional mapping of target head to valid pixel threshold.
        focal_head:
            optional target head name for partition stratification.
        non_overlapping_test_grid:
            whether to restrict external test blocks to non-overlapping.

    Returns:
        DataBlocksView:
            filtered metadata and block mappings for partitioning.
    '''
    # valid blocks filtered by pixel thresholds
    valid_blocks = _filter_blocks(inputs.catalog, head_pxs_thres)

    # blocks on base grid (no stride/overlap)
    image_shape = inputs.schema['tensor_shapes']['image']
    row_size, col_size = image_shape['H'], image_shape['W']
    base_coords = [
        k for k, v in valid_blocks.items()
        if v['row_col'][0] % row_size == 0 and v['row_col'][1] % col_size == 0
    ]

    # parse external test data catalog if provided
    if inputs.test_catalog is not None:
        test_blocks = _filter_blocks(inputs.test_catalog, head_pxs_thres)
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
    focal_head = focal_head or ''
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

    return NormalizedDataBlocksView(
        valid_blocks={k: v['file_path'] for k, v in valid_blocks.items()},
        external_test_blocks=test_blocks,
        raw_class_counts=raw_counts,
        valid_class_counts=valid_counts,
        base_class_counts=base_counts,
    )


# ----- private helpers
def _filter_blocks(
    catalog_dict: dict[str, geo_core.DatasetBlockMeta],
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

    catalog = geo_core.DatasetCatalog.from_dict(catalog_dict)
    valid_catalog = {
        k: v for k, v in catalog.items()
        if _is_valid_block(thresholds, v['valid_px_ratios'])
    }
    return valid_catalog
