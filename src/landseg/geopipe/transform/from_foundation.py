# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Catalog adapter utilities.

Provides helpers to load and filter a canonical blocks catalog and schema
to extract class counts and file paths needed for downstream sampling
and analysis.
'''

# standard imports
import dataclasses
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core

# typing aliases
CatalogDictCtrl = artifacts.Controller[dict[str, geo_core.CatalogEntry]]
SchemaCtrl = artifacts.Controller[geo_core.DataSchema]

@dataclasses.dataclass
class DataBlocksView:
    '''High-level view of data blocks from dev and optional test data.'''
    dev_base_class_counts: dict[tuple[int, int], list[int]]
    dev_valid_class_counts: dict[tuple[int, int], list[int]]
    dev_blocks: dict[tuple[int, int], str]
    external_test_blocks: list[str] | None

@dataclasses.dataclass
class _Parsed:
    '''Internal parsed representation of a blocks catalog.'''
    base_class_counts: dict[tuple[int, int], list[int]]
    valid_class_counts: dict[tuple[int, int], list[int]]
    valid_file_paths: dict[tuple[int, int], str]

def data_blocks_adapter(
    dev_catalog: str,
    dev_schema: str,
    test_catalog: str,
    *,
    valid_px_threshold: float,
    non_overlapping_test_grid: bool = True
):
    '''
    Load and adapt development and test blocks into a structured view.

    Filters blocks based on a minimum valid-pixel threshold, derives
    class counts, and optionally restricts test blocks to the base grid
    (non-overlapping).

    Args:
        dev_catalog: Path to development blocks catalog JSON.
        dev_schema: Path to dataset schema JSON.
        test_catalog: Path to external test blocks catalog JSON.
        valid_px_threshold: Minimum fraction/amount of valid pixels
            required for a block to be included.
        non_overlapping_test_grid: If True, restrict test blocks to the
            base non-overlapping grid aligned with schema block size.

    Returns:
        DataBlocksView containing filtered dev metadata and optional
        test blocks.
    '''

    # try load schema first
    data_schema = SchemaCtrl.load_json_or_fail(dev_schema).fetch()
    assert data_schema # typing assertion

    # get block size from schema
    image_shape = data_schema['tensor_shapes']['image']
    block_size = (image_shape['H'], image_shape['W'])

    # parse dev data catalog
    t = valid_px_threshold
    dev = _parse(dev_catalog, t, block_size)
    # try parse test data catalog
    try:
        test = _parse(test_catalog, t, block_size)
    except artifacts.ArtifactError:
        test = None

    # get test blocks
    if not test:
        test_blocks = None
    else:
    # whether to filter test blocks only on a non-overlapping grid (base)
        if non_overlapping_test_grid:
            test_blocks = list(
                v for k, v in test.valid_file_paths.items()
                if k in test.base_class_counts
            )
        else:
            test_blocks = list(test.valid_file_paths.values())

    # return
    return DataBlocksView(
        dev_base_class_counts=dev.base_class_counts,
        dev_valid_class_counts=dev.valid_class_counts,
        dev_blocks=dev.valid_file_paths,
        external_test_blocks=test_blocks
    )

def _parse(
    fpath: str,
    valid_px_threshold: float,
    block_size: tuple[int, int],
):
    '''Parse acatalog JSON into filtered class counts and file paths.'''

    # read catalog JSON to instantiate a class object
    catalog_dict = CatalogDictCtrl.load_json_or_fail(fpath).fetch()
    assert catalog_dict # typing assertion
    catalog = geo_core.DataCatalog.from_dict(catalog_dict)

    # all valid entries from catalog
    t = valid_px_threshold
    work_catalog = {k: v for k, v in catalog.items() if v['base_valid_px'] > t}
    catalog_counts = {k: v['base_class_count'] for k, v in work_catalog.items()}

    # entries on the base grid (no overlap)
    row_size, col_size = block_size
    base_catalog = {
        k: v for k, v in work_catalog.items()
        # both row and col are divisible
        if v['row_col'][0] % row_size == 0 and v['row_col'][1] % col_size == 0
    }
    base_counts = {k: v['base_class_count'] for k, v in base_catalog.items()}

    # all block file paths
    valid_file_paths = {k: v['file_path'] for k, v in work_catalog.items()}

    return _Parsed(
        base_class_counts=base_counts,
        valid_class_counts=catalog_counts,
        valid_file_paths=valid_file_paths
    )
