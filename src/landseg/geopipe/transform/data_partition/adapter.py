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

Provides helpers to load and filter a canonical blocks catalog and
extract class counts and file paths needed for downstream sampling
and analysis.
'''

# standard imports
import dataclasses
# local imports
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.core as geo_core

@dataclasses.dataclass
class ParsedCatalog:
    '''Parsed view of a blocks catalog with commonly used subsets.'''
    dev_base_class_counts: dict[tuple[int, int], list[int]]
    dev_valid_class_counts: dict[tuple[int, int], list[int]]
    dev_blocks: dict[tuple[int, int], str]
    external_test_blocks: list[str] | None

@dataclasses.dataclass
class _Parsed:
    '''Parsed view of a blocks catalog with commonly used subsets.'''
    base_class_counts: dict[tuple[int, int], list[int]]
    valid_class_counts: dict[tuple[int, int], list[int]]
    valid_file_paths: dict[tuple[int, int], str]

def parse_catalog(
    dev_fpath: str,
    test_fpath: str,
    schema_fpath: str,
    *,
    valid_px_threshold: float,
):
    '''doc'''

    # try load meta first
    data_schema: geo_core.DataSchema
    load_status, m, data_schema = artifacts.load_json_hash(schema_fpath)
    if load_status: # non-zero status indicates false catalog.json
        raise ValueError('INFO', f'Data schema JSON loading error: {m}')
    # get block size from schema
    image_shape = data_schema['tensor_shapes']['image']
    block_size = (image_shape['H'], image_shape['W'])

    # parse dev data catalog
    t = valid_px_threshold
    dev = _parse(dev_fpath, t, block_size)
    # try parse test data catalog
    try:
        test = _parse(test_fpath, t, block_size)
    except ValueError:
        test = None
    test_blocks = list(test.valid_file_paths.values()) if test else None

    # return
    return ParsedCatalog(
        dev_base_class_counts=dev.base_class_counts,
        dev_valid_class_counts=dev.valid_class_counts,
        dev_blocks=dev.valid_file_paths,
        external_test_blocks=test_blocks
    )

def _parse(
    fpath: str,
    valid_px_threshold: float,
    block_size: tuple[int, int]
):
    '''
    Parse a canonical blocks catalog and extract usable block metadata.

    Loads the catalog JSON, filters out blocks without valid base pixels,
    derives class counts for all valid blocks and for base grid blocks
    (non-overlapping), and collects file paths for valid block artifacts.

    Args:
        fpath: Path to the blocks catalog JSON.
        block_size: Block size (rows, cols) used to identify base grid
            tiles.

    Returns:
        ParsedCatalog containing class counts and file paths.
    '''

    # read catalog JSON to instantiate a class object
    load_status, m, catalog_dict = artifacts.load_json_hash(fpath)
    catalog = geo_core.DataCatalog.from_dict(catalog_dict)
    if load_status: # non-zero status indicates false catalog.json
        raise ValueError('INFO', f'Catalog JSON loading error: {m}')

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
