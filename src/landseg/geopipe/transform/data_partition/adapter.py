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
Catalog adapter.
'''

# standard imports
import dataclasses
# local imports
import landseg.geopipe.core as core

@dataclasses.dataclass
class ParsedCatalog:
    '''Information parsed from the data blocks catalog.'''
    base_class_counts: dict[tuple[int, int], list[int]]
    valid_class_counts: dict[tuple[int, int], list[int]]
    valid_file_paths: dict[tuple[int, int], str]

def parse_catalog(
    fpath: str,
    block_size: tuple[int, int]
):
    '''Load data blocks catalog.'''

    # read catalog JSON to instantiate a class object
    catalog = core.BlocksCatalog.from_json(fpath)

    # all valid entries from catalog
    work_catalog = {k: v for k, v in catalog.items() if v['valid_px']}
    catalog_counts = {k: v['class_count'] for k, v in work_catalog.items()}

    # entries on the base grid (no overlap)
    row_size, col_size = block_size
    base_catalog = {
        k: v for k, v in work_catalog.items()
        # both row and col are divisible
        if v['row_col'][0] % row_size == 0 and v['row_col'][1] % col_size == 0
    }
    base_counts = {k: v['class_count'] for k, v in base_catalog.items()}

    # all block file paths
    valid_file_paths = {k: v['file_path'] for k, v in work_catalog.items()}

    return ParsedCatalog(
        base_class_counts=base_counts,
        valid_class_counts=catalog_counts,
        valid_file_paths=valid_file_paths
    )
