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
Utilities for maintaining dataset catalogs and block-level metadata.

This module supports creation and incremental updates of
block-structured dataset catalogs (`catalog.json`) by inspecting block
artifact files and their embedded metadata. It records provenance links
to source imagery and labels, computes file-level integrity hashes, and
standardizes spatial and statistical descriptors.

Public APIs:
    - build_catalog: Builds or updates a dataset-level catalog.
'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.utils as geo_utils


# ----- public functions
def build_catalog(
    input_block_fpaths: list[str],
    *,
    original_catalog: geo_core.DatasetCatalog,
    mapped_grid_id: str,
    source_image: str,
    source_label: str | None,
) -> geo_core.DatasetCatalog:
    '''
    Build a new dataset catalog or update an existing one.

    Constructs a `DatasetCatalog` from a collection of block files.
    For each provided block artifact (`*.npz`), it loads embedded
    metadata, extracts spatial indices, computes hash checksums, and
    records provenance back to the source image and optional label.

    Args:
        input_block_fpaths:
            List of file paths to block artifacts to be cataloged.
        original_catalog:
            Existing `DatasetCatalog`. May be empty if creating a new
            catalog from scratch.
        mapped_grid_id:
            Identifier of aligned spatial grid used to generate the
            blocks.
        source_image:
            File path or identifier of source image raster.
        source_label:
            File path or identifier of source label raster, or `None`
            for unlabeled datasets.

    Returns:
        geo_core.DatasetCatalog:
            Catalog containing merged catalog entries for all provided
            blocks and any pre-existing catalog records.
    '''
    # return dict
    new_entries: dict[str, geo_core.DatasetBlockMeta] = {}

    # get hash values from input rasters
    img_hash = artifacts.Controller.get_sha256(source_image)
    if source_label:
        lbl_hash = artifacts.Controller.get_sha256(source_label)
    else:
        lbl_hash = None

    # add to current catalog dict
    for fp in input_block_fpaths:
        ctrl = artifacts.Controller(fp) # with no policy here
        meta = geo_core.DataBlock.load(fp).manifest
        row, col = geo_utils.name_xy(meta['block_name'])
        entry: geo_core.DatasetBlockMeta = {
            'block_name': meta['block_name'],
            'file_path': os.path.abspath(fp), # use absolute path
            'row_col': [row, col],
            'valid_px_ratios': meta['valid_ratios'],
            'class_count': meta['label_count'],
            'schema_version': '1.0.0',
            'creation_time': ctrl.creation_time,
            'sha_256': ctrl.sha256,
            'aligned_grid': mapped_grid_id,
            'source_image': source_image,
            'source_image_sha_256': img_hash,
            'source_label': source_label,
            'source_label_sha_256': lbl_hash,
        }
        new_entries[meta['block_name']] = entry

    # if the current catalog is not empty, append
    if original_catalog and len(original_catalog) > 0:
        originals = {v['block_name']: v for v in original_catalog.values()}
        new_entries = {**originals, **new_entries}
    # otherwise just create from input entries
    return geo_core.DatasetCatalog.from_dict(new_entries)
