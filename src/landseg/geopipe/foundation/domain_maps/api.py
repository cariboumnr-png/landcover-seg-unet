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
Tools for preparing domain knowledge mapped onto a world grid.

This module provides the public API for constructing per-domain
DomainTileMap objects. It coordinates the following operations:

    - Align the provided world grid to each domain raster using the
        grid's pixel-offset calculation.
    - Load an existing DomainTileMap artifact if present, validating its
        schema and integrity.
    - Otherwise, build a new DomainTileMap by reading all grid windows
        from the categorical raster, filtering tiles by valid-pixel
        fraction, computing majority statistics, deriving normalized
        frequency vectors, and projecting them onto PCA components to
        reach a target explained variance.
    - Persist each new DomainTileMap as a JSON payload and a metadata
        sidecar including schema_id, context, hash, and grid association.

Configuration is supplied via a structured dictionary containing:
    - 'dirpath': directory with domain rasters.
    - 'files': list of domain raster entries ('name', 'index_base').
    - 'valid_threshold': minimum fraction of valid pixels for tile
        acceptance.
    - 'target_variance': PCA target cumulative explained variance.
    - 'output_dirpath': where DomainTileMap artifacts are written.

The output is a mapping from domain base names (filename without suffix)
to DomainTileMap instances, suitable for downstream model conditioning or
task-level feature assembly.
'''

# standard imports
from __future__ import annotations
import copy
import dataclasses
# local imports
import landseg.geopipe.foundation.common as common
import landseg.geopipe.foundation.domain_maps as domain_maps
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DomainMappingConfig:
    '''Container for domain mapping configurations.'''
    source_dir: str
    file_list: list[DomainFile]
    output_dir: str
    valid_threshold: float
    target_variance: float

# --------------------------------private  type--------------------------------
@dataclasses.dataclass
class DomainFile:
    '''Typed domain file.'''
    filename: str
    index_base: int

# -------------------------------Public Function-------------------------------
def prepare_domain(
    world_grid: common.GridLayoutLike,
    config: DomainMappingConfig,
    logger: utils.Logger
) -> None:
    '''
    Prepare and persist domain tile maps for categorical raster(s).

    This function is the public entry point for domain preparation. It
    aligns the provided world grid to each configured domain raster,
    builds a `DomainTileMap` if no artifact exists yet, persists it, and
    returns a dictionary keyed by domain name (filename without suffix).

    Args:
        grid_id: Identifier of the world grid; written to domain metadata
            to preserve traceability between grid and domain artifacts.
        world_grid: A grid.GridLayout instance describing the tiling to
            use. The domain rasters must share its CRS and pixel size;
            pixel origin alignment is handled internally.
        config: Domain configuration dict. Expected keys
            - 'dirpath': directory containing domain rasters.
            - 'files': list of {'name': str, 'index_base': int}.
            - 'valid_threshold': float in [0, 1], min valid-pixel frac.
            - 'target_variance': float in (0, 1], PCA target EVR.
            - 'output_dirpath': directory for persisted artifacts.
        logger:
            Logger used for structured progress messages.

    Returns:
        dict: A mapping from domain base name to the prepared
            `DomainTileMap`.

    Notes:
    - Existing domain artifacts are loaded and returned without rebuild.
    - New artifacts are saved as JSON payload plus JSON metadata with a
    schema id and integrity hash for compatibility checks.
    '''

    # get a child logger
    logger = logger.get_child('dkmap')

    # prep output dict - domain maps indexed by filename without extension
    output: dict[str, domain_maps.DomainTileMap] = {}

    # whether to create/update flag
    build = False
    # prep each domain
    for file in config.file_list:

        # skip non raster files
        if not file.filename.endswith(('.tif', '.tiff')):
            continue
        # get filepath and filename without extension
        raster_fpath = f'{config.source_dir}/{file.filename}'
        name = file.filename.split('.', maxsplit=1)[0]

        # if domain JSON already exist, load
        try:
            logger.log('INFO', f'Loading domain {name}')
            dom = domain_maps.load_domain(name, config.output_dir)
            dom.logger = logger # add logger
            # check if domain was mapped to a different grid
            if dom.blk_size != world_grid.tile_size:
                logger.log('ERROR', f'{name} on a grid with different size')
                raise ValueError
            if not world_grid.tile_overlap in dom.blk_overlaps:
                logger.log('INFO', 'Adding suport to to input grid overlap')
                build = True
            else:
                output[name] = dom
                logger.log('INFO', f'No updates for domain {name}, load as is')
                continue
        # otherwise create domain accordingly
        except FileNotFoundError:
            logger.log('INFO', f'Domain {name} not found, create')
            build = True
            dom = domain_maps.DomainTileMap(
                config.valid_threshold,
                config.target_variance,
                logger
            ) # init a new domain map class object

        # create or update
        if build:
            domain_package = domain_maps.map_domain_to_grid(
                copy.deepcopy(world_grid),
                raster_fpath,
                logger,
                index_base=file.index_base,
            )
            dom.build(
                domain_package.block_size,
                domain_package.block_overlap,
                domain_package.index_range,
                domain_package.tiles_dict
            )
            output[name] = dom
            domain_maps.save_domain(name, dom, config.output_dir)
            logger.log('INFO', f'Domain {file.filename} created/updated')
