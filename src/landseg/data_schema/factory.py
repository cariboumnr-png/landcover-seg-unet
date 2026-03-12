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
Dataset loading and validation pipeline. Prepares the world grid and
domain features, validates/repairs dataprep artifacts, and emits a
`DataSpecs` object for training and evaluation.

Public APIs:
    - load_data: Prepare grid/domains, validate schema, rebuild if
      needed, and return a dataset specifications dataclass.
'''

# standard imports
import os
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.data_schema as data_schema
import landseg.prep_raster as prep_raster
import landseg.prep_domain as prep_domain
import landseg.prep_grid as prep_grid
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def load_data(
    inputs: configs.Inputs,
    prep: configs.Prep,
    logger: utils.Logger,
    *,
    single_block_mode: bool = False,
    single_block_dir: str = ''
) -> core.DataSpecs:
    '''
    Load dataset specifications after validating dataprep artifacts.

    Args:
        config: Hydra/OMEGACONF configuration containing extent, grid,
            dataset, artifacts, dataprep, and domain settings.
        logger: Logger for progress, warnings, and error messages.

    Returns:
        DataSpecs: Consolidated dataset specification ready for training.

    Raises:
        ValueError: If configuration fields required for grid/domain/
            dataprep are malformed or missing.
    '''

    # load/create world grid
    world_grid = prep_grid.prep_world_grid(inputs.extent, prep.grid, logger)

    # if single block mode
    if single_block_mode:
        # build a minimul schema dict from a single block
        blk_schema = prep_raster.prepare_dataset(
            world_grid,
            inputs.data,
            prep.data,
            logger,
            build_a_block=True,
            block_fpath=os.path.join(single_block_dir, 'overfit_test_block.npz')
        )
        assert blk_schema # sanity

        # build a dataspec from the schema with essential values
        dspecs = data_schema.build_dataspec_one_block(blk_schema)
        return dspecs

    # load/map domain
    domains = prep_domain.prepare_domain(
        world_grid,
        inputs.domain,
        prep.domain,
        logger
    )
    ids_domain = domains[prep.domain.as_ids] if prep.domain.as_ids else None
    vec_domain = domains[prep.domain.as_vec] if prep.domain.as_vec else None

    # validate data blocks
    status = data_schema.validate_schema(
        world_grid.gid,
        prep.data.output_dirpath,
        logger
    )

    # prompt data blocks rebuild
    if status == 1:
        logger.log('WARNING', 'Data schema check failed, rebuild data blocks')
        prep_raster.prepare_dataset(
            world_grid,
            inputs.data,
            prep.data,
            logger
        )
    # prompt data blocks fresh build (all steps will be redone)
    elif status == 2:
        logger.log('WARNING', 'Data schema not found, build data blocks')
        prep_raster.prepare_dataset(
            world_grid,
            inputs.data,
            prep.data,
            logger,
            rebuild_all=True
        )
    # passed
    else:
        logger.log('INFO', 'Data schema check passed, proceed')

    # build dataspec
    dataspec = data_schema.build_dataspec(
        f'{prep.data.output_dirpath}/schema.json',
        ids_domain,
        vec_domain
    )

    # return
    return dataspec
