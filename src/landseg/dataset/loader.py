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

# local imports
import landseg.configs as confgis
import landseg.dataset as dataset
import landseg.dataprep as dataprep
import landseg.domain as domain
import landseg.grid as grid
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def load_data(
    inputs: confgis.Inputs,
    prep: confgis.Prep,
    logger: utils.Logger
) -> dataset.DataSpecs:
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
    gid, world_grid = grid.prep_world_grid(inputs.extent, prep.grid, logger)

    # load/map domain
    domains = domain.prepare_domain(gid, world_grid, inputs.domain, prep.domain, logger)
    as_ids = prep.domain.as_ids
    as_vec = prep.domain.as_vec
    ids_domain = domains[as_ids] if as_ids else None
    vec_domain = domains[as_vec] if as_vec else None

    # validate data blocks
    status = dataset.validate_schema(gid, prep.data.output_dirpath, logger)

    # prompt data blocks rebuild
    if status == 1:
        logger.log('WARNING', 'Data schema check failed, rebuild data blocks')
        dataprep.prepare_data(
            world_grid=(gid, world_grid),
            input_config=inputs.data,
            prep_config=prep.data,
            logger=logger
        )
    # prompt data blocks fresh build (all steps will be redone)
    elif status == 2:
        logger.log('WARNING', 'Data schema not found, build data blocks')
        dataprep.prepare_data(
            world_grid=(gid, world_grid),
            input_config=inputs.data,
            prep_config=prep.data,
            logger=logger,
            rebuild_all=True
        )
    # passed
    else:
        logger.log('INFO', 'Data schema check passed, proceed')

    # build dataspec
    dataspec = dataset.build_dataspec(
        schema_fpath=f'{prep.data.output_dirpath}/schema.json',
        ids_domain=ids_domain,
        vec_domain=vec_domain
    )

    # return
    return dataspec
