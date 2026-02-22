'''Validate world grid, domain, datablocks as per configuration.'''

# third-party imports
import omegaconf
# local imports
import dataset_
import dataprep
import domain
import grid
import utils

def load_data(
    config: omegaconf.DictConfig,
    logger: utils.Logger
) -> dataset_.DataSpecs:
    '''doc'''

    # load/create world grid
    gid, world_grid = grid.prep_world_grid(config.extent, config.grid, logger)

    # load/map domain
    domains = domain.prepare_domain(gid, world_grid, config.domain, logger)
    as_ids = config.domain['use_type'].get('ids')
    as_vec = config.domain['use_type'].get('vec')
    ids_domain = domains[as_ids] if as_ids else None
    vec_domain = domains[as_vec] if as_vec else None

    # validate data blocks
    cache_root = f'{config.artifacts["cache"]}/{config.dataset["name"]}'
    status = dataset_.validate_schema(gid, cache_root, logger)

    # prompt data blocks rebuild
    if status == 1:
        logger.log('WARNING', 'Data schema check failed, rebuild data blocks')
        dataprep.prepare_data(
            world_grid=(gid, world_grid),
            inputs_config=config.dataset,
            artifact_config=config.artifacts,
            proc_config=config.dataprep,
            logger=logger
        )
    # prompt data blocks fresh build (all steps will be redone)
    elif status == 2:
        logger.log('WARNING', 'Data schema not found, build data blocks')
        dataprep.prepare_data(
            world_grid=(gid, world_grid),
            inputs_config=config.dataset,
            artifact_config=config.artifacts,
            proc_config=config.dataprep,
            logger=logger,
            rebuild_all=True
        )
    # passed
    else:
        logger.log('INFO', 'Data schema check passed, proceed')

    # build dataspec
    dataspec = dataset_.build_dataspec(
        schema_fpath=f'{cache_root}/schema.json',
        ids_domain=ids_domain,
        vec_domain=vec_domain
    )

    # return
    return dataspec
