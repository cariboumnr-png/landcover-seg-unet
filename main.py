# pylint: disable=no-value-for-parameter
'''Main entry point.'''

# standard imports
import sys
# third-party imports
import hydra
import omegaconf
# local imports
import src.dataprep
import src.dataset
import src.domain
import src.grid
import src.models
import src.training
import src.utils

# add resolvers to omegaconf for conveniences
#  - convert decimals to percentages
omegaconf.OmegaConf.register_new_resolver('c', lambda x: int(x * 100))


# main process
@hydra.main(config_path='./configs', config_name='config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''doc'''

    # create a centralized logger file named by current time stamp
    timestamp = src.utils.get_timestamp()
    logger = src.utils.Logger(
        name='main',
        log_file=f'./logs/{timestamp}.log',
        console_lvl=10 # debug
    )

    # -------------------------------------------------------------------------
    # TEST: world grid preparation
    gid, world_grid = src.grid.prep_world_grid(
        extent=config.extent,
        config=config.grid,
        logger=logger
    )
    print(world_grid)

    # TEST: domain mapping
    domains = src.domain.prepare_domain(
        grid_id=gid,
        world_grid=world_grid,
        config=config.domain,
        logger=logger
    )
    for k, dom in domains.items():
        print(k)
        print(dom)

    # TEST: data prep
    src.dataprep.prepare_data(
        world_grid=world_grid,
        inputs_config=config.dataset,
        artifact_config=config.artifacts,
        cache_config=config.cache,
        logger=logger
    )
    #--------------------------------------------------------------------------

    # data preparation
    data_summary = src.dataset.prepare_data(
        dataset_name=config.dataset_name,
        config=config,
        logger=logger,
        mode=config.dataprep_mode
    )
    print(data_summary)

    # setup multihead model
    model = src.models.multihead_unet(
        data_summary=data_summary,
        config=config.models
    )

    # setup trainer
    trainer = src.training.build_trainer(
        trainer_mode=config.trainer_mode,
        model=model,
        data_summary=data_summary,
        config=config.trainer,
        logger=logger
    )

    # setup curriculum
    controller = src.training.build_controller(
        trainer=trainer,
        config=config.curriculum,
        logger=logger
    )

    # start
    controller.fit()

if __name__ == '__main__':
    # handles keyboard interruption
    try:
        sys.exit(main()) # no args needed for hydra
    except KeyboardInterrupt:
        print('\nExperiment interrupted, exiting...')
        sys.exit(130)
