# pylint: disable=no-value-for-parameter
'''Main entry point.'''

# standard imports
import sys
# third-party imports
import hydra
import omegaconf
# local imports
import src.dataset_
import src.dataset
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
        console_lvl=20 # debug
    )

    # Branch test - data preparation
    datapecs = src.dataset_.load_data(config, logger)

    # # data preparation
    # data_summary = src.dataset.prepare_data(
    #     dataset_name=config.dataset_name,
    #     config=config,
    #     logger=logger,
    #     mode=config.dataprep_mode
    # )
    # print(data_summary)

    # setup multihead model
    model = src.models.multihead_unet(
        data_specs=datapecs,
        config=config.models
    )

    # setup trainer
    trainer = src.training.build_trainer(
        trainer_mode=config.trainer_mode,
        model=model,
        data_summary=datapecs,
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
