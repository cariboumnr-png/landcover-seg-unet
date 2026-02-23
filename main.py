# pylint: disable=no-value-for-parameter
'''Main entry point.'''

# standard imports
import sys
# third-party imports
import hydra
import omegaconf
# local imports
import src.controller
import src.dataset
import src.training
import src.utils

# main process
@hydra.main(config_path='./configs', config_name='config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''End-to-end process'''

    # create a centralized logger file named by current time stamp
    timestamp = src.utils.get_timestamp()
    logger = src.utils.Logger('main', f'./logs/{timestamp}.log')

    # data preparation
    data_specs = src.dataset.load_data(config, logger)

    # build trainer
    trainer = src.training.build_trainer(data_specs, config, logger)

    # build controller
    controller = src.controller.build_controller(trainer, config, logger)

    # run via controller
    controller.fit()

if __name__ == '__main__':
    # handles keyboard interruption
    try:
        sys.exit(main()) # no args needed for hydra
    except KeyboardInterrupt:
        print('\nExperiment interrupted, exiting...')
        sys.exit(130)
