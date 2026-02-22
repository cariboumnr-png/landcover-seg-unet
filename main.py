# pylint: disable=no-value-for-parameter
'''Main entry point.'''

# standard imports
import sys
# third-party imports
import hydra
import omegaconf
# local imports
import src.dataset_
import src.training
import src.utils

# add resolvers to omegaconf for conveniences
#  - convert decimals to percentages
omegaconf.OmegaConf.register_new_resolver('c', lambda x: int(x * 100))

# main process
@hydra.main(config_path='./configs', config_name='config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''End-to-end process'''

    # create a centralized logger file named by current time stamp
    timestamp = src.utils.get_timestamp()
    logger = src.utils.Logger('main', f'./logs/{timestamp}.log')

    # data preparation
    data_specs = src.dataset_.load_data(config, logger)

    # builder runner
    runner = src.training.build_runner(data_specs, config, logger)

    # start
    runner.fit()

if __name__ == '__main__':
    # handles keyboard interruption
    try:
        sys.exit(main()) # no args needed for hydra
    except KeyboardInterrupt:
        print('\nExperiment interrupted, exiting...')
        sys.exit(130)
