# pylint: disable=no-value-for-parameter
'''End-to-end experiment.'''

# standard imports
import os
import sys
import typing
# third-party imports
import hydra
import omegaconf
# local imports
import landseg.controller as controller
import landseg.dataset as dataset
import landseg.training as training
import landseg.utils as utils

# main process
@hydra.main('pkg://landseg/configs', 'config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''End-to-end experiment.'''

    # fetch user settings and merge with default config
    candidates = [f'{os.getcwd()}/settings.yaml']
    for p in candidates:
        if os.path.exists(p):
            user_cfg = omegaconf.OmegaConf.load(p)
            if not isinstance(user_cfg, omegaconf.DictConfig):
                raise TypeError('settings.yaml must have a mapping at the root')
            merged = omegaconf.OmegaConf.merge(config, user_cfg) # right wins
            config = typing.cast(omegaconf.DictConfig, merged)

    # resolve
    omegaconf.OmegaConf.resolve(config)

    # handles keyboard interruption
    try:
        # create a centralized logger file named by current time stamp
        timestamp = utils.get_timestamp()
        logger = utils.Logger('main', f'./logs/{timestamp}.log')

        # data preparation
        data_specs = dataset.load_data(config, logger)

        # build trainer
        trainer = training.build_trainer(data_specs, config, logger)

        # build controller
        runner = controller.build_controller(trainer, config, logger)

        # run via controller
        runner.fit()
    except KeyboardInterrupt:
        print('\nExperiment interrupted, exiting...')
        sys.exit(130)

if __name__ == '__main__':
    main()
