# pylint: disable=no-value-for-parameter
'''Main CLI entry point.'''

# standard imports
import os
import sys
import typing
# third-party imports
import hydra
import hydra.utils
import omegaconf
# local imports
import landseg.cli as cli
import landseg.utils as utils

CMD = {
    'end-to-end': cli.train_end_to_end,
    'overfit-test': cli.overfit_test
}

# main process
@hydra.main('pkg://landseg/configs', 'config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''Main CLI entry point.'''

    # safer CWD fetching
    original_cwd = hydra.utils.get_original_cwd()

    # user settings at root
    candidates = [os.path.join(original_cwd, 'settings.yaml')]
    # optional dev settings (untracked, supplied via CLI argument)
    aux = config.get('dev_settings_path')
    if aux:
        aux = aux if os.path.isabs(aux) else os.path.join(original_cwd, aux)
    candidates.append(aux)

    # merging overrides with default config tree and resolve
    for p in candidates:
        if os.path.exists(p):
            user_cfg = omegaconf.OmegaConf.load(p)
            if not isinstance(user_cfg, omegaconf.DictConfig):
                raise TypeError('./settings.yaml must have a mapping')
            # allow new phases to be added
            with omegaconf.open_dict(config.experiment.phases):
                merged = omegaconf.OmegaConf.merge(config, user_cfg) # right wins
                config = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(config)

    # master logger
    logger = utils.Logger('cli', os.path.join(original_cwd, 'log.log'))

    # run specified mode with exceptions handling
    mode = config['run_mode']
    try:
        CMD[mode](config)
    # manual keyboard interruption
    except KeyboardInterrupt:
        logger.log('INFO', '\nExperiment manually interrupted, exiting...')
        sys.exit(130)
    # capture others and log
    except Exception: # pylint: disable=broad-exception-caught
        logger.log('CRITICAL', 'Unhandled exception occurred', exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
