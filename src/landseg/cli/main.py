# pylint: disable=no-value-for-parameter
'''Main CLI entry point.'''

# standard imports
import os
import sys
import typing
# third-party imports
import hydra
import hydra.core.hydra_config
import hydra.utils
import omegaconf
# local imports
import landseg.cli as cli
import landseg.utils as utils

command_registry = {
    'end_to_end': cli.train_end_to_end,
    'overfit_test': cli.overfit_test
}

# main process
@hydra.main('pkg://landseg/configs', 'config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''Main CLI entry point.'''

    # get running profile
    get = hydra.core.hydra_config.HydraConfig.get()
    profile = get.runtime.choices['profile']

    # resolve config
    _config = _resolve_config(config)

    # master logger
    logger = utils.Logger('cli', os.path.join(_config['exp_root'], 'cli.log'))

    # run specified mode with exceptions handling
    try:
        logger.log('INFO', f'Runing profile: {profile} start')
        _config['profile'] = profile # add profile string to config tree
        command_registry[profile](_config)
        logger.log('INFO', f'Runing profile: {profile} finish')
    # manual keyboard interruption
    except KeyboardInterrupt:
        logger.log('INFO', '\nExperiment manually interrupted, exiting...')
        sys.exit(130)
    # capture others and log
    except Exception: # pylint: disable=broad-exception-caught
        logger.log('CRITICAL', 'Unhandled exception occurred', exc_info=True)
        sys.exit(1)

def _resolve_config(config: omegaconf.DictConfig) -> omegaconf.DictConfig:
    '''Resolve configs from difference sources.'''

    # safer CWD fetching
    original_cwd = hydra.utils.get_original_cwd()

    # user settings at root
    candidates = [os.path.join(original_cwd, 'settings.yaml')]
    # optional dev settings (untracked, supplied via CLI argument)
    aux = config.get('dev_settings_path')
    if aux:
        aux = aux if os.path.isabs(aux) else os.path.join(original_cwd, aux)
    candidates.append(aux)

    # overwrite from selected profile
    profile = config.profile

    # merging overrides with default config tree and resolve
    for p in candidates:
        if os.path.exists(p):
            user_cfg = omegaconf.OmegaConf.load(p)
            if not isinstance(user_cfg, omegaconf.DictConfig):
                raise TypeError('./settings.yaml must have a mapping')
            # allow new phases to be added
            with omegaconf.open_dict(config.experiment.phases):
                # right wins over left during merging
                merged = omegaconf.OmegaConf.merge(config, user_cfg, profile)
                config = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(config)

    # return
    return config

if __name__ == '__main__':
    main()
