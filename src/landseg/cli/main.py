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

# pylint: disable=no-value-for-parameter
'''
CLI entry that resolves Hydra configs, selects a profile, dispatches the
mapped command, and manages logging and error handling.
'''

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
    '''Run the selected CLI profile with resolved configuration.'''

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
    '''Merge and resolve config sources.'''

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
