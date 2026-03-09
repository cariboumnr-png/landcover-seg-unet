# pylint: disable=no-value-for-parameter
'''hydra schema test'''

# standard imports
import dataclasses
import json
import os
import typing
# third-party imports
import hydra
import hydra.utils
import omegaconf
# local imports
import landseg.configs as configs

# main process
@hydra.main('pkg://landseg/configs', 'config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''Run the selected CLI profile with resolved configuration.'''

    # safer CWD fetching
    cwd = hydra.utils.get_original_cwd()

    # get schema
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # user settings at root
    candidates = [os.path.join(cwd, 'settings.yaml')]

    # dev settings (untracked, supplied via CLI argument)
    aux = config.get('dev_settings_path')
    if aux:
        aux = aux if os.path.isabs(aux) else os.path.join(cwd, aux)
        candidates.append(aux)

    # overwrite from selected profile
    profile = config.profile
    print(profile)

    # merging overrides with default config tree and resolve
    for p in candidates:
        if os.path.exists(p):
            user_cfg = omegaconf.OmegaConf.load(p)
            if not isinstance(user_cfg, omegaconf.DictConfig):
                raise TypeError('./settings.yaml must have a mapping')
            # allow domain files to be added
            with omegaconf.open_dict(config.inputs.domain.files):
                merged = omegaconf.OmegaConf.merge(schema, config, user_cfg)
                config = typing.cast(omegaconf.DictConfig, merged)
            # allow new phases to be added
            with omegaconf.open_dict(config.controller.phases):
                merged = omegaconf.OmegaConf.merge(schema, config, user_cfg)
                config = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(config)

    cfg = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(config))
    cfg_dict = dataclasses.asdict(cfg)
    print(json.dumps(cfg_dict, indent=4))

if __name__ == '__main__':
    main()
