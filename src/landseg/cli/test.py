# pylint: disable=no-value-for-parameter
'''hydra schema test'''

# standard imports
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

    root_config = _resolve_configs(config)

def _resolve_configs(config: omegaconf.DictConfig) -> configs.RootConfig:
    '''Resolve configs from difference sources'''

        # list of configs to resolve
    config_list: list = []

    # add schema
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    config_list.append(schema)

    # get user settings at root (with safer CWD fetching)
    user = os.path.join(hydra.utils.get_original_cwd(), 'settings.yaml')
    if os.path.exists(user):
        user_settings = omegaconf.OmegaConf.load(user)
        assert isinstance(user_settings, omegaconf.DictConfig)
        config_list.append(user_settings)

    # get dev settings (untracked)
    dev = config.get('dev_settings_path')
    if dev and os.path.exists(dev):
        dev_settings = omegaconf.OmegaConf.load(dev)
        assert isinstance(dev_settings, omegaconf.DictConfig)
        config_list.append(dev_settings)

    # final profile overwrites
    config_list.append(config.profile)

    # merging overrides resolve
    with omegaconf.open_dict(config):
        merged = omegaconf.OmegaConf.merge(*config_list)
    cfg = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(cfg)

    # return the casted config dataclass
    return typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg))

if __name__ == '__main__':
    main()
