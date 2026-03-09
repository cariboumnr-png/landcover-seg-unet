# pylint: disable=no-value-for-parameter
'''hydra schema test'''

# standard imports
import dataclasses
import json
import typing
# third-party imports
import hydra
import omegaconf
# local imports
import landseg.configs as configs

# main process
@hydra.main('pkg://landseg/configs', 'config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''Run the selected CLI profile with resolved configuration.'''

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    merged = omegaconf.OmegaConf.merge(schema, config)
    omegaconf.OmegaConf.resolve(merged)

    cfg = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(merged))
    cfg_dict = dataclasses.asdict(cfg)
    print(json.dumps(cfg_dict, indent=4))

if __name__ == '__main__':
    main()
