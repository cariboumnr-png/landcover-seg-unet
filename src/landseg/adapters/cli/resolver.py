# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''
Hydra configs resolver
'''

# standard imports
import os
import pathlib
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs

# register omegaconf.OmegaConf.resolvers
omegaconf.OmegaConf.register_new_resolver('concat', lambda x, y: x + y)


# ----- constants
SECTION_RECIPES = [
    'world-grid',
    'data-harmonize',
    'data-ingest',
    'data-prepare',
    'diagnose-overfit',
    'model-evaluate',
    'model-train',
    'batch-ingest',
    'study-analysis',
    'study-sweep'
]

E2E_RECIPES = [
    'e2e-intake',
    'e2e-experiment',
]


# ----- public functions
def resolve_configs(
    config: omegaconf.DictConfig,
    *,
    e2e_override: bool = True
) -> configs.RootConfig:
    '''Resolve configs from difference sources'''

    # list of configs to resolve
    config_list: list = []

    # add schema - dataclass scaffolding with default dummy values
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    config_list.append(schema)

    # add Hydra-composed config
    # - as the single source of truth in API mode
    # - overridden by additional settings (*yaml) below in CLI mode
    config_list.append(config)

    # get command from config
    cmd = omegaconf.OmegaConf.select(config, 'command', default=None)

    # resolve sectional recipes
    targets: list[str] = []
    root_dpath = pathlib.Path(__file__).resolve().parents[4]
    if cmd == 'default' or cmd is None:
        pass
    elif cmd == 'preflight':
        target = omegaconf.OmegaConf.select(config, 'execution.preflight_target')
        if target is None or target == 'all':
            targets = SECTION_RECIPES
        else:
            targets.append(target)
    else:
        targets.append(cmd)

    for t in targets:
        recipe_cfg = _resolve_recipe_config(t, root_dpath)
        config_list.append(recipe_cfg)

    # resolve end-to-end recipes
    if e2e_override and cmd in E2E_RECIPES:
        for t in E2E_RECIPES:
            recipe_cfg = _resolve_recipe_config(t, root_dpath)
            config_list.append(recipe_cfg)

    # add dev settings (optional and untracked)
    dev = omegaconf.OmegaConf.select(config, 'execution.dev_cfg', default=None)
    if dev:
        if os.path.exists(dev):
            dev_cfg = omegaconf.OmegaConf.load(dev)
            assert isinstance(dev_cfg, omegaconf.DictConfig)
            config_list.append(dev_cfg)
        else:
            raise FileNotFoundError(f'Dev override config not found: {dev}')

    # merging configs in order (last wins)
    # dev -> e2e recipe -> section recipe -> hydra defaults -> schema defaults
    with omegaconf.open_dict(config):
        merged = omegaconf.OmegaConf.merge(*config_list)
    cfg = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(cfg)

    # construct and cast config dataclass
    root = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg))
    root.execution.cli_mode = True

    root.validate()
    return root


# ----- private helpers
def _resolve_recipe_config(
    cmd: str,
    root_dpath: pathlib.Path,
) -> omegaconf.DictConfig:
    norm_cmd = cmd.replace('-', '_')

    if norm_cmd == 'batch_ingest':
        norm_cmd = 'data_ingest'
    elif norm_cmd == 'end_to_end_experiment':
        norm_cmd = 'e2e_experiment'
    elif norm_cmd == 'end_to_end_intake':
        norm_cmd = 'e2e_intake'

    candidates = [
        root_dpath / 'configs' / f'{norm_cmd}.yaml',
        root_dpath / 'configs' / 'end_to_end' / f'{norm_cmd}.yaml',
    ]

    resolved_path = None
    for path in candidates:
        if path.exists():
            resolved_path = path

    if resolved_path is None:
        expected = root_dpath / 'configs' / f'{norm_cmd.replace("-", "_")}.yaml'
        raise FileNotFoundError(
            f'Tracked recipe not found for command "{cmd}" at: {expected}'
        )

    resolved_configs = omegaconf.OmegaConf.load(resolved_path)
    assert isinstance(resolved_configs, omegaconf.DictConfig)
    return resolved_configs
