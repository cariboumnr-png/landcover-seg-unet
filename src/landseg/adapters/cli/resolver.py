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
import pathlib
import typing
# third-party imports
import omegaconf
# local imports
import landseg._constants as c
import landseg.configs as configs

# register omegaconf.OmegaConf.resolvers
omegaconf.OmegaConf.register_new_resolver('concat', lambda x, y: x + y)


# ----- public functions
def resolve_configs(config: omegaconf.DictConfig) -> configs.RootConfig:
    '''
    Resolve hierarchical configuration from schemas, recipes, and overrides.

    Args:
        config:
            Hydra-composed input configuration.
        e2e_override:
            If True and running an e2e command, end-to-end composite recipes
            override sectional settings.

    Returns:
        configs.RootConfig:
            Fully resolved and validated root configuration instance.
    '''

    # list of configs to resolve
    config_list: list = []

    # add schema - dataclass scaffolding with default dummy values
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    config_list.append(schema)

    # add Hydra-composed config
    # - as the single source of truth in API mode
    # - overridden by additional settings (*yaml) below in CLI mode
    config_list.append(config)

    # get command and preflight target from input config
    cmd = omegaconf.OmegaConf.select(config, 'command', default=None)
    pf_target = omegaconf.OmegaConf.select(
        config, 'execution.preflight_target', default=None
    )
    targets = configs.RootConfig.commands_to_validate(cmd, pf_target)

    # add config recipes
    recipe_configs = _resolve_recipes(config, targets)
    config_list.extend(recipe_configs)

    # add dev settings (optional and untracked)
    dev_configs = _resolve_dev(config)
    config_list.extend(dev_configs)

    # merging configs in order (last wins)
    # dev -> e2e recipe -> section recipe -> hydra defaults -> schema defaults
    with omegaconf.open_dict(config):
        merged = omegaconf.OmegaConf.merge(*config_list)
    cfg = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(cfg)

    # construct and cast config dataclass
    root = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg))
    root.execution.cli_mode = True

    # validate and return
    root.validate(targets)
    return root


# ----- private helpers
def _resolve_recipes(
    config: omegaconf.DictConfig,
    targets: list[str],
) -> list[omegaconf.DictConfig]:
    recipe_dir = omegaconf.OmegaConf.select(
        config, 'execution.recipe_dir', default='./configs'
    ) or './configs'
    recipe_dpath = pathlib.Path(recipe_dir)

    # sort in-place. Section recipes first, then E2Es
    ordered_recipes = {value: i for i, value in enumerate(c.RECIPES)}
    sorted_targets = sorted(
        targets, key=lambda x: ordered_recipes.get(x, len(c.RECIPES))
    )

    # load each target
    resolved_recipes: list[omegaconf.DictConfig] = []
    for target in sorted_targets:

        norm_target = target.replace('-', '_') # command str to yaml file name
        candidates: list[pathlib.Path] = [
            recipe_dpath / f'{norm_target}.yaml',
            recipe_dpath / 'end_to_end' / f'{norm_target}.yaml',
        ]

        found = False
        for path in candidates:
            if path.exists():
                resolved = _load_config(path)
                resolved_recipes.append(resolved)
                found = True
                break

        if not found:
            raise FileNotFoundError(
                f'Tracked recipe not found for command "{target}" at: '
                f'{candidates[0]}'
            )

    return resolved_recipes


def _resolve_dev(config: omegaconf.DictConfig) -> list[omegaconf.DictConfig]:
    fp = omegaconf.OmegaConf.select(config, 'execution.dev_cfg')
    if fp is not None:
        return [_load_config(fp)]
    return []


def _load_config(path: str | pathlib.Path) -> omegaconf.DictConfig:
    path = pathlib.Path(path)
    if path.exists():
        loaded_cfg = omegaconf.OmegaConf.load(path)
        if not isinstance(loaded_cfg, omegaconf.DictConfig):
            raise ValueError(f'Invalid config at: {str(path.resolve())}')
    else:
        raise FileNotFoundError(f'Config not found: {path}')
    return loaded_cfg
