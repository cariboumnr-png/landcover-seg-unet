'''Config accessor with validation hooks.'''

# standard imports
import typing
# third-party imports
import omegaconf
# local imports
import _types

class ConfigAccessError(Exception):
    '''Custom exception for config access errors.'''


class ConfigKeyError(ConfigAccessError):
    '''Exception for missing config keys.'''


class ConfigTypeError(ConfigAccessError):
    '''Exception for config type mismatches.'''


class ConfigValueError(ConfigAccessError):
    '''Exception for invalid config values.'''


class ConfigAccess:
    '''
    Framework-agnostic config accessor with validation hooks (TODO).
    Works with dict-like configs (dict, OmegaConf containers, JSON, YAML).
    '''

    def __init__(
            self,
            config: _types.ConfigType,
        ) -> None:
        '''Initialize with a config mapping.'''

        self._config = self.from_omega(config)

    def get_asset(
            self,
            section: str,
            asset_type: str,
            asset_name: str
        ) -> str:
        '''
        Get asset file path from config (organized by section/type/name).
        '''

        try:
            assets = self._config[section][asset_type]
        except KeyError as exc:
            raise ConfigKeyError(
                f'Missing asset group: {section}.{asset_type}'
            ) from exc

        if not isinstance(assets, list):
            raise ConfigTypeError(f'{section}.{asset_type} must be a list')

        for a in assets:
            if not isinstance(a, dict):
                raise ConfigTypeError('Asset entry must be a dict')
            if a.get('name') == asset_name:
                if 'path' not in a:
                    raise ConfigValueError(f'Asset missing path: {asset_name}')
                return a['path']

        raise ConfigKeyError(
            f'Asset not found: {section}.{asset_type}.{asset_name}'
        )

    def get_option(
            self,
            *path: str,
            default: typing.Any = None
        ) -> typing.Any:
        '''
        Retrieve nested config value by path; Returns default if not found.
        '''

        node = self._config
        try:
            for key in path:
                node = node[key]
            return node
        except (KeyError, TypeError):
            return default

    def get_section_as_dict(
            self,
            section: str
        ) -> _types.ConfigType:
        '''
        Retrieve a section and return as a dict.
        '''

        try:
            sec: _types.ConfigType = self._config[section]
            if not isinstance(sec, typing.Mapping):
                raise ValueError('Config section not a dict')
            return sec
        except KeyError as e:
            raise e

    @staticmethod
    def from_omega(cfg: _types.ConfigType) -> _types.ConfigType:
        '''Convert OmegaConf.DictConfig to standard dict recursively.'''

        if isinstance(cfg, omegaconf.DictConfig):
            _cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(_cfg, typing.Mapping):
                raise ConfigTypeError('Converted config is not a mapping/dict')
            return _cfg
        return cfg
