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

# pylint: disable=missing-class-docstring

'''
Model architecture schema
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.configs.schema.base as base

# alias
field = dataclasses.field


# ----- public types
class ProjectionConfig(typing.TypedDict):
    use_mlp: typing.NotRequired[bool]
    hidden_dim: typing.NotRequired[int | None]
    num_hidden_layers: typing.NotRequired[int]
    dropout: typing.NotRequired[float]
    activation: typing.NotRequired[str]


class ConditionerAdapter(typing.TypedDict):
    hidden_dim: typing.NotRequired[int] # currently for FiLM


class DomainTargetConfig(base.BaseConfigSection):
    name: str
    use_ids: bool
    use_vec: bool
    ids_embd_dims: int
    vec_proj_dims: int
    vec_proj_config: ProjectionConfig
    conditioner_config: ConditionerAdapter

    def __post_init__(self):
        if self.use_ids and not self.ids_embd_dims:
            raise ValueError(...)
        if self.use_vec and not self.vec_proj_dims:
            raise ValueError(...)

    def validate(self) -> None:
        self.require_attr_type_range('ids_embd_dims', int, (1, None))
        self.require_attr_type_range('vec_proj_dims', int, (1, None))


def _concat_proj() -> ProjectionConfig:
    return {'use_mlp': False}


def _concat_cond() -> ConditionerAdapter:
    return {}


@dataclasses.dataclass
class Concat(DomainTargetConfig):
    name: str = 'concat'
    use_ids: bool = True
    use_vec: bool = True
    ids_embd_dims: int = 4
    vec_proj_dims: int = 4
    vec_proj_config: ProjectionConfig = field(default_factory=_concat_proj)
    conditioner_config: ConditionerAdapter = field(default_factory=_concat_cond)


def _film_proj() -> ProjectionConfig:
    return {
        'use_mlp': True,
        'hidden_dim': 128,
        'num_hidden_layers': 1,
        'activation': 'gelu',
        'dropout': 0.1,
    }


def _film_cond() -> ConditionerAdapter:
    return {'hidden_dim': 128}

@dataclasses.dataclass
class FiLM(DomainTargetConfig):

    name: str = 'film'
    use_ids: bool = True
    use_vec: bool = True
    ids_embd_dims: int = 4
    vec_proj_dims: int = 4
    vec_proj_config: ProjectionConfig = field(default_factory=_film_proj)
    conditioner_config: ConditionerAdapter = field(default_factory=_film_cond)


def default_conditioners() -> dict[str, typing.Any]:
    '''Return default conditioner registry.'''
    return {
        'concat': Concat(),
        'film': FiLM()
    }
