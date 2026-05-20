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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods

'''
Model architecture schema
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing

# alias
field = dataclasses.field

# ---------------------------------MODELS CONFIGS------------------------------
# ----- UNet bodies
class _UNetBodyConfig(typing.Protocol):
    body: str
    base_ch: int
    conv_params: dict[str, typing.Any]

@dataclasses.dataclass
class _ConvParams:
    norm: str = 'gn'
    gn_groups: int = 8
    p_drop: float = 0.0

@dataclasses.dataclass
class _UNet(_UNetBodyConfig):
    body: str = 'unet'
    base_ch: int = 32
    conv_params: dict[str, typing.Any] = field(
        default_factory=lambda: {
            'downs': dataclasses.asdict(_ConvParams()),
            'ups': dataclasses.asdict(_ConvParams())
        }
    )

@dataclasses.dataclass
class _UNetPP(_UNetBodyConfig):
    body: str = 'unetpp'
    base_ch: int = 32
    conv_params: dict[str, typing.Any] = field(
        default_factory=lambda: {
            'downs': dataclasses.asdict(_ConvParams()),
            'nodes': dataclasses.asdict(_ConvParams())
        }
    )

@dataclasses.dataclass
class _UNetPPP(_UNetBodyConfig):
    body: str = 'unetpp'
    base_ch: int = 32
    conv_params: dict[str, typing.Any] = field(
        default_factory=lambda: {
            'downs': dataclasses.asdict(_ConvParams()),
            'nodes': dataclasses.asdict(_ConvParams())
        }
    )

# ----- conditioners
class _DomainTargetConfig(typing.Protocol):
    name: str
    use_ids: bool
    use_vec: bool
    ids_embd_dims: int
    vec_proj_dims: int
    vec_proj_config: _DomainProjectionConfig

class _DomainProjectionConfig(typing.TypedDict):
    use_mlp: typing.NotRequired[bool]
    hidden_dim: typing.NotRequired[int | None]
    num_hidden_layers: typing.NotRequired[int]
    dropout: typing.NotRequired[float]
    activation: typing.NotRequired[str]

class _DomainTargetConfigBase(_DomainTargetConfig):
    def __post_init__(self):
        if self.use_ids and not self.ids_embd_dims:
            raise ValueError(...)
        if self.use_vec and not self.vec_proj_dims:
            raise ValueError(...)

@dataclasses.dataclass
class _Concat(_DomainTargetConfigBase):
    name: str = 'concat'
    use_ids: bool = True
    use_vec: bool = True
    ids_embd_dims: int = 4
    vec_proj_dims: int = 4
    vec_proj_config: _DomainProjectionConfig = field(
        default_factory=lambda: {
            'use_mlp': False
        }
    )

@dataclasses.dataclass
class _FiLM(_DomainTargetConfigBase):
    name: str = 'film'
    use_ids: bool = True
    use_vec: bool = True
    ids_embd_dims: int = 4
    vec_proj_dims: int = 4
    vec_proj_config: _DomainProjectionConfig = field(
        default_factory=lambda: {
            'use_mlp': True,
            'hidden_dim': 256,
            'num_hidden_layers': 1,
            'activation': 'gelu',
            'dropout': 0.1,
        }
    )

# ----- MODELS
@dataclasses.dataclass
class ModelsConfig:
    model_body: str = 'unet'
    model_body_registry: dict[str, _UNetBodyConfig] = field(
        default_factory=lambda: {
            'unet': _UNet(),
            'unetpp': _UNetPP(),
            'unetppp': _UNetPPP(),
        }
    )
    conditioner: list[str] = ['concat', 'film']
    conditioner_registry: dict[str, _DomainTargetConfig] = field(
        default_factory=lambda: {
            'concat': _Concat,
            'film': _FiLM
        }
    )
    enable_clamp: bool = True
    clamp_range: tuple[float, float] = (1e-4, 1e4)

    @property
    def model_body_config(self) -> _UNetBodyConfig:
        assert self.model_body in self.model_body_registry, 'Invalid model body'
        return self.model_body_registry[self.model_body]

    @property
    def conditioning_config(self) -> dict[str, _DomainTargetConfig]:
        assert all(c in self.conditioner_registry for c in self.conditioner)
        return {c: self.conditioner_registry[c] for c in self.conditioner}

    def validate(self) -> None:
        # cross-check clamp range ordering
        lo, hi = self.clamp_range
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError('invalid clamp_range ordering or non-positive')
