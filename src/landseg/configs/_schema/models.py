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
@dataclasses.dataclass
class _ConvParams:
    norm: str = 'gn'
    gn_groups: int = 8
    p_drop: float = 0.0

@dataclasses.dataclass
class _UnetBody:
    body: str = 'unet'
    base_ch: int = 32
    conv_params: dict[str, typing.Any] = field(
        default_factory=lambda: {
            'downs': dataclasses.asdict(_ConvParams()),
            'ups': dataclasses.asdict(_ConvParams())
        }
    )

@dataclasses.dataclass
class _UnetPPBody:
    body: str = 'unetpp'
    base_ch: int = 32
    conv_params: dict[str, typing.Any] = field(
        default_factory=lambda: {
            'downs': dataclasses.asdict(_ConvParams()),
            'nodes': dataclasses.asdict(_ConvParams())
        }
    )

@dataclasses.dataclass
class _Concat:
    out_dim: int = 4
    use_ids: bool = True
    use_vec: bool = True
    use_mlp: bool = True

@dataclasses.dataclass
class _FiLM:
    embed_dim: int = 4
    use_ids: bool = True
    use_vec: bool = True
    hidden: int = 128

@dataclasses.dataclass
class _Conditioning:
    mode: str | None = None  #  'concat' | 'film' | 'hybrid'
    concat: _Concat = field(default_factory=_Concat)
    film: _FiLM = field(default_factory=_FiLM)

@dataclasses.dataclass
class _ModelFlags:
    enable_logit_adjust: bool = True
    enable_clamp: bool = True

# ----- MODELS
@dataclasses.dataclass
class ModelsConfig:
    use_body: str = 'unet'
    body_registry: dict[str, typing.Any] = field(
        default_factory=lambda: {
            'unet': _UnetBody(),
            'unetpp': _UnetPPBody(),
        }
    )
    conditioning: _Conditioning = field(default_factory=_Conditioning)
    clamp_range: tuple[float, float] = (1e-4, 1e4)
    flags: _ModelFlags = field(default_factory=_ModelFlags)

    def __post_init__(self):
        mode = self.conditioning.mode
        if mode and mode not in ['hybrid', 'concat', 'film']:
            raise ValueError(f'Invalid conditionning mode: {mode}')

    def validate(self) -> None:
        # cross-check clamp range ordering
        lo, hi = self.clamp_range
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError('invalid clamp_range ordering or non-positive')
