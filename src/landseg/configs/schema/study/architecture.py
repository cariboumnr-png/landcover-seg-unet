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
Architecture and geometry hyperparameter search spaces for study sweeps.
'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base

# alias
field = dataclasses.field


MODEL_BODIES: list[str] = ['unet', 'unetpp', 'unetppp']
BOTTLENECKS: list[str] = ['conv', 'transformer', 'hybrid']

@dataclasses.dataclass
class DataGeometrySearchSpace(base.BaseConfigSection):
    patch_size: tuple[int, int, int] = (64, 128, 64)
    batch_size: tuple[int, int, int] = (16, 64, 16)


@dataclasses.dataclass
class ContextWindowSearchSpace(base.BaseConfigSection):
    patch_size: tuple[int, int, int] = (64, 128, 64)


@dataclasses.dataclass
class ArchitectureSearchSpace(base.BaseConfigSection):
    model_body: list[str] = field(default_factory=lambda: list(MODEL_BODIES))
    base_channel: tuple[int, int, int] = (16, 64, 16)
    bottleneck: list[str] = field(default_factory=lambda: list(BOTTLENECKS))


@dataclasses.dataclass
class BottleneckSearchSpace(base.BaseConfigSection):
    bottleneck: list[str] = field(
        default_factory=lambda: list(BOTTLENECKS)
    )
    num_conv_blocks: tuple[int, int, int] = (1, 4, 1)
    num_transformer_blocks: tuple[int, int, int] = (1, 4, 1)
    num_heads: list[int] = field(default_factory=lambda: [2, 4, 8])
    mlp_ratio: tuple[float, float] = (1.0, 4.0)
    dropout: tuple[float, float] = (0.0, 0.5)
    attn_dropout: tuple[float, float] = (0.0, 0.5)


@dataclasses.dataclass
class ConditioningSearchSpace(base.BaseConfigSection):
    conditioners: list[list[str]] = field(
        default_factory=lambda: [[], ['film'], ['concat'], ['film', 'concat']]
    )
