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

'''Composite architecture search space configuration.'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base

# alias
field = dataclasses.field


MODEL_BODIES: list[str] = ['unet', 'unetpp', 'unetppp']
BOTTLENECKS: list[str] = ['conv', 'transformer', 'hybrid']
CONDITIONERS: list[list[str]] = [[], ['film'], ['concat'], ['film', 'concat']]

@dataclasses.dataclass
class DataGeometry(base.BaseConfigSection):
    '''Data geometry search space.'''
    patch_size: tuple[int, int, int] = (64, 128, 64)
    batch_size: tuple[int, int, int] = (16, 64, 16)


@dataclasses.dataclass
class ContextWindow(base.BaseConfigSection):
    '''Context window search space.'''
    patch_size: tuple[int, int, int] = (64, 128, 64)


@dataclasses.dataclass
class Architecture(base.BaseConfigSection):
    '''Model core architecture search space.'''
    model_body: list[str] = field(default_factory=lambda: MODEL_BODIES)
    base_channel: tuple[int, int, int] = (16, 64, 16)
    bottleneck: list[str] = field(default_factory=lambda: BOTTLENECKS)


@dataclasses.dataclass
class Bottleneck(base.BaseConfigSection):
    '''Bottleneck search space.'''
    bottleneck: list[str] = field(default_factory=lambda: BOTTLENECKS)
    num_conv_blocks: tuple[int, int, int] = (1, 4, 1)
    num_transformer_blocks: tuple[int, int, int] = (1, 4, 1)
    num_heads: list[int] = field(default_factory=lambda: [2, 4, 8])
    mlp_ratio: tuple[float, float] = (1.0, 4.0)
    dropout: tuple[float, float] = (0.0, 0.5)
    attn_dropout: tuple[float, float] = (0.0, 0.5)


@dataclasses.dataclass
class Conditioning(base.BaseConfigSection):
    '''Conditioner search space.'''
    conditioners: list[list[str]] = field(default_factory=lambda: CONDITIONERS)


@dataclasses.dataclass
class ArchitectureSearchConfig(base.BaseConfigSection):
    '''Composite architecture search space configuration.'''
    data_geometry: DataGeometry = field(default_factory=DataGeometry)
    context_window: ContextWindow = field(default_factory=ContextWindow)
    architecture: Architecture = field(default_factory=Architecture)
    bottleneck: Bottleneck = field(default_factory=Bottleneck)
    conditioning: Conditioning = field(default_factory=Conditioning)

    def validate(self) -> None:
        self.data_geometry.validate()
        self.context_window.validate()
        self.architecture.validate()
        self.bottleneck.validate()
        self.conditioning.validate()
