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

'''Models configuration section'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.configs.schema.base as base
import landseg.configs.schema.models.backbones as backbones_sec
import landseg.configs.schema.models.conditioners as conditioners_sec
import landseg.configs.schema.models.safety as safety_sec

# ----- aliases
field = dataclasses.field
default_bodies = backbones_sec.default_bodies
default_bottlenecks = backbones_sec.default_bottlenecks
default_conditioners = conditioners_sec.default_conditioners
UNetBackbone = backbones_sec.UNetBackboneConfig
DomainTarget = conditioners_sec.DomainTargetConfig
NumericSafety = safety_sec.NumericSafety


@dataclasses.dataclass
class ModelsConfig(base.BaseConfigSection):
    '''Composite configuration section for `models` module.'''
    model_body: str = 'unet'
    model_body_registry: dict[str, typing.Any] = field(default_factory=default_bodies)
    bottleneck: str = 'conv'
    bottleneck_registry: dict[str, typing.Any] = field(default_factory=default_bottlenecks)
    conditioners: list[str] = field(default_factory=lambda: [])
    conditioner_registry: dict[str, typing.Any] = field(default_factory=default_conditioners)
    numeric_safety: NumericSafety = field(default_factory=NumericSafety)

    @property
    def unet_backbone_config(self) -> UNetBackbone:
        '''Return configured UNet backbone configuration.'''
        return UNetBackbone(
            body=self.model_body_registry[self.model_body],
            bottleneck=self.bottleneck_registry[self.bottleneck]
        )

    @property
    def conditioning_config(self) -> dict[str, DomainTarget]:
        '''Return configured conditioners configuration.'''
        return {c: self.conditioner_registry[c] for c in self.conditioners}

    def set_base_channel(self, base_channel: int) -> None:
        '''Set `base channel` to the current model body.'''
        self.model_body_registry[self.model_body].base_ch = base_channel

    def validate(self) -> None:
        if not self.model_body in self.model_body_registry:
            raise ValueError(
                f'Invalid model body: {self.model_body} ',
                f'expected: {list(self.model_body_registry.keys())}'
            )

        if (
            self.conditioners and
            not all(c in self.conditioner_registry for c in self.conditioners)
        ):
            raise ValueError(
                f'Invalid conditioner(s): {self.conditioners} '
                f'expected: {list(self.conditioner_registry.keys())}'
            )

        self.numeric_safety.validate()
