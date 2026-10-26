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

'''Configuration for `geopipe.ingestion` module.'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base


POLICY = ['skip', 'overwrite', 'error']
TOPO = ['slope', 'tpi']
SPECTRAL =  ['ndvi', 'ndmi', 'nbr']


@dataclasses.dataclass
class Domains(base.BaseConfigSection):
    '''Configuration for `geopipe.ingestion.domains` module.'''
    valid_threshold: float = 0.7
    target_variance: float = 0.9

    def validate(self) -> None:
        self.require_float_max(self.valid_threshold, 'valid_threshold', 1.0)
        self.require_float_max(self.target_variance, 'target_variance', 1.0)


@dataclasses.dataclass
class DataBlocks(base.BaseConfigSection):
    '''Configuration for `geopipe.ingestion.blocks` module.'''
    ignore_index: int = 255
    image_dem_pad: int = 8
    collision_policy: str = 'skip'
    add_topo: list[str] | None = None
    add_spectral: list[str] | None = None

    def validate(self) -> None:
        self.require_int(self.ignore_index, 'ignore_index')
        self.require_int_min(self.image_dem_pad, 'image_dem_pad', 1)
        self.require_allowed_str(self.collision_policy, 'collision_policy', POLICY)

        if self.add_topo is not None:
            self.require_sequence(self.add_topo, 'add_topo')
            for v in self.add_topo:
                self.require_allowed_str(v, 'topo_feature', TOPO)

        if self.add_spectral is not None:
            self.require_sequence(self.add_spectral, 'add_spectral')
            for v in self.add_spectral:
                self.require_allowed_str(v, 'spectral_feature', SPECTRAL)


@dataclasses.dataclass
class DataIngestionConfig(base.BaseConfigSection):
    '''Configuration for `geopipe.ingestion` module.'''
    domains: Domains = dataclasses.field(default_factory=Domains)
    datablocks: DataBlocks = dataclasses.field(default_factory=DataBlocks)
    rebuild: bool = False
    harmonization_run: int | str | None = None
    output_dpath: str = '${execution.exp_root}/artifacts/ingested_data'

    def validate(self) -> None:
        self.domains.validate()
        self.datablocks.validate()

        if self.harmonization_run is not None:
            try:
                self.require_int_min(self.harmonization_run, 'harmonization_run', 1)
            except base.ConfigValidationError:
                self.require_str(self.harmonization_run, 'harmonization_run')
            except Exception as e:
                raise base.ConfigValidationError(
                    'Failed to validate "harmonization_run" attribute'
                ) from e
