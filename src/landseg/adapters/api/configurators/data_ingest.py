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
Data ingestion configurator
'''

# standard imports
import typing
# local imports
import landseg.adapters.api.configurators as configurators

class DataIngestionConfigurator(configurators.BaseConfigurator):
    '''Configure data ingestion.'''

    def __init__(
        self,
        experiment_root: str,
    ):
        super().__init__(experiment_root, 'data-ingest')

    def set_rebuild(self, rebuild: bool) -> typing.Self:
        '''Set whether to force rebuild ingestion artifacts.'''
        self._cfg.data.ingestion.rebuild = rebuild
        return self

    def set_harmonization_run(
        self,
        target_run: int | str | None = None
    ) -> typing.Self:
        '''Set targeted harmonization run index, folder name, or path.'''
        self._cfg.data.ingestion.harmonization_run = target_run
        return self

    def set_feature_engineering(
        self,
        add_topo: list[str] | None = None,
        add_spectral: list[str] | None = None,
    ) -> typing.Self:
        '''
        Configure automated topographic and spectral features.

        Args:
            add_topo:
                List of topographic features to calculate (e.g.,
                `['slope', 'tpi']`).
            add_spectral:
                List of spectral indices to calculate (e.g.,
                `['ndvi', 'ndmi', 'nbr']`).
        '''
        self._cfg.data.ingestion.datablocks.add_topo = add_topo
        self._cfg.data.ingestion.datablocks.add_spectral = add_spectral
        return self
