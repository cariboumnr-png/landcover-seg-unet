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
Data ingestion pipeline.

Ingests harmonized raster batches into the canonical raw data block
pool and updates ingestion run metadata.
'''

# standard imports
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.execution.pipelines.base as base
import landseg.geopipe as geopipe


# ----- public classes
class DataIngestion(base.GeoPipeline):
    '''Data ingestion pipeline.'''
    pipeline_name: str = 'data-ingest'
    pipeline_paths: artifacts.IngestionPaths

    @property
    def upstream_paths(self) -> artifacts.HarmonizationPaths:
        '''Return upstream `data-harmonize` artifact paths.'''
        return self.artifact_paths.data_harmonization

    def run(
        self,
        harmonization_record: typing.Mapping[str, typing.Any] | None = None,
    ) -> None:
        '''
        Run data ingestion from a specified harmonization run.

        Args:
            harmonization_record:
                Optional harmonization run metadata record to ingest.
                If None, resolves target run from configuration.
        '''
        self._initialize_run()
        ing_cfg = self.config.data.ingestion
        geopipe.run_data_ingestion_stage(
            ing_cfg,
            self.upstream_paths,
            self.pipeline_paths,
            harmonization_run=ing_cfg.harmonization_run,
            rebuild=ing_cfg.rebuild,
            harmonization_record=harmonization_record,
        )
