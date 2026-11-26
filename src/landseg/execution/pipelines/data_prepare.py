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
Data preparation (experiment-materialized) pipeline.

Splits raw blocks into train/val(/test), computes train-only band
statistics, normalizes all splits, and emits the final dataset schema.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.execution.pipelines.base as base
import landseg.geopipe as geopipe


# ----- public classes
class DataPreparation(base.GeoPipeline):
    '''Data preparation pipeline.'''
    pipeline_name: str = 'data-prepare'
    pipeline_paths: artifacts.PreparationPaths

    @property
    def upstream_paths(self) -> artifacts.IngestionPaths:
        '''Return upstream data ingestion artifact paths.'''
        return self.artifact_paths.data_ingestion

    def run(self) -> None:
        '''Run data preparation pipeline.'''
        self._initialize_run()
        geopipe.run_data_preparation_stage(
            self.config.data.preparation,
            self.upstream_paths,
            self.pipeline_paths,
        )
