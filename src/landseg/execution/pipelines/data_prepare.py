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
import landseg.geopipe.prepare as prepare


class DataPreparation(base.Pipeline[prepare.PreparationLogger]):
    '''Data preparetion pipline.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.upstream_paths = self.artifact_paths.data_ingestion
        self.pipeline_paths = self.artifact_paths.data_preparation
        self.pipeline_paths.init_pipeline_folders()

        self.logger = prepare.PreparationLogger(
            name='data-prep',
            log_file=self.pipeline_paths.report,
            enable_file_log=False
        )
        self.logger.init_summary(run_id='prepare')

        # persist running config as JSON
        config_ctrl = artifacts.Controller[dict](self.pipeline_paths.config)
        config_ctrl.persist(self.config.as_dict)

    def run(self):
        '''Run data preparation pipeline'''
        try:
            self.logger.log_sep()

            # resolve lifecycle policy dynamically
            policy = (
                artifacts.LifecyclePolicy.REBUILD
                if self.config.data.preparation.rebuild
                else artifacts.LifecyclePolicy.BUILD_IF_MISSING
            )

            # run pipeline
            prepare.run_data_preparation(
                self.artifact_paths,
                self.config.data.preparation,
                self.config.data.world_grid.tile_specs_tuple,
                policy=policy,
                logger=self.logger,
            )

        except Exception as e:
            self.logger.set_summary_status('FAILED')
            self.logger.log('ERROR', f'Preparation pipeline failed: {e}', exc_info=True)
            raise e

        finally:
            self.logger.log_sep()
            self.logger.close()

    def validate(self) -> None: ...
