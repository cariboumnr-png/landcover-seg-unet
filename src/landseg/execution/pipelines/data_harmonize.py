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
Data harmonization pipeline command implementation.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.execution.pipelines.base as base
import landseg.geopipe.harmonize as harmonize


class DataHarmonization(base.Pipeline):
    '''Data harmonziation pipeline.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipeline_paths = self.artifact_paths.data_harmonization
        self.pipeline_paths.init_pipeline_folders()

        self.logger = harmonize.HarmonizationLogger(
            name='data-harmonize',
            log_file=self.pipeline_paths.report,
            enable_file_log=False
        )
        self.logger.init_summary(run_id=self.pipeline_paths.run_id)

        # persist running config as JSON
        config_ctrl = artifacts.Controller[dict](self.pipeline_paths.config)
        config_ctrl.persist(self.config.as_dict)

    def run(self) -> None:
        '''Execute data harmonziation.'''
        try:
            self.logger.log_sep()
            self.logger.log('INFO', '[START] Data harmonization')
            harmonize.run_data_harmonization(
                self.config.data.world_grid.output_dpath,
                self.pipeline_paths,
                self.config.data.harmonization,
                logger=self.logger
            )
            self.logger.log('INFO', '[COMPLETE] Data harmonization')

        except Exception as e:
            self.logger.set_summary_status('FAILED')
            self.logger.log('ERROR', f'Data harmonization failed: {e}')
            raise

        finally:
            self.logger.update_runs_manifest(
                self.pipeline_paths.runs_manifest,
                self.pipeline_paths.effective_run_folder
            )
            self.logger.log_sep()
            self.logger.close()

    def validate(self) -> None: ...
