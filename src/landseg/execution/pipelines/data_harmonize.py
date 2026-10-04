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
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize as harmonize


# ----- public classes
class DataHarmonization(base.Pipeline):
    '''Data harmonization pipeline.'''

    pipeline_name: str = 'data-harmonize'
    context: harmonize.HarmonizationContext | None
    logger: harmonize.HarmonizationLogger | None
    pipeline_paths: artifacts.HarmonizationPaths

    def _create_logger(self) -> harmonize.HarmonizationLogger:
        '''Instantiate and configure the harmonization logger.'''
        logger = harmonize.HarmonizationLogger(
            name='data-harmonize',
            log_file=self.pipeline_paths.report,
            enable_file_log=False,
        )
        logger.init_summary(run_id=self.pipeline_paths.run_id)
        return logger

    def run(self) -> None:
        '''Execute data harmonization.'''
        if self.context is None:
            self.validate()

        self._initialize_run()
        assert self.logger is not None

        try:
            assert self.context is not None
            self.logger.log_sep()
            self.logger.log('INFO', '[START] Data harmonization')
            harmonize.run_data_harmonization(
                self.context,
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

    def validate(self) -> None:
        '''Check world-grid status and build harmonization context.'''
        invalid_upstream = False
        cfg = self.config.data.world_grid
        report_fp = geo_core.get_grid_report_fpath(cfg.output_dpath)
        try:
            status, _ = geo_core.read_grid_report(report_fp)
            if status != 'SUCCESS':
                invalid_upstream = True
        except artifacts.ArtifactError:
            invalid_upstream = True

        if invalid_upstream:
            raise RuntimeError(
                'Upstream pipeline "world-grid" has not been successfully '
                f'executed yet. Try see its report here: {report_fp}'
            )

        self.context = harmonize.build_harmonization_context(
            cfg.output_dpath,
            self.pipeline_paths.runs_manifest,
            self.config.data.harmonization
        )
