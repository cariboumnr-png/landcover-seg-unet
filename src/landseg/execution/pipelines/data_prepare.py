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


# ----- public classes
class DataPreparation(base.GeoPipeline):
    '''Data preparation pipeline.'''

    pipeline_name: str = 'data-prepare'
    logger: prepare.PreparationLogger
    pipeline_paths: artifacts.PreparationPaths

    @property
    def upstream_paths(self) -> artifacts.IngestionPaths:
        '''Return upstream data ingestion artifact paths.'''
        return self.artifact_paths.data_ingestion

    def run(self) -> None:
        '''Run data preparation pipeline.'''
        self._initialize_run()
        context = self._build_context()

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
                context,
                self.pipeline_paths,
                self.config.data.preparation,
                policy=policy,
                logger=self.logger,
            )

        except Exception as e:
            self.logger.set_summary_status('FAILED')
            self.logger.error(f'Preparation pipeline failed: {e}', exc_info=True)
            raise e

        finally:
            self.logger.log_sep()
            self.logger.close()

    def _create_logger(self) -> prepare.PreparationLogger:
        logger = prepare.PreparationLogger(
            name=self.pipeline_name,
            log_file=self.pipeline_paths.report,
            enable_file_log=False,
        )
        logger.init_summary(run_id='prepare')
        return logger

    def _build_context(self) -> prepare.PreparationContext:
        return prepare.build_preparation_context(
            self.upstream_paths.windows,
            self.upstream_paths.data_blocks.catalog,
            self.upstream_paths.data_blocks.schema,
            test_catalog_fpath=self.config.data.preparation.datasetview.test_catalog
        )
