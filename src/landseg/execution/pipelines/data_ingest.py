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

Prepares the world grid, materializes domain knowledge, and builds
the immutable raw block catalogue for later experiments.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.execution.pipelines.base as base
import landseg.geopipe.contracts as contracts
import landseg.geopipe.ingest as ingest


# ----- public classes
class DataIngestion(base.GeoPipeline):
    '''Data ingestion pipeline.'''

    pipeline_name: str = 'data-ingest'
    logger: ingest.IngestionLogger
    pipeline_paths: artifacts.IngestionPaths

    @property
    def upstream_paths(self) -> artifacts.HarmonizationPaths:
        '''Return upstream <data-harmonize> artifact paths.'''
        return self.artifact_paths.data_harmonization

    def run(
        self,
        harmonization_record: contracts.HarmonizationRunRecord | None = None,
    ):
        '''Run data ingestion from specified harmonization run.'''
        self._initialize_run()
        context = self._build_context(harmonization_record)

        try:
            self.logger.log_sep()
            self.logger.info(
                f'Ingesting harmonization run [{context.harmonization_run_id}]'
                f' ({context.run_uid}) into run [{self.pipeline_paths.run_id}]'
            )

            policy = (
                artifacts.LifecyclePolicy.REBUILD
                if self.config.data.ingestion.rebuild
                else artifacts.LifecyclePolicy.BUILD_IF_MISSING
            )

            ingest.run_data_ingestion(
                context,
                self.pipeline_paths,
                self.config.data.ingestion,
                policy=policy,
                logger=self.logger,
            )

        except Exception as e:
            self.logger.set_summary_status('FAILED')
            self.logger.exception(f'Ingestion pipeline failed: {e}')
            raise e

        finally:
            self.logger.update_runs_manifest(
                self.pipeline_paths.runs_manifest,
                self.pipeline_paths.effective_run_folder
            )
            self.logger.log_sep()
            self.logger.close()

    def _create_logger(self) -> ingest.IngestionLogger:
        logger = ingest.IngestionLogger(
            name=self.pipeline_name,
            log_file=self.pipeline_paths.report,
            enable_file_log=False,
        )
        logger.init_summary(run_id=self.pipeline_paths.run_id)
        return logger

    def _build_context(
        self,
        harmonization_record: contracts.HarmonizationRunRecord | None = None,
    ) -> ingest.IngestionContext:
        hm_record = harmonization_record or self._resolve_target_record()
        return ingest.build_ingestion_context(
            self.upstream_paths,
            hm_record,
            runs_manifest_fpath=self.pipeline_paths.runs_manifest,
            dataset_schema_fpath=self.pipeline_paths.data_blocks.schema,
            config=self.config.data.ingestion,
        )

    def _resolve_target_record(self) -> contracts.HarmonizationRunRecord:
        target = self.config.data.ingestion.harmonization_run or 'latest'
        batches = ingest.resolve_pending_ingestion_batches(
            self.artifact_paths.data_harmonization.runs_manifest,
            self.pipeline_paths.runs_manifest,
            target=target,
            rebuild=self.config.data.ingestion.rebuild,
        )
        return batches[0]
