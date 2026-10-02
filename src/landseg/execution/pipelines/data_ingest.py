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


class DataIngestion(base.Pipeline):
    '''Data ingestion pipeline.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.upstream_paths = self.artifact_paths.data_harmonization
        self.pipeline_paths = self.artifact_paths.data_ingestion
        self.pipeline_paths.init_pipeline_folders()

        self.logger = ingest.IngestionLogger(
            name='data-ingest',
            log_file=self.pipeline_paths.report,
            enable_file_log=False
            )
        self.logger.init_summary(run_id=self.pipeline_paths.run_id)

        # persist running config as JSON
        config_ctrl = artifacts.Controller[dict](self.pipeline_paths.config)
        config_ctrl.persist(self.config.as_dict)

    def run(
        self,
        harmonization_record: contracts.HarmonizationRunRecord | None = None
    ):
        '''Run data ingestion from specified harmonziation run.'''
        try:
            self.logger.log_sep()

            # resolve the latest harmonization run if not specified
            hm_record = ingest.resolve_pending_ingestion_batches(
                self.artifact_paths.data_harmonization.runs_manifest,
                self.pipeline_paths.runs_manifest,
                target='latest',
            )[0] if harmonization_record is None else harmonization_record

            self.logger.log(
                'INFO',
                f'Ingesting harmonization run [{hm_record["run_id"]}] '
                f'({hm_record["run_uid"]})'
                f' into run [{self.pipeline_paths.run_id}]'
            )

            policy = (
                artifacts.LifecyclePolicy.REBUILD
                if self.config.data.ingestion.rebuild
                else artifacts.LifecyclePolicy.BUILD_IF_MISSING
            )

            ingest.run_data_ingestion(
                self.upstream_paths,
                self.pipeline_paths,
                hm_record,
                self.config.data.ingestion,
                policy=policy,
                logger=self.logger,
            )

        except Exception as e:
            self.logger.set_summary_status('FAILED')
            self.logger.log(
                'ERROR', f'Ingestion pipeline failed: {e}', exc_info=True
            )
            raise e

        finally:
            self.logger.update_runs_manifest(
                self.pipeline_paths.runs_manifest,
                self.pipeline_paths.effective_run_folder
            )
            self.logger.log_sep()
            self.logger.close()

    def validate(self) -> None: ...
