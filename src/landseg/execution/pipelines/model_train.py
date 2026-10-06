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
Training entrypoint.

Builds data specifications from produced artifacts, constructs the
model, and runs the multi-phase training runner.
'''

# standard imports
import time
import typing
# third-party imports
import psutil
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.execution.pipelines.base as base
import landseg.geopipe as geopipe
import landseg.session as session


# ----- public classes
class ModelTraining(base.SessionPipeline):
    '''Model train pipeline runner class.'''

    pipeline_name: str = 'model-train'

    @property
    def upstream_paths(self) -> artifacts.PreparationPaths:
        '''Return upstream data preparation artifact paths.'''
        return self.artifact_paths.data_preparation

    def run(self) -> None:
        '''Initialize a pipeline runner and run training end-to-end.'''
        self._initialize_run()

        try:
            runner = self.build_session_runner()

            self.logger.log('INFO', '[START] Training session')
            start_t = time.perf_counter()
            final = runner.execute()
            self.timer['exec'] = time.perf_counter() - start_t
            self.logger.log('INFO', f'[COMPLETE] Training session (D_{self.timer['exec']:.2f}s)')

            self.logger.set_summary_status('SUCCESS')
            self.logger.set_results(self._summarize_results(final))

        except Exception as e:
            self.logger.set_summary_status('FAILED')
            self.logger.log('ERROR', f'Training pipeline failed: {e}', exc_info=True)
            raise e

        finally:
            self.logger.log_sep()
            self.logger.close()

    def _create_logger(self) -> session.SessionLogger:
        logger = session.SessionLogger(
            name=self.pipeline_name,
            log_file=self.pipeline_paths.report,
            console_lvl=self.console_level,
            enable_file_log=False,
        )
        logger.init_summary(
            run_id=self.pipeline_paths.run_id,
            command=self.config.command.name,
        )
        return logger

    def _build_context(self) -> core.DataSpecs:
        return geopipe.build_dataspec(
            self.artifact_paths,
            mode='default',
            ids_domain_name=self.config.data.specification.domain_ids_name,
            vec_domain_name=self.config.data.specification.domain_vec_name,
        )

    def _summarize_results(self, final: float) -> dict[str, typing.Any]:
        process = psutil.Process()
        peak_cpu_mb = process.memory_info().rss / (1024 * 1024)
        peak_gpu_mb = 0.0
        if torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        t_data = self.timer.get('data', 0.0)
        t_model = self.timer.get('model', 0.0)
        t_exec = self.timer.get('exec', 0.0)

        return {
            'best_value': final,
            'duration_sec': t_data + t_model + t_exec,
            'durations': {
                'data_specs_setup_sec': t_data,
                'model_assembly_sec': t_model,
                'execution_sec': t_exec
            },
            'system': {
                'peak_cpu_memory_mb': float(peak_cpu_mb),
                'peak_gpu_memory_mb': float(peak_gpu_mb)
            }
        }
