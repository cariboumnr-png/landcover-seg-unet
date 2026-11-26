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
Model training pipeline.

Builds data specifications from prepared artifacts, assembles the
model and session runner, and executes training and post-run visuals.
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
import landseg.visual as visual


# ----- public classes
class ModelTraining(base.SessionPipeline):
    '''Model train pipeline runner class.'''
    pipeline_name: str = 'model-train'

    @property
    def upstream_paths(self) -> artifacts.PreparationPaths:
        '''Return upstream data preparation artifact paths.'''
        return self.artifact_paths.data_preparation

    def run(self) -> None:
        '''
        Initialize a pipeline runner and run training end-to-end.

        Raises:
            Exception:
                If an error occurs during model training.
        '''
        self._initialize_run()

        try:
            runner = self.build_session_runner()

            self.logger.info('[START] Training session')

            start_t = time.perf_counter()
            final = runner.execute()
            t = time.perf_counter() - start_t
            self.timer['exec'] = t

            self.logger.info(f'[COMPLETE] Training session (D_{t:.2f}s)')
            self.logger.set_status('SUCCESS')
            self.logger.set_results(self._summarize_results(final))

            self._generate_post_run_visuals()

        except Exception as e:
            self.logger.set_status('FAILED')
            self.logger.error(f'Training pipeline failed: {e}', exc_info=True)
            raise e

        finally:
            self.logger.log_sep()
            self.logger.close()

    def _create_logger(self) -> session.SessionLogger:
        '''Instantiate and configure the training session logger.'''
        logger = session.SessionLogger(
            name=self.pipeline_name,
            log_file=self.pipeline_paths.report,
            console_lvl=self.console_level,
            enable_file_log=False,
        )
        logger.init_summary(
            run_id=self.pipeline_paths.run_id,
            command=self.config.command,
        )
        return logger

    def _build_context(self) -> core.DataSpecs:
        '''Build runtime data specifications from prepared artifacts.'''
        return geopipe.build_dataspec(
            self.artifact_paths.data_preparation,
            mode='default',
            ids_domain_name=self.config.data.specification.domain_ids_name,
            vec_domain_name=self.config.data.specification.domain_vec_name,
            knowledge_root=self.artifact_paths.knowledge_root,
        )

    def _summarize_results(self, final: float) -> dict[str, typing.Any]:
        '''Summarize training score, stage timings, and peak memory.'''
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

    def _generate_post_run_visuals(self) -> None:
        '''Plot training curves and test prediction previews.'''
        # 1. Training curves plot
        if self.pipeline_paths.step_results.is_file():
            try:
                curves_path = self.pipeline_paths.plots / 'training_curves.png'
                visual.plot_training_curves(
                    self.pipeline_paths.step_results,
                    save_path=curves_path,
                    show=False,
                )
                self.logger.info(f'[OUTPUT] Saved training curves plot to {curves_path}')
            except (ValueError, KeyError, OSError, RuntimeError, artifacts.ArtifactError) as err:
                self.logger.warning(f'Could not generate training curves plot: {err}')

        # 2. Inferred preview PNGs using best checkpoint
        dataspecs = getattr(self, 'context', None)
        if dataspecs is None:
            try:
                dataspecs = self._build_context()
            except (ValueError, KeyError, OSError, RuntimeError, artifacts.ArtifactError):
                dataspecs = None

        if dataspecs is None or not getattr(dataspecs.splits, 'test', None):
            self.logger.info(
                'No test dataset provided or split; skipping post-training preview generation.'
            )
            return

        try:
            best_ckpts = list(self.pipeline_paths.checkpoints.glob('*_best.pt'))
            if not best_ckpts:
                self.logger.warning('No best checkpoint found for preview generation.')
                return
            best_ckpt = max(best_ckpts, key=lambda p: p.stat().st_mtime)

            model = self._build_model(dataspecs)
            infer_results = session.run_preview_inference(
                checkpoint_fpath=best_ckpt,
                dataspecs=dataspecs,
                model=model,
                config=self.config.session,
                device=self.config.session.device,
                logger=self.logger,
            )
            if infer_results is None:
                return

            previews = visual.generate_prediction_previews(
                infer_results=infer_results,
                dataspecs=dataspecs,
                previews_dpath=self.pipeline_paths.previews,
            )
            if previews:
                self.logger.info(
                    f'[OUTPUT] Generated {len(previews)} test preview PNG(s) in '
                    f'{self.pipeline_paths.previews}'
                )
        except (ValueError, KeyError, OSError, RuntimeError, artifacts.ArtifactError) as err:
            self.logger.warning(f'Could not generate preview PNGs: {err}')
