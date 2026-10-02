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
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.execution.pipelines.base as base
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session


class ModelTraining(base.Pipeline):
    '''Model train pipeline runner class.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipeline_paths = self.artifact_paths.session
        self.pipeline_paths.init_pipeline_folders()

        self.logger = session.SessionLogger(
            name='session',
            log_file=self.pipeline_paths.summary,
            console_lvl=self.console_level,
            enable_file_log=False
        )
        self.logger.init_summary(
            run_id=self.pipeline_paths.run_id,
            pipeline=self.config.pipeline.name,
        )

        # persist running config as JSON
        config_ctrl = artifacts.Controller[dict](self.pipeline_paths.config)
        config_ctrl.persist(self.config.as_dict)

    def run(self) -> None:
        '''Initialize a pipeline runner and run training end-to-end.'''
        try:
            runner = self.build_runner()

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

    def validate(self) -> None: ...

    @typing.overload
    def build_runner(
        self,
        *,
        mode_override: typing.Literal['continuous'],
    ) -> session.ContinuousRunner: ...


    @typing.overload
    def build_runner(
        self,
        *,
        mode_override: typing.Literal['curriculum'],
    ) -> session.CurriculumRunner: ...

    @typing.overload
    def build_runner(
        self,
        *,
        mode_override: None = None,
    ) -> session.ContinuousRunner | session.CurriculumRunner: ...

    def build_runner(
        self,
        *,
        mode_override: typing.Literal['continuous', 'curriculum'] | None = None,
    ) -> session.ContinuousRunner | session.CurriculumRunner:
        '''doc.'''
        # collect artifacts and build `DataSpecs`
        self.logger.log('INFO', '[START] Data specifications setup')
        start_t = time.perf_counter()
        dataspecs = geopipe.build_dataspec(
            self.artifact_paths,
            mode='default',
            ids_domain_name=self.config.data.specification.domain_ids_name,
            vec_domain_name=self.config.data.specification.domain_vec_name
        )
        self.timer['data'] = time.perf_counter() - start_t
        self.logger.log('INFO', f'[COMPLETE] Data specs setup (D_{self.timer['data']:.2f}s)')

        for s in dataspecs.summary:
            self.logger.log('INFO', s)
        self.logger.log_sep()

        # setup the model
        self.logger.log('INFO', '[START] Model assembly')
        start_t = time.perf_counter()
        model = models.build_multihead_unet(
            patch_size=self.config.session.dataloader.patch_size,
            dataspecs=dataspecs,
            unet_backbone_config=self.config.models.unet_backbone_config,
            conditioning_config=self.config.models.conditioning_config,
            enable_clamp=self.config.models.numeric_safety.enable_clamp,
            clamp_range=self.config.models.numeric_safety.clamp_range
        )
        self.timer['model'] = time.perf_counter() - start_t
        self.logger.log('INFO', f'[COMPLETE] Model assembly (D_{self.timer['model']:.2f}s)')

        # summarize inputs and log
        total_p = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.set_inputs({
            'system': {
                'device':c.DEVICE_NAME,
                'torch_version': torch.__version__
            },
            'model': {
                'backbone': 'unet',
                'total_parameters': total_p,
                'trainable_parameters': trainable_p,
                'heads': list(dataspecs.heads.class_counts.keys())
            },
            'data': {
                'patch_size': self.config.session.dataloader.patch_size
            },
            'dataspecs': dataspecs.to_dict()
        })
        self.logger.log_sep()

        # build and return the session runner
        match mode_override:
            case 'continuous': session_type = 'continuous'
            case 'curriculum': session_type = 'curriculum'
            case None: session_type = self.config.session.training_mode

        runner = session.build_session_runner(
            dataspecs=dataspecs,
            model=model,
            config=self.config.session,
            context=session.SessionBuildContext(
                device=c.DEVICE,
                session_paths=self.pipeline_paths,
                eval_dataset='val',
                logger=self.logger
            ),
            session_type=session_type,
        )
        return runner

    def _summarize_results(self, final: float) -> dict[str, typing.Any]:
        '''Summarize peak memory and log final results and metrics.'''
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
