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
Evaluating a model.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.execution.pipelines.base as base
import landseg.session as session


class ModelEvaluation(base.Pipeline[session.SessionLogger]):
    '''Model evaluation pipeline.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipeline_paths = self.artifact_paths.session
        self.pipeline_paths.init_pipeline_folders()

        self.logger = session.SessionLogger(
            name='session',
            log_file=self.pipeline_paths.summary,
            console_lvl=20,
            enable_file_log=False
        )
        self.logger.init_summary(
            run_id=self.pipeline_paths.run_id,
            pipeline=self.config.pipeline.name,
        )

        # persist running config as JSON
        config_ctrl = artifacts.Controller[dict](self.pipeline_paths.config)
        config_ctrl.persist(self.config.as_dict)

    def run(self) -> float:
        '''Run model evaluation pipeline.'''

        try:
            self.logger.log_sep()

            # parse evaluation pipeline configs
            eval_config = self.config.pipeline.model_evaluate
            assert eval_config.checkpoint
            if eval_config.split not in ('val', 'test'):
                raise ValueError(f"Invalid split: {eval_config.split}")

            runner = self.build_session_runner(
                mode_override='evaluate',
                eval_split=eval_config.split
            )

            self.logger.set_inputs({
                'checkpoint': eval_config.checkpoint,
                'evaluation_split': eval_config.split
            })

            # evaluate
            evaluation_results = runner.run_epoch(0) # will always run
            assert evaluation_results.validation
            _metrics = evaluation_results.validation.head_metrics
            metrics = {h: m.as_dict for h, m in _metrics.items()}

            # persist the validation log as the current outputs
            output_ctrl = artifacts.Controller[dict](self.pipeline_paths.evaluation)
            output_ctrl.persist(metrics)

            # update summary
            self.logger.set_summary_status('SUCCESS')
            self.logger.set_results({'final': evaluation_results.target_metrics})

        except Exception as e:
            self.logger.set_summary_status('FAILED')
            self.logger.log('ERROR', f'Evaluation pipeline failed: {e}', exc_info=True)
            raise e

        finally:
            self.logger.log_sep()
            self.logger.close()

        return evaluation_results.target_metrics

    def validate(self) -> None: ...
