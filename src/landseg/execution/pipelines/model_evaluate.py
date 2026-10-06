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
import landseg.core as core
import landseg.execution.pipelines.base as base
import landseg.geopipe as geopipe
import landseg.session as session


# ----- public classes
class ModelEvaluation(base.SessionPipeline):
    '''Model evaluation pipeline.'''

    pipeline_name: str = 'model-evaluate'

    def run(self) -> float:
        '''Run model evaluation pipeline.'''
        self._initialize_run()

        eval_config = self.config.command.model_evaluate

        try:
            self.logger.log_sep()

            runner = self.build_session_runner(
                mode_override='evaluate',
                eval_split=eval_config.valid_split,
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
        dataspecs = geopipe.build_dataspec(
            self.artifact_paths,
            mode='default',
            ids_domain_name=self.config.data.specification.domain_ids_name,
            vec_domain_name=self.config.data.specification.domain_vec_name,
        )

        eval_config = self.config.command.model_evaluate
        split_dict = getattr(dataspecs.splits, eval_config.split, None)
        if not split_dict:
            raise RuntimeError(
                f'Evaluation split "{eval_config.split}" has no blocks '
                'in prepared dataset.'
            )

        return dataspecs
