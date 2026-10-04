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

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.execution.pipelines.base as base
import landseg.geopipe as geopipe
import landseg.session as session


# ----- public classes
class ModelEvaluation(base.Pipeline):
    '''Model evaluation pipeline.'''

    pipeline_name: str = 'model-evaluate'
    context: None
    logger: session.SessionLogger | None
    pipeline_paths: artifacts.SessionPaths

    def run(self) -> float:
        '''Run model evaluation pipeline.'''
        if self.dataspecs is None:
            self.validate()

        self._initialize_run()
        assert self.logger is not None

        try:
            self.logger.log_sep()

            # parse evaluation pipeline configs
            eval_config = self.config.command.model_evaluate
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

    def validate(self) -> None:
        '''Validate model evaluation prerequisites and build dataspecs.'''
        eval_config = self.config.command.model_evaluate
        if not eval_config.checkpoint or not os.path.exists(eval_config.checkpoint):
            raise FileNotFoundError(
                f'Evaluation checkpoint not found: {eval_config.checkpoint}'
            )

        if eval_config.split not in ('val', 'test'):
            raise ValueError(f'Invalid split: {eval_config.split}')

        # verify upstream data-prepare report
        report_fp = self.artifact_paths.data_preparation.report
        report_ctrl = artifacts.Controller[dict].load_json_or_fail(report_fp)
        try:
            report = report_ctrl.fetch()
        except artifacts.ArtifactError as e:
            raise RuntimeError(
                'Upstream pipeline "data-prepare" has not been executed yet. '
                f'Missing report at: {report_fp}'
            ) from e

        if report.get('status') != 'SUCCESS':
            status_val = report.get('status')
            raise RuntimeError(
                f'Upstream pipeline "data-prepare" status is "{status_val}".'
            )

        self.dataspecs = geopipe.build_dataspec(
            self.artifact_paths,
            mode='default',
            ids_domain_name=self.config.data.specification.domain_ids_name,
            vec_domain_name=self.config.data.specification.domain_vec_name,
        )

        split_dict = getattr(self.dataspecs.splits, eval_config.split, None)
        if not split_dict:
            raise RuntimeError(
                f'Evaluation split "{eval_config.split}" has no blocks '
                'in prepared dataset.'
            )

    def _create_logger(self) -> session.SessionLogger:
        '''Instantiate and configure the session logger.'''
        logger = session.SessionLogger(
            name='session',
            log_file=self.pipeline_paths.report,
            console_lvl=self.console_level,
            enable_file_log=False,
        )
        logger.init_summary(
            run_id=self.pipeline_paths.run_id,
            command=self.config.command.name,
        )
        return logger
