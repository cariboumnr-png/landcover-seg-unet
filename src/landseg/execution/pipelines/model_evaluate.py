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
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.execution.pipelines.base as base
import landseg.geopipe as geopipe
import landseg.session as session
import landseg.visual as visual


# ----- public classes
class ModelEvaluation(base.SessionPipeline):
    '''Model evaluation pipeline.'''

    pipeline_name: str = 'model-evaluate'

    def run(self) -> float:
        '''Run model evaluation pipeline.'''
        self._initialize_run()

        try:
            self.logger.log_sep()

            eval_config = self.config.session.evaluation
            runner = self.build_session_runner(
                mode_override='evaluate',
                eval_split=eval_config.valid_split,
            )
            runner.set_head_state()

            self.logger.set_inputs({
                'checkpoint': eval_config.checkpoint,
                'evaluation_split': eval_config.split
            })

            # evaluate
            results = runner.run_epoch(0) # will always run
            assert results.validation
            _metrics = results.validation.head_metrics
            metrics = {h: m.as_dict for h, m in _metrics.items()}

            # persist the validation log as the current outputs
            Ctrl = artifacts.Controller[dict]
            Ctrl(self.pipeline_paths.evaluation).persist(metrics)

            # update summary
            self.logger.set_status('SUCCESS')
            self.logger.set_results({'final': results.target_metrics})

            self._generate_evaluation_visuals(results, runner)

        except Exception as e:
            self.logger.set_status('FAILED')
            self.logger.error(f'Evaluation pipeline failed: {e}', exc_info=True)
            raise e

        finally:
            self.logger.log_sep()
            self.logger.close()

        return results.target_metrics

    def _generate_evaluation_visuals(
        self,
        results: core.SessionStepSummary,
        runner: typing.Any,
    ) -> None:
        '''Generate diagnostic evaluation plots (confusion matrix and per-class metrics).'''
        logger = getattr(self, 'logger', None)
        if not results.validation or not results.validation.head_metrics:
            return

        plots_dir = (
            getattr(self.pipeline_paths, 'plots', None) or
            (self.pipeline_paths.effective_run_folder / 'plots')
        )
        plots_dir.mkdir(parents=True, exist_ok=True)

        for head_name, h_metrics in results.validation.head_metrics.items():
            try:
                per_cls = (
                    getattr(h_metrics, 'class_iou', None) or
                    getattr(h_metrics, 'per_class_iou', None)
                )
                if per_cls is not None:
                    p_path = plots_dir / f'{head_name}_per_class_iou.png'
                    visual.plot_per_class_metrics(
                        per_cls,
                        metric_name='IoU',
                        title=f'Per-Class IoU - [{head_name}]',
                        save_path=p_path,
                        show=False,
                    )
                    if logger:
                        logger.info(f'[OUTPUT] Saved per-class IoU plot to {p_path}')
            except Exception as err:
                if logger:
                    logger.warning(
                        f'Could not plot per-class metrics for {head_name}: {err}'
                    )

            try:
                cm_matrix = getattr(h_metrics, 'confusion_matrix', None)
                if cm_matrix is None and hasattr(runner, 'evaluator'):
                    cm_obj = getattr(runner.evaluator, 'metrics', {}).get(head_name, None)
                    cm_matrix = getattr(cm_obj, 'matrix', None)

                if cm_matrix is not None:
                    cm_path = plots_dir / f'{head_name}_confusion_matrix.png'
                    visual.plot_confusion_matrix(
                        cm_matrix,
                        title=f'Confusion Matrix - [{head_name}]',
                        save_path=cm_path,
                        show=False,
                    )
                    if logger:
                        logger.info(
                            f'[OUTPUT] Saved confusion matrix plot to {cm_path}'
                        )
            except Exception as err:
                if logger:
                    logger.warning(
                        f'Could not plot confusion matrix for {head_name}: {err}'
                    )

    def _create_logger(self) -> session.SessionLogger:
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
        dataspecs = geopipe.build_dataspec(
            self.artifact_paths.data_preparation,
            mode='default',
            ids_domain_name=self.config.data.specification.domain_ids_name,
            vec_domain_name=self.config.data.specification.domain_vec_name,
            knowledge_root=self.artifact_paths.knowledge_root,
        )

        eval_config = self.config.session.evaluation
        split_dict = getattr(dataspecs.splits, eval_config.split, None)
        if not split_dict:
            raise RuntimeError(
                f'Evaluation split "{eval_config.split}" has no blocks '
                'in prepared dataset.'
            )

        return dataspecs
