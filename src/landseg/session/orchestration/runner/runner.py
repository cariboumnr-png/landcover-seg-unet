# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Curriculum-based training runner for multi-phase model execution.

The TrainingRunner orchestrates a sequence of training phases over a
shared training engine. Each phase defines its own training configuration,
including active heads, learning rate scaling, and scheduling behavior.

The runner consumes an event stream produced by each phase policy,
handling logging, checkpointing, and control flow, while yielding
epoch-level metrics as a generator.

Early stopping is supported only for single-phase runs and is enforced
at configuration time.

Phase progress is persisted to disk, enabling safe resumption of
interrupted experiments.
'''

# standard imports
import dataclasses
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.session.common as common
import landseg.session.orchestration.events as events
import landseg.session.orchestration.phases as phases
import landseg.session.orchestration.policy as policy
import landseg.utils as utils

@dataclasses.dataclass
class RunnerConfig:
    '''Configuration for TrainingRunner execution.'''
    artifacts_paths: artifacts.ResultsPaths
    resume_from_last: bool = False
    verbose: bool = True
    track_mode: str = 'max'
    enable_early_stop: bool = True
    patience_epochs: int | None = 5
    delta: float | None = 0.005

@dataclasses.dataclass
class TrainingStep:
    '''Immutable training step result summary.'''
    phase_name: str # identity / location
    phase_index: int
    epoch: int = 1
    metrics: common.EpochMetricsLike | None = None # metrics
    is_phase_end: bool = False # termination signals
    is_run_end: bool = False
    early_stop_reason: str | None = None

# --------------------------------Public  Class--------------------------------
class TrainingRunner:
    '''
    Orchestrates curriculum-based training across one or more phases.

    The runner executes phases sequentially using a shared training
    engine. Each phase emits structured events via its policy, which the
    runner consumes to drive logging, checkpointing, and control flow.

    The runner itself is implemented as a generator, yielding metrics
    at the end of each epoch.

    Design principles:
        - Single execution model for both phased and non-phased training
        - Event-driven control flow via pattern matching
        - Separation of concerns between runner (orchestration) and
          policy (training logic)

    Early stopping:
        - Supported only for single-phase runs
        - Automatically disabled when multiple phases are configured
    '''
    def __init__(
        self,
        epoch_runner: common.EpochEngineLike,
        training_phases: typing.Sequence[phases.PhaseLike],
        config: RunnerConfig,
        *,
        logger: utils.Logger,
        **kwargs
    ):
        '''
        Initialize the training runner.

        Responsibilities:
            - Store phase sequence and training engine
            - Configure logging and verbosity
            - Validate and gate early stopping behavior
            - Initialize tracking configuration for phase policies
            - Configure logit adjustment settings for trainer/evaluator

        Early stopping:
            If multiple phases are provided, early stopping is disabled
            automatically to ensure full execution of the curriculum.

        Args:
            epoch_runner: Engine responsible for executing individual
                training epochs.
            training_phases: Ordered list of phase configurations
                defining the curriculum.
            config: Runner configuration parameters.
            logger: Base logger instance used for structured logging.
            **kwargs: Optional overrides for logit adjustment behavior.
        '''

        # parse arguments
        self.epoch_runner = epoch_runner
        self.phases = training_phases
        self.config = config

        # a child from base logger
        self.logger = logger.get_child('phase')

        # early config validation gating
        if len(training_phases) > 1 and config.enable_early_stop:
            m = (
                'Runner configured with multiple phases but "enable_early_stop"'
                'is set to "True"; Early stop mechanism will forced OFF.'
            )
            self.allow_early_stop = False
            self.logger.log('WARNING', m)
        else:
            self.allow_early_stop = True
        enable_early_stop = config.enable_early_stop and self.allow_early_stop

        # tracking config
        self.tracking_config = policy.TrackingConfig(
            enable_early_stop=enable_early_stop,
            patience_epochs=config.patience_epochs,
            track_mode=config.track_mode,
            delta=config.delta
        )

        # set la status
        self.trainer.model.set_logit_adjust_alpha(
            kwargs.get('logit_adjust_alpha', 1.0)
        )
        self.trainer.config_logit_adjust(
            enable_train_logit_adjust=kwargs.get('train_logit_adjust', True)
        )
        if self.evaluator:
            self.evaluator.model.set_logit_adjust_alpha(
                kwargs.get('logit_adjust_alpha', 1.0)
            )
            self.evaluator.config_logit_adjust(
                enable_val_logit_adjust=kwargs.get('val_logit_adjustment', True),
                enable_test_logit_adjust=kwargs.get('test_logit_adjustment', False)
            )

    @property
    def trainer(self) -> common.BatchEngineLike:
        '''Return training policy engine.'''
        assert self.epoch_runner.trainer # typing
        return self.epoch_runner.trainer

    @property
    def evaluator(self) -> common.BatchEngineLike:
        '''Return evaluating policy engine.'''
        assert self.epoch_runner.evaluator # typing
        return self.epoch_runner.evaluator

    @property
    def paths(self) -> artifacts.ResultsPaths:
        '''Return the run-level artifacts file paths class.'''
        return self.config.artifacts_paths

    def run(self) -> typing.Generator[TrainingStep, None, None]:
        '''
        Execute the configured training curriculum.

        Yields:
            TrainingStep: One per completed epoch. Exactly one terminal
            step is yielded at the end of the run.
        '''

        # tracking
        current_epoch: int = -1
        last_metrics: common.EpochMetricsLike | None = None

        # iterate through provided phases
        for phase_idx, phase in enumerate(self.phases):

            # print phase info if verbose
            if self.config.verbose:
                self._print_phase(phase)

            # resuming behaviour
            # temporary file path check here
            fp = self.paths.last_checkpoint(f'{phase.name}')
            if os.path.exists(fp) and self.config.resume_from_last:
                last_epoch = self._load_progress(fp)
                self.logger.log('INFO', f'Resumed from epoch {last_epoch}')
            else:
                last_epoch = 1
            current_epoch = last_epoch - 1

            # get phase events stream
            events_stream = policy.PhasePolicy(
                epoch_runner=self.epoch_runner,
                phase_config=phase,
                track_config=self.tracking_config,
                start_epoch=last_epoch
            ).run()

            # manually advance the generator to capture both yields and returns
            while True:

                try:
                    e = next(events_stream)
                except StopIteration:
                    # phase policy already reached the end
                    break

                # runner intercepts events to perform side effects
                match e:

                    case events.EpochEnd(epoch_index=ep, metrics=metrics):
                        # typing
                        assert isinstance(metrics, common.EpochMetricsLike)
                        #
                        current_epoch = ep
                        last_metrics = metrics
                        is_phase_end = ep==phase.num_epochs
                        self._log_metrics(ep, phase.num_epochs, metrics)
                        yield TrainingStep(
                            phase_name=phase.name,
                            phase_index=phase_idx,
                            epoch=ep,
                            metrics=metrics,
                            is_phase_end=is_phase_end
                        )
                        # exit current phase
                        if is_phase_end:
                            break

                    case events.StopRun(reason=reason):
                        if not self.allow_early_stop:
                            continue # do not yield

                        yield TrainingStep(
                            phase_name=phase.name,
                            phase_index=phase_idx,
                            epoch=current_epoch,
                            metrics=last_metrics,
                            is_run_end=True,
                            early_stop_reason=reason
                        )
                        # exit the whole run
                        return

                    case events.CheckpointRequest(tag=tag):
                        # always save current as 'last'
                        fp = self.paths.last_checkpoint(f'{phase.name}')
                        self._save_progress(fp)
                        # save another one if it is the 'best'
                        if tag == 'best':
                            fp = self.paths.best_checkpoint(f'{phase.name}')
                            self._save_progress(fp)
                        self.logger.log('DEBUG', f'Checkpoint saved: {fp}')

        # final step
        if last_metrics is not None:
            yield TrainingStep(
                phase_name=self.phases[-1].name,
                phase_index=len(self.phases) - 1,
                epoch=current_epoch,
                metrics=last_metrics,
                is_run_end=True,
                early_stop_reason=None,
            )

    def execute(self) -> float:
        '''
        Execute the training curriculum in a blocking manner.

        Returns:
            float: Final target metric from the terminal `TrainingStep`.
        '''

        # tracking
        last_step: TrainingStep | None = None
        # consume self.run()
        for step in self.run():
            last_step = step

        # exceptions handling
        if last_step is None:
            raise RuntimeError('Training produced no steps')
        if not last_step.is_run_end:
            raise RuntimeError('Training did not terminate cleanly')
        if not (last_step.metrics and last_step.metrics.validation):
            raise RuntimeError('Invalid validation results')

        # return the final scalar
        return last_step.metrics.validation.target_metrics

    def _log_metrics(
        self,
        epoch_idx: int,
        total_epochs: int,
        metrics: common.EpochMetricsLike
    ) -> None:
        '''Parse and log a concise epoch results summary to console.'''

        assert metrics.training
        assert metrics.validation
        msg = (
            f' | Total Loss: {metrics.training.mean_total_loss:.4f} | '
            f'Mean IoU {metrics.validation.target_metrics:.4f} | '
        )
        t = total_epochs
        n = len(str(t))
        self.logger.log('INFO', f'Epoch {epoch_idx:0{n}d}/{t}{msg}')

    def _load_progress(self, fpath: str) -> int:
        '''
        Restore training state from a checkpoint.

        Loads model weights, optimizer state, scheduler state, and
        tracked metrics, and updates the trainer's internal progress
        state.
        '''

        # load and return epoch number
        meta = artifacts.load_checkpoint(
            model=self.trainer.model,
            fpath=fpath,
            map_device=self.trainer.device,
            optimizer=self.trainer.comps.optimization.optimizer,
            scheduler=self.trainer.comps.optimization.scheduler,
        )
        self.trainer.state.progress.epoch = meta['epoch'] + 1
        self.trainer.state.progress.global_step = meta['step']
        self.trainer.state.progress.current_metrics = meta['metric']
        return meta['epoch']

    def _save_progress(self, fpath: str) -> None:
        '''
        Persist current training state to a checkpoint.

        Saves model parameters along with optimizer state, scheduler
        state, and training metadata required for resumption.
        '''

        ckpt_meta: artifacts.CheckpointMeta = {
            'metric': self.trainer.state.progress.current_metrics,
            'epoch': self.trainer.state.progress.epoch,
            'step': self.trainer.state.progress.global_step
        }
        artifacts.save_checkpoint(
            model=self.trainer.model,
            fpath=fpath,
            ckpt_meta=ckpt_meta,
            optimizer=self.trainer.comps.optimization.optimizer,
            scheduler=self.trainer.comps.optimization.scheduler,
        )

    @staticmethod
    def _print_phase(phase: phases.PhaseLike):
        '''Pretty print a phase to console.'''

        print('__Phase details__')
        ss = '\n'.join([
            f'- Phase Name:\t{phase.name}',
            f'- Max Epochs:\t{phase.num_epochs}',
            f'- LR Scale:\t{phase.lr_scale}',
            f'- Active Heads:\t{phase.active_heads}',
            f'- Frozen Heads:\t{phase.frozen_heads}',
        ])
        print(ss)
