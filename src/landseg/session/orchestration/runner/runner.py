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
import landseg.session.engine as engine
import landseg.session.orchestration.event as event
import landseg.session.orchestration.phases as phases
import landseg.session.orchestration.policy as policy
import landseg.utils as utils

@dataclasses.dataclass
class RunnerConfig:
    '''Configuration for TrainingRunner execution.'''
    artifacts_paths: artifacts.ResultsPaths
    resume_from_last: bool = False
    verbose: bool = True
    enable_early_stop: bool = True
    track_mode: str = 'max'
    patience_epochs: int | None = 5
    delta: float | None = 0.005

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
        epoch_runner: engine.TrainingEpochRunner,
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
    def trainer(self) -> engine.MultiHeadTrainer:
        '''Return training policy engine.'''
        return self.epoch_runner.trainer

    @property
    def evaluator(self) -> engine.MultiHeadEvaluator | None:
        '''Return evaluating policy engine.'''
        return self.epoch_runner.evaluator

    @property
    def paths(self) -> artifacts.ResultsPaths:
        '''Return the run-level artifacts file paths class.'''
        return self.config.artifacts_paths

    def run(self) -> typing.Generator[event.Event, None, float]:
        '''
        Execute the configured training curriculum.

        Iterates through each phase, consuming its event stream and
        handling side effects such as logging, checkpointing, and early
        termination.

        Yields:
            event.Event: The immutable event stream emitted by the
                policies, passed upward for top-level observers (e.g.,
                Optuna pruners).

        Returns:
            float: The final target metric achieved after the curriculum
                completes.

        Behavior:
            - Processes phases sequentially
            - Consumes event streams generated by phase policies
            - Logs phase and epoch boundaries
            - Loads and resumes from the last checkpoint if configured
            - Persists checkpoints on request
            - Yields raw orchestration events as a generator
            - Terminates early only in single-phase mode when early
              stopping is triggered
            - Returns the definitive curriculum scalar for hyperparameter
                sweeping
        '''
        final_curriculum_metric = -float('inf')

        # iterate through provided phases
        for phase in self.phases:

            # resuming behaviour
            # temporary file path check here
            fp = self.paths.last_checkpoint(f'{phase.name}')
            if os.path.exists(fp) and self.config.resume_from_last:
                last_epoch = self._load_progress(fp)
                self.logger.log('INFO', f'Resumed from epoch {last_epoch}')
            else:
                last_epoch = 1

            # get phase events stream
            events_stream = policy.PhasePolicy(
                epoch_runner=self.epoch_runner,
                phase_config=phase,
                track_config=self.tracking_config,
                start_epoch=last_epoch
            ).run()
            phase_best_metric = -float('inf')

            # manually advance the generator to capture both yields and returns
            while True:
                try:
                    e = next(events_stream)

                    # runner intercepts events to perform side effects
                    match e:
                        case event.PhaseStart(phase_name=phase_name):
                            m = f'__Phase [{phase_name}] started__'
                            self.logger.log('INFO', m)
                            if self.config.verbose:
                                self._print_phase(phase)

                        case event.EpochStart(epoch_index=epoch_index):
                            m = f'__Epoch: {epoch_index}/{phase.num_epochs}__'
                            self.logger.log('INFO', m)

                        case event.EpochEnd(metrics=metrics):
                            # to be parsed to formal logging
                            m = metrics
                            self.logger.log('INFO', m)

                        case event.CheckpointRequest(tag=tag):
                            if tag == 'best':
                                fp = self.paths.best_checkpoint(f'{phase.name}')
                            else:
                                fp = self.paths.last_checkpoint(f'{phase.name}')
                            self._save_progress(fp)
                            m = f'Checkpoint saved at: {fp}'
                            self.logger.log('INFO', m)

                        case event.StopRun(reason=reason):
                            if not self.allow_early_stop:
                                m = f'Ignored StopRun event: {reason}'
                                self.logger.log('WARNING', m)
                                # do not yield ignored stops to observers
                                continue

                            m = f'Training stopping due to: {reason}'
                            self.logger.log('INFO', m)

                            # yield the final event, then return immediately
                            yield e
                            return phase_best_metric

                        case event.PhaseEnd(phase_name=phase_name):
                            m = f'__Phase [{phase_name}] ended__'
                            self.logger.log('INFO', m)

                    # Pass the event UP to the top-level observer (e.g. Optuna)
                    yield e

                except StopIteration as exc:
                    # capture the definitive scalar returned by PhasePolicy
                    phase_best_metric = exc.value
                    break

            # update the overall curriculum metric
            final_curriculum_metric = phase_best_metric

        return final_curriculum_metric

    def execute(self):
        '''Execute the full training curriculum in a blocking manner.'''
        return self.run()

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
            optimizer=self.trainer.optimization.optimizer,
            scheduler=self.trainer.optimization.scheduler,
        )
        self.trainer.state.progress.epoch = meta['epoch'] + 1
        self.trainer.state.progress.global_step = meta['step']
        self.trainer.state.metrics.best_value = meta['metric']
        return meta['epoch']

    def _save_progress(self, fpath: str) -> None:
        '''
        Persist current training state to a checkpoint.

        Saves model parameters along with optimizer state, scheduler
        state, and training metadata required for resumption.
        '''

        ckpt_meta: artifacts.CheckpointMeta = {
            'metric': self.trainer.state.metrics.curr_value,
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
