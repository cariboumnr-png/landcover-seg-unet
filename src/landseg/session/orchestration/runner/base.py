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
Orchestration runner base abstractions.

This module defines the shared execution and coordination interface for
training runners that expose training progress as a stream of epoch-level
steps.

Concrete runner implementations (e.g. continuous or curriculum-based) are
responsible for:
    - driving epoch execution through policies
    - translating internal orchestration events into public TrainingStep
      records
    - arbitrating termination and exposing a consistent generator
      contract to external consumers

This module deliberately contains no training-phase semantics and no
resume logic; such behavior is defined by concrete subclasses and
higher-level CLI orchestration.
'''

# standard imports
import abc
import dataclasses
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.session.common as common
import landseg.session.orchestration.phases as phases
import landseg.session.orchestration.policy as policy
import landseg.utils as utils

@dataclasses.dataclass
class BaseRunnerConfig:
    '''
    Configuration shared by all runner implementations.

    This configuration governs runner-level behavior and must remain
    stable across different execution styles (e.g. continuous vs
    curriculum-based).

    Attributes:

        artifacts_paths: Provider of run-level filesystem paths used for
            checkpoint persistence and artifact emission.
        verbose: Enable verbose console output during execution.
        ...
    '''
    artifacts_paths: artifacts.ResultsPaths
    verbose: bool = True
    track_mode: str = 'max'
    enable_early_stop: bool = False
    patience_epochs: int | None = 5
    delta: float | None = 0.0005

# --------------------------------Public  Class--------------------------------
class BaseRunner(abc.ABC):
    '''
    Abstract base class for generator-based training runners.

    A BaseRunner defines the public execution contract for training:
    it exposes progress as a stream of `TrainingStep` records and
    delegates all execution semantics to concrete subclasses.

    Responsibilities:
        - Own the public generator interface for training execution
        - Arbitrates run termination and guarantees emission invariants
        - Translate internal execution results into immutable steps
        - Provide common utilities for logging and checkpoint persistence

    Non-responsibilities:
        - Defining training phases or curricula
        - Managing resume / load behavior
        - Encapsulating training policy logic

    Concrete implementations (e.g. continuous or curriculum runners) must
    implement `run()` and `execute()`, defining how epochs and phases are
    orchestrated while preserving the step-stream contract.

    Termination semantics:
        - Runners are the sole authority for stopping execution
        - Termination may be requested internally (policies) or
          externally (callers), but is always enacted by the runner
        - Exactly one terminal TrainingStep is emitted on termination
    '''

    def __init__(
        self,
        epoch_runner: common.EpochEngineLike,
        base_config: BaseRunnerConfig,
        *,
        logger: utils.Logger,
    ):
        '''
        Initialize the base runner.

        This constructor configures execution-wide behaviors that are
        independent of how epochs or phases are orchestrated.

        Responsibilities:
            - Bind the underlying epoch execution engine
            - Configure logging scope and verbosity
            - Apply global logit-adjustment configuration consistently
            to trainer and evaluator policies

        Note:
            Execution flow, phase semantics, and resume behavior are not
            defined at this level and must be implemented by subclasses
            or higher-level orchestration.

        Args:
            epoch_runner:
                Engine responsible for executing individual epochs.
            config:
                Shared runner configuration.
            logger:
                Base logger instance used for structured logging.
        '''

        # parse arguments
        self.epoch_runner = epoch_runner
        self.config = base_config
        self.tracking = policy.TrackingConfig(
                track_mode=self.config.track_mode,
                enable_early_stop=self.config.enable_early_stop,
                patience_epochs=self.config.patience_epochs,
                delta=self.config.delta,
            )
        # a child from base logger
        self.logger = logger.get_child('phase')
        # internal tracking attributes
        self._is_phase_end: bool = False
        self._current_epoch: int = -1
        self._current_metrics: core.EpochResults = core.EpochResults() # epoch

    @property
    def trainer(self) -> common.EngineBaseLike:
        '''Return training policy engine.'''
        assert self.epoch_runner.trainer # typing
        return self.epoch_runner.trainer

    @property
    def evaluator(self) -> common.EngineBaseLike:
        '''Return evaluating policy engine.'''
        assert self.epoch_runner.evaluator # typing
        return self.epoch_runner.evaluator

    @property
    def paths(self) -> artifacts.ResultsPaths:
        '''Return the run-level artifacts file paths class.'''
        return self.config.artifacts_paths

    @abc.abstractmethod
    def run(self) -> typing.Generator[core.TrainingSessionStep, None, None]:
        '''
        Execute training as a stream of TrainingStep records.

        Implementations must:
            - Yield exactly one TrainingStep per completed epoch
            - Yield exactly one terminal TrainingStep per run
            - Never expose partial or in-progress execution state
            - Preserve generator semantics under normal completion and
            early termination

        This method defines the primary execution interface consumed by
        external systems (CLI pipelines, sweep controllers, debuggers).
        '''

    def execute(self) -> float:
        '''
        Execute training in a blocking manner and return a scalar result.

        This is a convenience adapter that fully consumes the step stream
        produced by run() and derives a terminal scalar value, typically
        used by CLI workflows and legacy integrations.

        Returns:
            float: Final target metric achieved at run termination.
        '''

        # init a JSON artifact to store step results
        ctrl = artifacts.Controller[list[dict]](self.paths.step_results)
        steps: list[dict] = []

        # tracking
        last_step: core.TrainingSessionStep | None = None
        # consume self.run()
        for step in self.run():
            last_step = step
            steps.append(dataclasses.asdict(step))

        # persist the JSON
        ctrl.persist(steps)

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
        *,
        epoch_idx: int,
        total_epochs: int,
        best_so_far: tuple[int, float],
        metrics: core.EpochResults
    ) -> None:
        '''Parse and log a concise epoch results summary to console.'''

        # training metrics is always neends
        assert metrics.training
        # validation may or may not be run every epoch
        if metrics.validation:
            mean_iou = metrics.validation.target_metrics
        else:
            mean_iou = 0.0
        # best so far
        best_epoch, best_value = best_so_far
        msg = (
            f'|Total Loss: {metrics.training.mean_total_loss:.4f}|'
            f'Mean IoU: {mean_iou:.4f}|'
            f'Best Epoch: {best_epoch}|Best Value: {best_value:.4f}|'
        )
        t = total_epochs
        n = len(str(t))
        self.logger.log('INFO', f'[Epoch {epoch_idx:0{n}d}/{t}] {msg}')

    def _save_progress(
        self,
        phase_name: str,
        metrics: core.EpochResults,
        *,
        is_best: bool
    ) -> None:
        '''
        Persist the current training state to disk.

        This method saves a checkpoint representing the most recently
        completed epoch for the given phase. A "*last*" checkpoint is
        always written, and a corresponding "*best*" checkpoint is
        optionally written when the current state is designated as the
        best observed so far.

        The checkpoint includes:
            - Model parameters
            - Optimizer state
            - Scheduler state
            - Minimal training metadata required for resumption

        Args:
            phase_name: Name of the phase associated with the current
                training state. This value is incorporated into the
                checkpoint filename for disambiguation.
            is_best: Whether the current training state should also be
                recorded as the best-performing checkpoint for the phase.
        '''

        # build checkpoint meta dict
        validation = metrics.validation
        ckpt_meta: artifacts.CheckpointMeta = {
            'metric': validation.target_metrics if validation else 0.0,
            'epoch': self.trainer.state.progress.epoch,
            'step': self.trainer.state.progress.global_step
        }
        ep = self.trainer.state.progress.epoch
        # always save a '*last.pt'
        fp = self.paths.last_checkpoint(f'{phase_name}_epoch_{ep}')
        artifacts.save_checkpoint(
            model=self.trainer.model,
            fpath=fp,
            ckpt_meta=ckpt_meta,
            optimizer=self.trainer.optimization.optimizer,
            scheduler=self.trainer.optimization.scheduler,
        )
        self.logger.log('DEBUG', f'Checkpoint saved: {fp}')
        # if this is also the best, save/overwrite the '*.best.pt'
        if is_best:
            fp = self.paths.best_checkpoint(f'{phase_name}_epoch_{ep}')
            artifacts.save_checkpoint(
                model=self.trainer.model,
                fpath=fp,
                ckpt_meta=ckpt_meta,
                optimizer=self.trainer.optimization.optimizer,
                scheduler=self.trainer.optimization.scheduler,
            )
            self.logger.log('DEBUG', f'Checkpoint saved: {fp}')

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
