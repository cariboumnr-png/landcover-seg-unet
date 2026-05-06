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
Continuous training runner.

This module defines a concrete `BaseRunner` implementation that executes
a single training phase continuously. It is intended for simple,
non-curriculum training workflows where a single phase defines the full
training lifecycle.

The runner consumes events emitted by `PhasePolicy` and exposes progress
as a stream of TrainingStep records, one per completed epoch.
'''

# standard imports
import typing
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.orchestration.events as events
import landseg.session.orchestration.policy as policy
import landseg.session.orchestration.runner as runner

# --------------------------------Public  Class--------------------------------
class ContinuousRunner(runner.BaseRunner):
    '''
    Continuous (single-phase) training runner.

    This runner executes exactly one training phase using a shared epoch
    execution engine. It consumes the event stream emitted by a
    `PhasePolicy` and translates epoch-completion events into immutable
    `TrainingStep` records.

    The runner exposes training progress exclusively as a generator
    stream, yielding one `TrainingStep` per completed epoch and exactly
    one terminal `TrainingStep` at run termination.

    Characteristics:
        - Single-phase execution (no curriculum sequencing)
        - Generator-based, step-wise execution
        - No resume or load logic at this level
        - Early stopping handled entirely by the `PhasePolicy`

    This runner is ideal for:
        - Standard training jobs
        - Hyperparameter sweeps
        - Debug and development workflows
    '''

    def __init__(
        self,
        *,
        phase: common.PhaseLike,
        **kwargs: typing.Any
    ):
        '''
        Initialize a continuous training runner.

        This runner is configured with a single training phase and a
        corresponding tracking configuration. All training behavior,
        including early stopping, is delegated to the `PhasePolicy`.

        Args:
            phase: The sole training phase defining model configuration
                and maximum number of epochs.
            tracking_config: Metric tracking and early stopping config
                passed to the `PhasePolicy`.
            start_epoch: Epoch index at which execution begins. This is
                typically provided by higher-level orchestration (e.g.,
                CLI) for resumed runs.
            **kwargs: Forwarded to `BaseRunner` initialization (epoch
                runner, logger, and configuration).
        '''

        super().__init__(**kwargs)
        # parse arguments
        self.phase = phase # single phase
        # tracking
        self._best_value_so_far: float = 0.0
        self._best_epoch_so_far: int = -1
        self._is_best_epoch: bool = False

    def run(self) -> typing.Generator[core.TrainingSessionStep, None, None]:
        '''
        Execute continuous training as a stream of TrainingStep records.

        Behavior:
        - Executes a single training phase.
        - Consumes the event stream emitted by `PhasePolicy`.
        - Translates EpochEnd events into `TrainingStep` records.
        - Handles logging and checkpoint persistence as side effects.
        - Terminates only when the phase completes or an explicit
        `StopRun` event is emitted by the `PhasePolicy`.

        Notes:
        - This method does not expose partial or in-progress execution.
        - External consumers may stop consuming the stream at any time.

        Yields:
            TrainingStep:
                - One TrainingStep per completed epoch.
                - Exactly one terminal TrainingStep with
                `is_run_end == True` upon termination.
        '''

        # dispatch at phase begininng
        self.dispatcher.on_train_phase_begin(self.phase)

        # get phase events stream
        events_stream = policy.PhasePolicy(
            epoch_runner=self.epoch_runner,
            phase_config=self.phase,
            track_config=self.tracking,
        ).run()

        # manually advance the generator to capture both yields and returns
        while True:

            try:
                e = next(events_stream)
            except StopIteration:
                return # already at the end of phase

            # runner intercepts events to perform side effects
            match e:

                case events.EpochStart(epoch_index=epoch):
                    self.dispatcher.on_epoch_begin(epoch)

                case events.EpochEnd(epoch_index=epoch):

                    # epoch tracking
                    self._current_epoch = epoch
                    self._is_phase_end = epoch==self.phase.num_epochs
                    # dispatch
                    self.dispatcher.on_epoch_end(epoch)

                case events.MetricsReport(
                    best_so_far=best_so_far,
                    best_epoch=best_epoch,
                    is_best_epoch=is_best_epoch,
                    raw_metrics=metrics
                ):

                    # metrics tracking
                    self._current_metrics = metrics
                    self._best_value_so_far=best_so_far
                    self._best_epoch_so_far=best_epoch
                    self._is_best_epoch=is_best_epoch

                    # normal yield
                    if not self._is_phase_end:
                        yield self._get_step(reason=None)
                    # yield at the end
                    else:
                        reason = 'Max epoch reached'
                        self.dispatcher.on_train_phase_end(self.phase.name, reason)
                        yield self._get_step(reason=reason)
                        return

                case events.StopRun(reason=reason):

                    # yield and exit on stop signal
                    self.dispatcher.on_train_phase_end(self.phase.name, reason)
                    yield self._get_step(reason=reason)
                    return

                case events.CheckpointRequest(tag=tag):
                    self._save_progress(
                        self.phase.name,
                        self._current_metrics,
                        is_best=tag=='best'
                    )

    def _get_step(self, reason: str | None = None) -> core.TrainingSessionStep:
        '''Helper to generate a step dataclass from self trackers.'''

        # poplulate step results container
        step = core.TrainingSessionStep(
            # id/loc
            phase_name=self.phase.name,
            phase_index=0,
            phase_max_epoch=self.phase.num_epochs,
            epoch_in_phase=self._current_epoch,
            global_epoch=self._current_epoch,
            # control
            is_phase_end=self._is_phase_end,
            is_run_end=self._is_phase_end, # single phase
            stop_reason=reason,
            # metrics
            objective_name=self._current_metrics.target_objective,
            objective_value=self._current_metrics.target_metrics,
            best_value_so_far=self._best_value_so_far,
            best_epoch_so_far=self._best_epoch_so_far,
            is_best_epoch=self._is_best_epoch,
            metrics=self._current_metrics,
        )
        # when this method is called it means this training step is done
        self.dispatcher.on_train_step_end(step)
        return step
