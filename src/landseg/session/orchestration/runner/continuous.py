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
import landseg.session.common as common
import landseg.session.orchestration.events as events
import landseg.session.orchestration.phases as phases
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
        phase: phases.PhaseLike,
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

    def run(self) -> typing.Generator[runner.TrainingStep, None, None]:
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

        # tracking
        current_epoch: int = -1
        current_metrics: common.EpochMetricsLike | None = None

        # print phase info if verbose
        if self.config.verbose:
            self._print_phase(self.phase)

        # get phase events stream
        events_stream = policy.PhasePolicy(
            epoch_runner=self.epoch_runner,
            phase_config=self.phase,
            track_config=self.config.tracking,
        ).run()

        # manually advance the generator to capture both yields and returns
        while True:

            try:
                e = next(events_stream)
            except StopIteration:
                return # already at the end of phase

            # runner intercepts events to perform side effects
            match e:

                case events.EpochEnd(epoch_index=epoch, metrics=metrics):
                    # typing correctness
                    assert isinstance(metrics, common.EpochMetricsLike)
                    # tracking
                    current_epoch = epoch
                    current_metrics = metrics
                    is_phase_end = epoch==self.phase.num_epochs
                    self._log_metrics(epoch, self.phase.num_epochs, metrics)

                    yield runner.TrainingStep(
                        phase_name=self.phase.name,
                        phase_index=0,
                        epoch=epoch,
                        metrics=metrics,
                        is_phase_end=is_phase_end
                    )
                    if is_phase_end:
                        reason = 'Max epoch reached'
                        self.logger.log('INFO', f'Exit training: {reason}')
                        return # normal exit

                case events.StopRun(reason=reason):

                    yield runner.TrainingStep(
                        phase_name=self.phase.name,
                        phase_index=0,
                        epoch=current_epoch,
                        metrics=current_metrics,
                        is_run_end=True,
                        early_stop_reason=reason
                    )
                    self.logger.log('INFO', f'Exit training: {reason}')
                    return # exit the whole run

                case events.CheckpointRequest(tag=tag):
                    self._save_progress(self.phase.name, is_best=tag=='best')
