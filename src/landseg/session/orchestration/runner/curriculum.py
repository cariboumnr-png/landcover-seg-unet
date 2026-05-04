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
Curriculum-based training runner.

This module defines a concrete BaseRunner implementation that executes
a sequence of training phases (a curriculum) over a shared epoch engine.

Each phase is executed sequentially via its PhasePolicy, and training
progress is exposed exclusively as a stream of TrainingStep records,
one per completed epoch.
'''

# standard imports
import typing
# local imports
import landseg.core as core
import landseg.session.orchestration.events as events
import landseg.session.orchestration.phases as phases
import landseg.session.orchestration.policy as policy
import landseg.session.orchestration.runner as runner

# --------------------------------Public  Class--------------------------------
class CurriculumRunner(runner.BaseRunner):
    '''
    Curriculum-based training runner.

    This runner executes a sequence of training phases in order, using
    a shared epoch execution engine. Each phase defines its own training
    configuration, and phases are executed to completion before the
    next phase begins.

    The runner consumes event streams emitted by PhasePolicy instances
    and translates epoch-completion events into immutable `TrainingStep`
    records.

    Characteristics:
        - Multi-phase (curriculum) execution
        - Generator-based, step-wise progress reporting
        - No resume or load logic at this level
        - No early stopping (phases always complete fully)

    External consumers observe training progress by consuming the
    `TrainingStep` stream and may abandon execution at any point by
    stopping consumption.
    '''

    def __init__(
        self,
        *,
        training_phases: typing.Sequence[phases.PhaseLike],
        **kwargs: typing.Any
    ):
        '''
        Initialize a curriculum-based training runner.

        Args:
            training_phases:
                Ordered sequence of Phase configurations defining the
                curriculum. Phases are executed sequentially and each
                phase runs to completion before the next begins.
            **kwargs:
                Forwarded to BaseRunner initialization, including the
                epoch execution engine, runner configuration, and logger.
        '''

        super().__init__(**kwargs)
        # parse arguments
        self.phases = training_phases
        # tracking
        self._global_epoch: int = 1
        self._is_run_end: bool = False

    def run(self) -> typing.Generator[core.TrainingSessionStep, None, None]:
        '''
        Execute the curriculum as a stream of `TrainingStep` records.

        Yields:
            `TrainingStep`:
                - One `TrainingStep` per completed epoch.
                - Exactly one terminal `TrainingStep` with
                is_run_end == True after all phases complete.

        Behavior:
            - Executes phases sequentially.
            - Consumes internal event streams emitted by phase policies.
            - Translates `EpochEnd` events into `TrainingStep` records.
            - Emits phase boundary information via step metadata.
            - Handles checkpoint persistence and logging as side effects.

        Notes:
            - This runner does not support early stopping.
            - Resume and load behavior must be handled by higher-level
                orchestration (e.g. CLI).
            - External consumers control execution by consuming or
                abandoning the step stream.
        '''

        # iterate through provided phases
        for i, phase in enumerate(self.phases):

            # print phase info if verbose
            if self.config.verbose:
                self._print_phase(phase)

            # get phase events stream
            events_stream = policy.PhasePolicy(
                epoch_runner=self.epoch_runner,
                phase_config=phase,
                track_config=self.tracking,
            ).run()

            # manually advance the generator to capture both yields and returns
            while True:

                try:
                    e = next(events_stream)
                    self._global_epoch += 1
                except StopIteration:
                    break # already at the end of phase

                # runner intercepts events to perform side effects
                match e:

                    case events.EpochEnd(epoch_index=epoch, metrics=metrics):

                        # epoch and run tracking
                        self._current_epoch = epoch
                        self._is_phase_end = epoch==phase.num_epochs
                        self._is_run_end = (
                            self._is_phase_end and
                            i == len(self.phases) - 1
                        )

                    case events.MetricsReport(
                        best_so_far=best_so_far,
                        best_epoch=best_epoch,
                        is_best_epoch=is_best_epoch,
                        raw_metrics=metrics
                    ):

                        # metrics logging
                        self._current_metrics = metrics
                        self._log_metrics(
                            epoch_idx=self._current_epoch,
                            total_epochs=phase.num_epochs,
                            best_so_far=(best_epoch, best_so_far),
                            metrics=metrics
                        )

                        # normal yield
                        yield core.TrainingSessionStep(
                            # id/loc
                            phase_name=phase.name,
                            phase_index=i,
                            epoch=self._current_epoch,
                            global_epoch=self._global_epoch,
                            # control
                            is_phase_end=self._is_phase_end,
                            is_run_end=self._is_run_end,
                            stop_reason=None,
                            # metrics
                            objective_name=self._current_metrics.target_objective,
                            objective_value=self._current_metrics.target_metrics,
                            best_value_so_far=best_so_far,
                            best_epoch_so_far=best_epoch,
                            is_best_epoch=is_best_epoch,
                            metrics=metrics,
                        )

                    case events.CheckpointRequest(tag=tag):
                        self._save_progress(
                            phase.name,
                            self._current_metrics,
                            is_best=tag=='best'
                        )

                # end of current phase
                if self._is_phase_end:
                    break
