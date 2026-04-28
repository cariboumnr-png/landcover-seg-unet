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
import landseg.session.common as common
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
        training_phases: typing.Sequence[phases.PhaseLike],
        **kwargs
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

    def run(self) -> typing.Generator[runner.TrainingStep, None, None]:
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

        # tracking
        current_phase_idx: int = 0
        current_epoch: int = 1
        current_metrics: common.EpochMetricsLike | None = None

        # iterate through provided phases
        for i, phase in enumerate(self.phases):

            # print phase info if verbose
            if self.config.verbose:
                self._print_phase(phase)

            # get phase events stream
            events_stream = policy.PhasePolicy(
                epoch_runner=self.epoch_runner,
                phase_config=phase,
                track_config=self.config.tracking,
            ).run()

            # manually advance the generator to capture both yields and returns
            while True:

                try:
                    e = next(events_stream)
                except StopIteration:
                    break # already at the end of phase

                # runner intercepts events to perform side effects
                match e:

                    case events.EpochEnd(epoch_index=epoch, metrics=metrics):
                        # typing correctness
                        assert isinstance(metrics, common.EpochMetricsLike)
                        # tracking
                        current_phase_idx = i
                        current_epoch = epoch
                        current_metrics = metrics
                        is_phase_end = epoch==phase.num_epochs
                        self._log_metrics(epoch, phase.num_epochs, metrics)
                        yield runner.TrainingStep(
                            phase_name=phase.name,
                            phase_index=i,
                            epoch=epoch,
                            metrics=metrics,
                            is_phase_end=is_phase_end
                        )
                        # exit current phase
                        if is_phase_end:
                            break

                    case events.CheckpointRequest(tag=tag):
                        self._save_progress(phase.name, is_best=tag=='best')

        # simple check that we are actually at the end
        assert current_phase_idx == len(self.phases) - 1
        assert current_epoch == self.phases[current_phase_idx].num_epochs
        # final terminal step after all phases complete
        yield runner.TrainingStep(
            phase_name=self.phases[current_phase_idx].name,
            phase_index=current_phase_idx,
            epoch=current_epoch,
            metrics=current_metrics,
            is_phase_end=True,
            is_run_end=True,
        )
