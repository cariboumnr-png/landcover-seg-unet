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
Shared engine policy base class.

This module defines an abstract policy layer shared by concrete session
engines (e.g., trainer and evaluator). It provides common orchestration,
state interpretation, and callback wiring while delegating all batch-level
execution mechanics to a shared execution core.
'''

# standard imports
import copy
# local imports
import landseg.session.common as common
import landseg.session.engine.runtime as runtime
import landseg.session.engine.protocols as protocols

class EngineBase:
    '''
    Base class for session engines defining policy on top of execution.

    An engine encapsulates *policy* and *orchestration behavior* over a
    shared batch execution runtime. Concrete implementations (e.g.,
    trainer and evaluator engines) differ in how they interpret runtime
    state and coordinate execution over time, while sharing:

    - runtime state access and interpretation
    - model/head configuration
    - observer (callback) dispatching
    - device placement

    This class intentionally excludes:
    - batch execution mechanics (handled by `runtime`)
    - optimization logic (e.g., optimizer/scheduler)
    - epoch and phase control flow
    '''

    def __init__(
        self,
        engine_runtime: runtime.EngineRuntime,
        dataloaders: protocols.DataLoadersLike,
        dispatcher: common.SessionObserverLike,
        *,
        device: str,
    ):
        '''
        Initialization with shared runtime and orchestration bindings.

        The engine operates over an external execution runtime and does
        not own mutable training state. Instead, it interprets runtime
        outputs according to its policy (e.g., training, validation,
        inference) and coordinates high-level execution behavior.

        Args:
            engine_runtime:
                Shared execution runtime responsible for batch-level
                computation and state mutation.
            dataloaders:
                Data access interface providing train/val/test loaders
                and associated metadata.
            dispatcher:
                Observer interface for emitting lifecycle events during
                execution.
            device:
                Target device for model placement (e.g., 'cpu', 'cuda').

        Notes:
            - The model is moved to the target device during
              initialization.
            - The engine assumes all components are preconfigured and
              focuses purely on orchestration logic.
        '''

        # execution core
        self.runtime = engine_runtime
        # data loader
        self.dataloaders = dataloaders
        # callback system
        self.dispatcher = dispatcher
        # device placement
        self.device = device
        self.model.to(self.device)

    # ----- access attributes as properties
    @property
    def engine(self):
        '''Return the engine'''
        return self.runtime.engine

    @property
    def model(self):
        '''Return the model'''
        return self.runtime.engine.model

    @property
    def state(self):
        '''Return the engine state.'''
        return self.runtime.engine.state

    @property
    def headspecs(self):
        '''Access head specifications.'''
        return self.runtime.engine_tasks.headspecs

    @property
    def headlosses(self):
        '''Access head-specific loss modules.'''
        return self.runtime.engine_tasks.headlosses

    @property
    def headmetrics(self):
        '''Access head-specific metric modules.'''
        return self.runtime.engine_tasks.headmetrics

    @property
    def optimization(self):
        '''Optim'''
        return self.runtime.engine_optim

    # ----- head configuration helpers
    def set_head_state(
        self,
        active_heads: list[str] | None = None,
        frozen_heads: list[str] | None = None,
    ) -> None:
        '''
        Configure active and frozen heads and apply class exclusions.

        This method updates both the model and runtime state to reflect
        the current head configuration. Per-head specifications, loss
        modules, and metrics are deep-copied into runtime state to
        ensure isolation across phases.

        Side effects:
            - Updates model active and frozen heads.
            - Installs active head specs, losses, and metrics into
              RuntimeState.
            - Applies per-head class exclusions where specified.

        Args:
            active_heads:
                Heads to activate. Defaults to all available heads when
                set to None.
            frozen_heads:
                Heads to freeze at the model level.
            excluded_cls:
                Mapping of head name to class indices to exclude from
                loss computation and validation metrics.
        '''

        # if no active heads provided, make all heads active
        if active_heads is None:
            active_heads = self.state.heads.all_heads

        # update runtime state
        self.state.heads.active_heads = active_heads
        self.state.heads.frozen_heads = frozen_heads

        # configure model
        self.model.set_active_heads(active_heads)

        # deep-copy per-head components into state
        self.state.heads.active_hspecs = {
            h: copy.deepcopy(self.headspecs[h]) for h in active_heads
        }
        # set loss module for active heads
        self.state.heads.active_hloss = {
            h: copy.deepcopy(self.headlosses[h]) for h in active_heads
        }
        # set metric module for active heads
        self.state.heads.active_hmetrics = {
            h: copy.deepcopy(self.headmetrics[h]) for h in active_heads
        }

        # set frozen heads to model if provided
        if frozen_heads is not None:
            self.model.set_frozen_heads(frozen_heads)

    def reset_head_state(self) -> None:
        '''
        Reset all runtime head configuration.

        This restores the model and runtime state to an unconfigured
        head state (no active or frozen heads, no per-head overrides).
        '''

        self.model.reset_heads()
        self.state.heads.active_heads = None
        self.state.heads.frozen_heads = None
        self.state.heads.active_hspecs = None
        self.state.heads.active_hloss = None
        self.state.heads.active_hmetrics = None

    # ----- batch context/output reset
    def _batch_reset(self, bidx: int, _batch: tuple) -> None:
        '''Refresh batch context and output from engine state.'''
        # refresh batch ctx
        self.state.batch_cxt.refresh(bidx, _batch)
        # refresh batch results
        self.state.batch_out.refresh(bidx)
