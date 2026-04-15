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
import landseg.session.engine.core as engine_core

class EngineBase:
    '''
    Base class for session engines.

    An engine defines *policy* and *orchestration* on top of a shared
    batch execution core. Concrete engines (e.g., trainer, evaluator)
    differ in how they interpret execution results over time, but
    share:

    - runtime state interpretation
    - head configuration logic
    - callback wiring
    - device placement

    This class deliberately does **not** implement:
    - batch execution
    - optimizer or scheduler logic
    - epoch or phase control
    '''

    def __init__(
        self,
        engine: engine_core.BatchExecutionEngine,
        state: engine_core.RuntimeState,
        components: common.TrainerComponentsLike,
        config: common.TrainerConfigShape,
        device: str,
        **kwargs
    ):
        '''
        Initialize an engine policy instance.

        The engine is constructed with an already-initialized batch
        execution core and a shared RuntimeState. Runtime state is not
        owned by the engine; it is deterministically mutated by the
        execution core and interpreted according to engine-specific
        policy (training, evaluation, or inference).

        Args:
            engine:
                BatchExecutionEngine responsible for all batch-level
                execution mechanics.
            state:
                Shared RuntimeState instance updated by the execution
                core and consumed by the engine.
            components:
                Engine components including dataloaders, callbacks,
                loss modules, metric modules, and (if applicable)
                optimization components.
            config:
                Runtime configuration controlling scheduling,
                precision, and monitoring behavior.
            device:
                Device identifier (e.g., 'cpu', 'cuda', 'cuda:0') to
                which the model is moved at engine construction.
            kwargs:
                Runtime control flags:
                - skip_log:
                    Disable logging callbacks.
                - enable_train_la:
                    Enable logit adjustment during training.
                - enable_val_la:
                    Enable logit adjustment during validation.
                - enable_test_la:
                    Enable logit adjustment during inference.
        '''

        # execution core and shared state
        self.engine = engine
        self.model = engine.model
        self.state = state

        # engine components and configuration
        self.comps = components
        self.config = config

        # device placement
        self.device = device
        self.model.to(self.device)

        # runtime flags
        self.flags = {
            'skip_log': kwargs.get('skip_log', False),
            'enable_train_la': kwargs.get('enable_train_la', False),
            'enable_val_la': kwargs.get('enable_val_la', False),
            'enable_test_la': kwargs.get('enable_test_la', False),
        }

        # setup callbacks
        for callback in self.callbacks:
            callback.setup(self, self.flags['skip_log'])

    # ----- convenience properties

    @property
    def dataloaders(self):
        '''Access configured dataloaders.'''
        return self.comps.dataloaders

    @property
    def headspecs(self):
        '''Access head specifications.'''
        return self.comps.headspecs

    @property
    def headlosses(self):
        '''Access head-specific loss modules.'''
        return self.comps.headlosses

    @property
    def optimization(self):
        '''Access optimization configuration and components.'''
        return self.comps.optimization

    @property
    def headmetrics(self):
        '''Access head-specific metric modules.'''
        return self.comps.headmetrics

    @property
    def callbacks(self):
        '''Access registered callbacks.'''
        return self.comps.callbacks

    # ----- head configuration helpers
    def set_head_state(
        self,
        active_heads: list[str] | None = None,
        frozen_heads: list[str] | None = None,
        excluded_cls: dict[str, list[int]] | None = None
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

        # set excluded classes to active heads
        if excluded_cls is not None:
            for h in active_heads:
                excl = excluded_cls.get(h)
                if excl is not None:
                    self.state.heads.active_hspecs[h].set_exclude(tuple(excl))
                    self.state.heads.active_hmetrics[h].exclude_class_1b = tuple(excl)

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

    # ----- runtime configuration helpers
    def config_logit_adjustment(
        self,
        *,
        enable_train_logit_adjustment: bool,
        enable_val_logit_adjustment: bool,
        enable_test_logit_adjustment: bool,
        **kwargs
    ) -> None:
        '''
        Configure logit adjustment usage flags.

        This helper exists to centralize runtime flag manipulation and
        provide a stable interface for trainer and evaluator policy.
        '''

        # assign flags
        self.flags['enable_train_la'] = enable_train_logit_adjustment
        self.flags['enable_val_la'] = enable_val_logit_adjustment
        self.flags['enable_test_la'] = enable_test_logit_adjustment
        # implemented for signature flexibility
        if kwargs:
            pass

    # ----- callback dispatch
    def _emit(self, hook: str, *args, **kwargs) -> None:
        '''Emit a lifecycle hook to all registered callbacks.'''

        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if callable(method):
                method(*args, **kwargs)
