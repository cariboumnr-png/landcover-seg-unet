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
Engine base class
'''

# standard imports
import copy
# local imports
import landseg.session.common as common
import landseg.session.engine.core as engine_core

class EngineBase:
    '''
    Engine base class
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
        Initialize the trainer.

        The trainer is constructed with an already-initialized batch
        execution engine and a shared RuntimeState. Runtime state is not
        owned by the trainer but is interpreted and mutated according
        to training and evaluation policy.

        Args:
            engine:
                BatchExecutionEngine responsible for batch-level execution.
            state:
                Shared RuntimeState instance updated by the batch executor
                and consumed by the trainer.
            components:
                Trainer components including dataloaders, callbacks,
                optimizer, loss modules, and metric modules.
            config:
                Runtime configuration controlling training schedule,
                precision, and monitoring behavior.
            device:
                Device identifier (e.g., 'cpu', 'cuda', 'cuda:0') applied
                at the trainer level.
            kwargs:
                Runtime control flags:
                - skip_log: Disable logging callbacks
                - enable_train_la: Enable logit adjustment during training
                - enable_val_la: Enable logit adjustment during validation
                - enable_test_la: Enable logit adjustment during inference
        '''

        # get attributes from engine
        self.engine = engine
        self.model = engine.model
        self.state = state
        self.comps = components
        # move model to device
        self.device = device
        self.model.to(self.device)
        # get model runtime config
        self.config = config
        # populate runtime flags from kwargs
        self.flags = {
            'skip_log': kwargs.get('skip_log', False),
            'enable_train_la': kwargs.get('enable_train_la', False),
            'enable_val_la': kwargs.get('enable_val_la', False),
            'enable_test_la': kwargs.get('enable_test_la', False),
        }
        # setup callback classes
        for callback in self.callbacks:
            callback.setup(self, self.flags['skip_log'])

    # ----- property
    @property
    def dataloaders(self):
        '''Shortcut to dataloaders.'''
        return self.comps.dataloaders

    @property
    def headspecs(self):
        '''Shortcut to headspecs.'''
        return self.comps.headspecs

    @property
    def headlosses(self):
        '''Shortcut to headlosses.'''
        return self.comps.headlosses

    @property
    def optimization(self):
        '''Shortcut to optimization.'''
        return self.comps.optimization

    @property
    def headmetrics(self):
        '''Shortcut to headmetrics.'''
        return self.comps.headmetrics

    @property
    def callbacks(self):
        '''Shortcut to callbacks.'''
        return self.comps.callbacks

    def set_head_state(
        self,
        active_heads: list[str] | None=None,
        frozen_heads: list[str] | None=None,
        excluded_cls: dict[str, list[int]] | None=None
    ) -> None:
        '''
        Set active/frozen heads and per-head class exclusions.

        Side effects:
            - Updates model active/frozen heads.
            - Deep-copies and installs per-head specs, loss, and metrics
                into `self.state`.
            - Applies per-head class exclusions to specs and metrics.

        Args:
            active_heads: Heads to activate. Defaults to all heads when
                set to `None`.
            frozen_heads: Heads to freeze (if provided).
            excluded_cls: Mapping of head -> tuple of class indices to
                exclude from loss and validation metrics.
        '''

        # if no active heads provided, make all heads active
        if active_heads is None:
            active_heads = self.state.heads.all_heads

        # set active and frozen heads
        self.state.heads.active_heads = active_heads
        self.state.heads.frozen_heads = frozen_heads

        # set active heads at model
        self.model.set_active_heads(active_heads)
        # set active heads specs
        self.state.heads.active_hspecs = {
            k: copy.deepcopy(self.headspecs[k]) for k in active_heads
        }
        # set loss module for active heads
        self.state.heads.active_hloss = {
            k: copy.deepcopy(self.headlosses[k]) for k in active_heads
        }
        # set metric module for active heads
        self.state.heads.active_hmetrics = {
            k: copy.deepcopy(self.headmetrics[k]) for k in active_heads
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

    def reset_head_state(self):
        '''
        Reset runtime training heads.

        Side effects:
        - Calls `model.reset_heads()`.
        - Clears active/frozen heads and related per-head modules.
        '''

        self.model.reset_heads()
        self.state.heads.active_heads = None
        self.state.heads.frozen_heads = None
        self.state.heads.active_hspecs = None
        self.state.heads.active_hloss = None
        self.state.heads.active_hmetrics = None

    def config_logit_adjustment(
        self,
        *,
        enable_train_logit_adjustment: bool,
        enable_val_logit_adjustment: bool,
        enable_test_logit_adjustment: bool,
        **kwargs
    ) -> None:
        '''
        Simple helper to set logit adjustment use flags.
        '''

        # assign flags
        self.flags['enable_train_la'] = enable_train_logit_adjustment
        self.flags['enable_val_la'] = enable_val_logit_adjustment
        self.flags['enable_test_la'] = enable_test_logit_adjustment
        # implemented for signature flexibility
        if kwargs:
            pass

    def _emit(self, hook: str, *args, **kwargs) -> None:
        '''
        Invoke a named hook from callbacks with the provided arguments.

        Args:
            hook: Hook method to call (e.g., 'on_train_batch_begin').
            *args: Positional arguments passed to the callback method.
            **kwargs: Keyword arguments passed to the callback method.
        '''

        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if callable(method):
                method(*args, **kwargs)
