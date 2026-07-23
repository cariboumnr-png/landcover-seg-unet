# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      (c) King's Printer for Ontario, 2026.                  #
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

# pylint: disable=protected-access

'''Unit tests for optim module (optimization.py, builder.py, __init__.py).'''

# standard imports
import dataclasses
# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.runtime.optim as optim


# ----- `Optimization` runtime wrapper tests
def test_optimization_init_defaults(dummy_optimizer):
    '''
    Given: PyTorch optimizer instance.
    When: Instantiating `Optimization` with default parameters.
    Then: Scheduler is None and tracking metadata defaults to None.
    '''
    opt_wrap = optim.Optimization(dummy_optimizer)

    assert opt_wrap.optimizer is dummy_optimizer
    assert opt_wrap.scheduler is None
    assert opt_wrap.grad_clip_norm is None
    assert opt_wrap._sched_cls is None
    assert opt_wrap._sched_factory is None
    assert opt_wrap._sched_args is None


def test_optimization_lrs_property(dummy_optimizer):
    '''
    Given: `Optimization` wrapping an optimizer with parameter groups.
    When: Accessing `lrs` property.
    Then: Return list of learning rates for all parameter groups.
    '''
    opt_wrap = optim.Optimization(dummy_optimizer)

    assert opt_wrap.lrs == [1e-3]


def test_optimization_step_optimizer(dummy_model, dummy_optimizer):
    '''
    Given: `Optimization` wrapper around an optimizer with gradients.
    When: Calling `step_optimizer`.
    Then: Delegate execution to underlying optimizer.
    '''
    opt_wrap = optim.Optimization(dummy_optimizer)
    out = dummy_model.linear(torch.tensor([[1.0, 1.0]]))
    loss = out.sum()
    loss.backward()

    opt_wrap.step_optimizer()


def test_optimization_zero_grad(dummy_model, dummy_optimizer):
    '''
    Given: `Optimization` wrapper after backward pass.
    When: Calling `zero_grad(set_to_none=True)`.
    Then: Parameter gradients are reset to None.
    '''
    opt_wrap = optim.Optimization(dummy_optimizer)
    out = dummy_model.linear(torch.tensor([[1.0, 1.0]]))
    loss = out.sum()
    loss.backward()

    assert dummy_model.linear.weight.grad is not None
    opt_wrap.zero_grad(set_to_none=True)
    assert dummy_model.linear.weight.grad is None


def test_optimization_step_scheduler_none(dummy_optimizer):
    '''
    Given: `Optimization` without a scheduler.
    When: Calling `step_scheduler`.
    Then: No operation occurs and no exception is raised.
    '''
    opt_wrap = optim.Optimization(dummy_optimizer, scheduler=None)

    opt_wrap.step_scheduler()


def test_optimization_step_scheduler_active(dummy_optimizer):
    '''
    Given: `Optimization` with active `CosineAnnealingLR` scheduler.
    When: Calling `step_scheduler`.
    Then: Advance scheduler state and update learning rate.
    '''
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(dummy_optimizer, T_max=10)
    opt_wrap = optim.Optimization(dummy_optimizer, scheduler=sched)

    initial_lr = opt_wrap.lrs[0]
    opt_wrap.step_scheduler()
    assert opt_wrap.lrs[0] != initial_lr


# ----- `Optimization.reconfigure` tests
def test_reconfigure_lr_only(dummy_optimizer):
    '''
    Given: `Optimization` wrapper.
    When: Calling `reconfigure` with new learning rate.
    Then: Learning rate is updated across all parameter groups.
    '''
    opt_wrap = optim.Optimization(dummy_optimizer)
    opt_wrap.reconfigure(lr=5e-4)

    assert opt_wrap.lrs == [5e-4]


def test_reconfigure_disable_scheduler(dummy_optimizer):
    '''
    Given: `Optimization` wrapper with active scheduler.
    When: Calling `reconfigure(disable_scheduler=True)`.
    Then: Scheduler instance and stored configuration metadata are cleared.
    '''
    sched_factory = torch.optim.lr_scheduler.CosineAnnealingLR
    sched = sched_factory(dummy_optimizer, T_max=10)
    opt_wrap = optim.Optimization(
        dummy_optimizer,
        scheduler=sched,
        sched_cls='CosAnneal',
        sched_factory=sched_factory,
        sched_args={'T_max': 10}
    )

    opt_wrap.reconfigure(disable_scheduler=True)

    assert opt_wrap.scheduler is None
    assert opt_wrap._sched_factory is None
    assert opt_wrap._sched_cls is None
    assert opt_wrap._sched_args is None


def test_reconfigure_no_op(dummy_optimizer):
    '''
    Given: `Optimization` wrapper without scheduler arguments.
    When: Calling `reconfigure` with no parameters.
    Then: No state changes occur.
    '''
    opt_wrap = optim.Optimization(dummy_optimizer)
    opt_wrap.reconfigure()

    assert opt_wrap.scheduler is None


def test_reconfigure_incomplete_config_raises(dummy_optimizer):
    '''
    Given: `Optimization` wrapper without stored scheduler factory.
    When: Calling `reconfigure` with new `sched_args`.
    Then: Raise `ValueError`.
    '''
    opt_wrap = optim.Optimization(dummy_optimizer)

    with pytest.raises(ValueError, match='Scheduler configuration incomplete'):
        opt_wrap.reconfigure(sched_args={'T_max': 20})


def test_reconfigure_scheduler_rebuild(dummy_optimizer):
    '''
    Given: `Optimization` wrapper initialized with scheduler metadata.
    When: Calling `reconfigure` with updated `sched_args`.
    Then: Rebuild scheduler instance with merged arguments.
    '''
    sched_factory = torch.optim.lr_scheduler.CosineAnnealingLR
    sched = sched_factory(dummy_optimizer, T_max=10)
    opt_wrap = optim.Optimization(
        dummy_optimizer,
        scheduler=sched,
        sched_cls='CosAnneal',
        sched_factory=sched_factory,
        sched_args={'T_max': 10}
    )

    opt_wrap.reconfigure(sched_args={'T_max': 20})

    assert opt_wrap.scheduler is not None
    assert opt_wrap._sched_args == {'T_max': 20}


# ----- `build_optimization` factory tests
def test_build_optimization_from_session_config(
    dummy_model,
    session_config
):
    '''
    Given: Default `SessionConfig` from `session_config` fixture.
    When: Calling `build_optimization` with `session_config.engine_optim`.
    Then: Successfully construct `Optimization` wrapper.
    '''
    opt_wrap = optim.build_optimization(
        dummy_model,
        session_config.engine_optim
    )

    assert isinstance(opt_wrap.optimizer, torch.optim.AdamW)
    assert opt_wrap.scheduler is not None


def test_build_optimization_adamw_no_scheduler(
    dummy_model,
    session_config
):
    '''
    Given: `_OptimConfig` specifying AdamW with no scheduler.
    When: Calling `build_optimization`.
    Then: Construct `Optimization` with AdamW and specified `grad_clip_norm`.
    '''
    config = dataclasses.replace(
        session_config.engine_optim,
        opt_cls='AdamW',
        sched_cls=None,
        sched_args={},
        grad_clip_norm=1.0
    )

    opt_wrap = optim.build_optimization(dummy_model, config)

    assert isinstance(opt_wrap.optimizer, torch.optim.AdamW)
    assert opt_wrap.scheduler is None
    assert opt_wrap.grad_clip_norm == 1.0


def test_build_optimization_sgd_with_cosanneal_scheduler(
    dummy_model,
    session_config
):
    '''
    Given: `_OptimConfig` specifying SGD and `CosAnneal` scheduler.
    When: Calling `build_optimization`.
    Then: Construct `Optimization` with SGD and `CosineAnnealingLR`.
    '''
    config = dataclasses.replace(
        session_config.engine_optim,
        opt_cls='SGD',
        lr=1e-2,
        weight_decay=0.0,
        sched_cls='CosAnneal',
        sched_args={'T_max': 10},
        grad_clip_norm=0.5
    )

    opt_wrap = optim.build_optimization(dummy_model, config)

    assert isinstance(opt_wrap.optimizer, torch.optim.SGD)
    assert isinstance(
        opt_wrap.scheduler,
        torch.optim.lr_scheduler.CosineAnnealingLR
    )
    assert opt_wrap.grad_clip_norm == 0.5
    assert opt_wrap._sched_cls == 'CosAnneal'


def test_build_optimization_onecycle_scheduler(
    dummy_model,
    session_config
):
    '''
    Given: `_OptimConfig` specifying `OneCycle` scheduler.
    When: Calling `build_optimization`.
    Then: Construct `Optimization` with `OneCycleLR`.
    '''
    config = dataclasses.replace(
        session_config.engine_optim,
        opt_cls='AdamW',
        sched_cls='OneCycle',
        sched_args={'max_lr': 1e-2, 'total_steps': 100}
    )

    opt_wrap = optim.build_optimization(dummy_model, config)

    assert isinstance(
        opt_wrap.scheduler,
        torch.optim.lr_scheduler.OneCycleLR
    )


def test_build_optimization_unknown_optimizer_raises(
    dummy_model,
    session_config
):
    '''
    Given: `_OptimConfig` with unregistered `opt_cls`.
    When: Calling `build_optimization`.
    Then: Raise `ValueError`.
    '''
    config = dataclasses.replace(
        session_config.engine_optim,
        opt_cls='InvalidOptimizer'
    )

    with pytest.raises(ValueError, match='Unknown optimizer: InvalidOptimizer'):
        optim.build_optimization(dummy_model, config)


def test_build_optimization_unknown_scheduler_raises(
    dummy_model,
    session_config
):
    '''
    Given: `_OptimConfig` with unregistered `sched_cls`.
    When: Calling `build_optimization`.
    Then: Raise `ValueError`.
    '''
    config = dataclasses.replace(
        session_config.engine_optim,
        sched_cls='InvalidScheduler'
    )

    with pytest.raises(ValueError, match='Unknown scheduler: InvalidScheduler'):
        optim.build_optimization(dummy_model, config)
