# pylint: disable=missing-function-docstring, too-few-public-methods
'''Simple optimizer factory.'''

# standard imports
import dataclasses
import typing
# third-party imports
import torch
# local imports
import landseg.alias as alias
import landseg.utils as utils

# ---------------------------------Public Type---------------------------------
@typing.runtime_checkable
class ModelWithParams(typing.Protocol):
    '''Minimal protocol for a model that contains torch parameters.'''
    def parameters(self) -> typing.Iterable['torch.nn.Parameter']: ...

# a callable that constructs an Optimizer (e.g., torch.optim.AdamW)
p1 = typing.ParamSpec('p1')
OptimizerFactory: typing.TypeAlias = typing.Callable[p1, "torch.optim.Optimizer"]

# a callable that constructs a Scheduler (e.g., CosineAnnealingLR)
p2 = typing.ParamSpec('p2')
SchedulerFactory: typing.TypeAlias = typing.Callable[p2, "torch.optim.lr_scheduler.LRScheduler"]

_OPTIMIZERS: dict[str, OptimizerFactory] = {
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}
_SCHEDULERS: dict[str, SchedulerFactory] = {
    "CosAnneal": torch.optim.lr_scheduler.CosineAnnealingLR,
    "OneCycle": torch.optim.lr_scheduler.OneCycleLR,
}

# -------------------------------Public Function-------------------------------
@dataclasses.dataclass
class Optimization:
    '''Wrapper for optimization components'''
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None

# -------------------------------Public Function-------------------------------
def build_optimization(
        model: ModelWithParams,
        config: alias.ConfigType
    ) -> Optimization:
    '''Factory-like wrapper function.'''

    # config accessor
    cfg = utils.ConfigAccess(config)

    optimizer = _build_optimizer(
        model=model,
        optim_cls=cfg.get_option('opt_cls'),
        lr=cfg.get_option('lr'),
        weight_decay=cfg.get_option('weight_decay')
    )
    scheduler = _build_scheduler(
        optimizer,
        sched_cls=cfg.get_option('sched_cls'),
        sched_args=cfg.get_option('sched_args')
    )
    return Optimization(optimizer=optimizer, scheduler=scheduler)

# ------------------------------private  function------------------------------
def _build_optimizer(
        model: ModelWithParams,
        optim_cls: str,
        lr: float,
        weight_decay: float
    ) -> torch.optim.Optimizer:
    '''
    Docstring for build_optimizer

    :param model: Description
    :param cfg: Description
    '''

    optimizer_class = _OPTIMIZERS.get(optim_cls)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer: {optim_cls}")
    return optimizer_class(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        sched_cls: str,
        sched_args: dict
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
    '''
    Docstring for build_scheduler

    :param optimizer: Description
    :param cfg: Description
    '''
    if sched_cls is None:
        return None
    scheduler = _SCHEDULERS.get(sched_cls)
    if scheduler is None:
        raise ValueError(f"Unknown scheduler: {sched_cls}")
    return scheduler(optimizer=optimizer, **sched_args)
