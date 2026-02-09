'''Optimizer factory.'''

# standard imports
import dataclasses
# third-party imports
import torch
# local imports
import alias
import training.optim
import utils

@dataclasses.dataclass
class Optimization:
    '''Wrapper for optimization components'''
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None

@dataclasses.dataclass
class _Config:
    '''doc'''
    # optimizer config
    opt_cls: str = 'AdamW' # default
    lr: float = 1e-4
    weight_decay: float = 1e-3
    # optional scheduler config
    sched_cls: str | None = 'CosAnneal' # default
    sched_args: dict = dataclasses.field(default_factory=dict)

# add more if needed
_OPTIMIZERS: dict[str, training.optim.OptimizerFactory] = {
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}
_SCHEDULERS: dict[str, training.optim.SchedulerFactory] = {
    "CosAnneal": torch.optim.lr_scheduler.CosineAnnealingLR,
    "OneCycle": torch.optim.lr_scheduler.OneCycleLR,
}

# public functions
def build_optimizer(
        model: training.optim.ModelWithParams,
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

def build_scheduler(
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

def build_optimization(
        model: training.optim.ModelWithParams,
        config: alias.ConfigType
    ) -> Optimization:
    '''Factory-like wrapper function.'''

    # config accessor
    cfg = utils.ConfigAccess(config)

    optimizer = build_optimizer(
        model=model,
        optim_cls=cfg.get_option('opt_cls'),
        lr=cfg.get_option('lr'),
        weight_decay=cfg.get_option('weight_decay')
    )
    scheduler = build_scheduler(
        optimizer,
        sched_cls=cfg.get_option('sched_cls'),
        sched_args=cfg.get_option('sched_args')
    )
    return Optimization(optimizer=optimizer, scheduler=scheduler)
