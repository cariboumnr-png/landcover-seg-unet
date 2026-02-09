'''Module internal: trainer runtime state.'''

# standard imports
from __future__ import annotations
import dataclasses
# local imports
import alias
import utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class RuntimeConfig:
    '''Minimal runtime config'''
    data: _DataConfig
    schedule: _Schedule
    monitor: _Monitor
    precision: _Precision
    optim: _OptimConfig

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _DataConfig:
    dom_ids_name: str | None
    dom_vec_name: str | None

@dataclasses.dataclass
class _Schedule:
    '''Progression and scheduling'''
    max_epoch: int                  # total epochs target
    max_step: int | None            # optional hard cap on steps (global)
    logging_interval: int         # log every N steps (or batches)
    eval_interval: int | None      # evaluate every N steps (optional)
    checkpoint_interval: int | None # checkpoint every N steps (optional)
    patience_epochs: int | None     # for early stopping
    min_delta: float | None      # minimum improvement threshold

@dataclasses.dataclass
class _Monitor:
    enabled: tuple[str, ...] # e.g., ('iou', ...)
    metric: str                    # e.g., 'iou'
    head: str                    # e.g., 'layer1' (the parent layer)
    mode: str                     # e.g., 'max'

@dataclasses.dataclass
class _Precision:
    '''Compute precision.'''
    use_amp: bool

@dataclasses.dataclass
class _OptimConfig:
    '''Optimization related.'''
    grad_clip_norm: float | None

# -------------------------------Public Function-------------------------------
def get_config(config: alias.ConfigType):
    '''Factory function.'''

    # config accessor
    cfg = utils.ConfigAccess(config)

    return RuntimeConfig(
        data=_DataConfig(
            cfg.get_option('data', 'domain_ids_name'),
            cfg.get_option('data', 'domain_vec_name')
        ),
        schedule=_Schedule(
            cfg.get_option('schedule', 'max_epoch'),
            cfg.get_option('schedule', 'max_step'),
            cfg.get_option('schedule', 'log_every'),
            cfg.get_option('schedule', 'val_every'),
            cfg.get_option('schedule', 'ckpt_every'),
            cfg.get_option('schedule', 'patience'),
            cfg.get_option('schedule', 'min_delta'),
        ),
        monitor=_Monitor(
            ('iou',),
            cfg.get_option('monitor', 'metric_name'),
            cfg.get_option('monitor', 'track_head_name'),
            cfg.get_option('monitor', 'track_mode'),
        ),
        precision=_Precision(
            cfg.get_option('precision', 'use_amp')
        ),
        optim=_OptimConfig(
            cfg.get_option('grad_clip_norm')
        )
    )
