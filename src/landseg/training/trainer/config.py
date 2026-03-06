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
Internal runtime configuration structures for the trainer.

Defines typed dataclasses for scheduling, monitoring, precision, and
optimization settings, along with a factory that constructs a unified
RuntimeConfig from user configuration files.
'''

# standard imports
from __future__ import annotations
import dataclasses
# local imports
import landseg.alias as alias
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class RuntimeConfig:
    '''Container holding all trainer runtime configuration sections.'''
    schedule: _Schedule
    monitor: _Monitor
    precision: _Precision
    optim: _OptimConfig

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _Schedule:
    '''Training schedule related: epochs/logging/eval/ckpt/patience.'''
    max_epoch: int                  # total epochs target
    max_step: int | None            # optional hard cap on steps (global)
    logging_interval: int           # log every N steps (or batches)
    eval_interval: int | None       # evaluate every N steps (optional)
    checkpoint_interval: int | None # checkpoint every N steps (optional)
    patience_epochs: int | None     # for early stopping
    min_delta: float | None         # minimum improvement threshold

@dataclasses.dataclass
class _Monitor:
    '''Metric tracking configuration for validation/early stopping.'''
    enabled: tuple[str, ...]        # e.g., ('iou', ...)
    metric: str                     # e.g., 'iou'
    head: str                       # e.g., 'layer1' (the parent layer)
    mode: str                       # e.g., 'max'

@dataclasses.dataclass
class _Precision:
    '''AMP/numerical precision settings.'''
    use_amp: bool

@dataclasses.dataclass
class _OptimConfig:
    '''Optimization-related runtime settings.'''
    grad_clip_norm: float | None

# -------------------------------Public Function-------------------------------
def get_config(config: alias.ConfigType):
    '''
    Build a RuntimeConfig from user configuration.

    Args:
        config: Configuration object providing schedule, monitor,
            precision, and optimization fields via nested keys.

    Returns:
        RuntimeConfig: fully populated runtime settings for the trainer.
    '''

    # config accessor
    cfg = utils.ConfigAccess(config)

    return RuntimeConfig(
        schedule=_Schedule(
            max_epoch=cfg.get_option('schedule', 'max_epoch'),
            max_step=cfg.get_option('schedule', 'max_step'),
            logging_interval=cfg.get_option('schedule', 'log_every'),
            eval_interval=cfg.get_option('schedule', 'val_every'),
            checkpoint_interval=cfg.get_option('schedule', 'ckpt_every'),
            patience_epochs=cfg.get_option('schedule', 'patience'),
            min_delta=cfg.get_option('schedule', 'min_delta'),
        ),
        monitor=_Monitor(
            enabled=('iou',),
            metric=cfg.get_option('monitor', 'metric_name'),
            head=cfg.get_option('monitor', 'track_head_name'),
            mode=cfg.get_option('monitor', 'track_mode'),
        ),
        precision=_Precision(
            use_amp=cfg.get_option('precision', 'use_amp')
        ),
        optim=_OptimConfig(
            grad_clip_norm=cfg.get_option('grad_clip_norm')
        )
    )
