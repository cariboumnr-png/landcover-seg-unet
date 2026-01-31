'''Common protocols'''

from .data import DataSummaryLike

from .checkpoint import CheckpointMetaLike
from .trainer import (
    TrainerLike
)
from .trainer_comps import (
    TrainerComponentsLike,
    MultiheadModelLike,
    DataLoadersLike,
    HeadSpecsLike,
    SpecLike,
    HeadLossesLike,
    CompositeLossLike,
    HeadMetricsLike,
    MetricLike,
    OptimizationLike,
    CallBacksLike,
    ProgressCallbackLike,
    TrainCallbackLike,
    ValCallbackLike,
    LoggingCallbackLike
)
from .trainer_config import (
    RuntimeConfigLike,
    DataConfigLike,
    ScheduleLike,
    MonitorLike,
    PrecisionLike,
    OptimConfigLike
)
from .trainer_state import RuntimeStateLike

__all__ = [
    #
    'CheckpointMetaLike',
    #
    'TrainerLike',
    #
    'TrainerComponentsLike',
    'MultiheadModelLike',
    'DataLoadersLike',
    'HeadSpecsLike',
    'SpecLike',
    'HeadLossesLike',
    'CompositeLossLike',
    'HeadMetricsLike',
    'MetricLike',
    'OptimizationLike',
    #
    'RuntimeConfigLike',
    'DataConfigLike',
    'ScheduleLike',
    'MonitorLike',
    'PrecisionLike',
    'OptimConfigLike',
    #
    'RuntimeStateLike',
    #
    'DataSummaryLike',
    #
    'CallBacksLike',
    'ProgressCallbackLike',
    'TrainCallbackLike',
    'ValCallbackLike',
    'LoggingCallbackLike'
]
