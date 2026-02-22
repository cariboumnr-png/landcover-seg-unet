'''Common protocols'''

from .dataset import DataSpecsLike
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
    InferCallbackLike,
    LoggingCallbackLike
)
from .trainer_config import (
    RuntimeConfigLike,
    ScheduleLike,
    MonitorLike,
    PrecisionLike,
    OptimConfigLike
)
from .trainer_state import RuntimeStateLike

__all__ = [
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
    'ScheduleLike',
    'MonitorLike',
    'PrecisionLike',
    'OptimConfigLike',
    #
    'RuntimeStateLike',
    #
    'DataSpecsLike',
    #
    'CallBacksLike',
    'ProgressCallbackLike',
    'TrainCallbackLike',
    'ValCallbackLike',
    'InferCallbackLike',
    'LoggingCallbackLike'
]
