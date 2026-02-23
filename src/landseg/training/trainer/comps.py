'''Collection of external protocols for trainer components.'''

# standard imports
import dataclasses
# local imports
import landseg.training.common as common

# ---------------------------trainer runtime config---------------------------
@dataclasses.dataclass
class TrainerComponents:
    '''Trainer components protocol.'''
    model: common.MultiheadModelLike
    dataloaders: common.DataLoadersLike
    headspecs: common.HeadSpecsLike
    headlosses: common.HeadLossesLike
    headmetrics: common.HeadMetricsLike
    optimization: common.OptimizationLike
    callbacks: common.CallBacksLike
