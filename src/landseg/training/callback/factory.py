'''Compose a list of callback classes.'''

# standard imports
import dataclasses
# local imports
import landseg.training.callback as callback
import landseg.training.common as common
import landseg.utils as utils

@dataclasses.dataclass
class CallbackSet:
    '''Wrapper for concrete callback classes.'''
    train: common.TrainerCallbackLike
    validate: common.ValCallbackLike
    infer: common.InferCallbackLike
    logging: common.LoggingCallbackLike
    progress: common.ProgressCallbackLike

    def __iter__(self):
        return iter((getattr(self, f.name) for f in dataclasses.fields(self)))

def build_callbacks(logger: utils.Logger) -> CallbackSet:
    '''Public API'''

    return CallbackSet(
        train=callback.TrainCallback(logger),
        validate=callback.ValCallback(logger),
        infer=callback.InferCallback(logger),
        logging=callback.LoggingCallback(logger),
        progress=callback.ProgressCallback(logger),
    )
