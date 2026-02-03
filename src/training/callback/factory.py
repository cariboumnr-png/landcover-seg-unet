'''Compose a list of callback classes.'''

# standard imports
import dataclasses
# local imports
import training.callback
import training.common
import utils

@dataclasses.dataclass
class CallbackSet:
    '''Wrapper for concrete callback classes.'''
    train: training.common.TrainCallbackLike
    validate: training.common.ValCallbackLike
    infer: training.common.InferCallbackLike
    logging: training.common.LoggingCallbackLike
    progress: training.common.ProgressCallbackLike

    def __iter__(self):
        return iter((getattr(self, f.name) for f in dataclasses.fields(self)))

def build_callbacks(logger: utils.Logger) -> CallbackSet:
    '''Public API'''

    return CallbackSet(
        train=training.callback.TrainCallback(logger),
        validate=training.callback.ValCallback(logger),
        infer=training.callback.InferCallback(logger),
        logging=training.callback.LoggingCallback(logger),
        progress=training.callback.ProgressCallback(logger),
    )
