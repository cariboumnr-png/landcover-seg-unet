# pylint: disable=missing-function-docstring
# pylint: disable=too-many-public-methods
# pylint: disable=unused-argument
'''Base class for callbacks.'''

# local imports
import training.common
import utils

class Callback:
    '''Base class for callbacks. Subclass to implement behaviors'''

    def __init__(self, logger: utils.Logger) -> None:
        self._trainer: training.common.TrainerLike | None = None
        self.train_logger = logger.get_child('train')
        self.valdn_logger = logger.get_child('valdn')

    def setup(self, trainer: training.common.TrainerLike) -> None:
        if self._trainer is not None:
            raise RuntimeError("Callback.setup() called twice.")
        self._trainer = trainer

    def log_train(self, level: str, message: str) -> None:
        '''Centralized callback logging'''
        self.train_logger.log(level, message)

    def log_valdn(self, level: str, message: str) -> None:
        '''Centralized callback logging'''
        self.valdn_logger.log(level, message)

    # -----------------------------training phase-----------------------------
    def on_train_epoch_begin(self, epoch: int) -> None: ...
    def on_train_batch_begin(self, bidx: int, batch: tuple) -> None: ...
    def on_train_batch_forward(self) -> None: ...
    def on_train_batch_compute_loss(self) -> None: ...
    def on_train_backward(self) -> None: ...
    def on_train_before_optimizer_step(self) -> None: ...
    def on_train_optimizer_step(self) -> None: ...
    def on_train_batch_end(self) -> None: ...
    def on_train_epoch_end(self) -> None: ...

    # ----------------------------validation phase----------------------------
    def on_validation_begin(self) -> None: ...
    def on_validation_batch_begin(self, bidx: int, batch: tuple) -> None: ...
    def on_validation_batch_forward(self) -> None: ...
    def on_validation_batch_end(self) -> None: ...
    def on_validation_end(self) -> None: ...

    # -------------------------convenience properties-------------------------
    @property
    def trainer(self):
        if self._trainer is None:
            raise RuntimeError('Trainer accessed before setup.')
        return self._trainer

    @property
    def config(self):
        return self.trainer.config

    @property
    def state(self):
        return self.trainer.state
