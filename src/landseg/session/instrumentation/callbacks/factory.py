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

'''Compose a list of callback classes in sequence for the trainer.'''

# standard imports
import dataclasses
# local imports
import landseg.session.common as common
import landseg.session.instrumentation.callbacks as callbacks
import landseg.utils as utils

@dataclasses.dataclass
class _CallbackSet:
    '''Collection of callback class contracts.'''
    train: callbacks.TrainCallback
    validate: callbacks.ValCallback
    infer: callbacks.InferCallback
    logging: callbacks.LoggingCallback
    progress: callbacks.ProgressCallback

    def __iter__(self):
        return iter((getattr(self, f.name) for f in dataclasses.fields(self)))

def build_callbacks(
    state: common.StateLike,
    config: common.ConfigLike,
    logger: utils.Logger,
    *,
    device: str,
    skip_log: bool = False
) -> _CallbackSet:
    '''Factory to generate a set of callback class instances.'''

    # build set
    callback_set = _CallbackSet(
        train=callbacks.TrainCallback(logger),
        validate=callbacks.ValCallback(logger),
        infer=callbacks.InferCallback(logger),
        logging=callbacks.LoggingCallback(logger),
        progress=callbacks.ProgressCallback(logger),
    )
    # set up all callback instances
    callback_set.train.setup(state, config, device=device, skip_log=skip_log)
    callback_set.validate.setup(state, config, device=device, skip_log=skip_log)
    callback_set.infer.setup(state, config, device=device, skip_log=skip_log)
    callback_set.logging.setup(state, config, device=device, skip_log=skip_log)
    callback_set.progress.setup(state, config, device=device, skip_log=skip_log)
    # return
    return callback_set
