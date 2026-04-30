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

'''Compose a list of callback classes in sequence for the engines.'''

# standard imports
import dataclasses
# local imports
import landseg.session.instrumentation.callbacks as callbacks

@dataclasses.dataclass
class _CallbackSet:
    '''Collection of callback class contracts.'''
    train: callbacks.TrainCallback
    validate: callbacks.ValCallback
    infer: callbacks.InferCallback
    progress: callbacks.ProgressCallback

    def __iter__(self):
        return iter((getattr(self, f.name) for f in dataclasses.fields(self)))

def build_callbacks(
    state: callbacks.EngineStateLike,
    *,
    device: str,
    verbose: bool = True
) -> _CallbackSet:
    '''Factory to generate a set of callback class instances.'''

    # return
    return _CallbackSet(
        train=callbacks.TrainCallback(state, device=device, verbose=verbose),
        validate=callbacks.ValCallback(state, device=device, verbose=verbose),
        infer=callbacks.InferCallback(state, device=device, verbose=verbose),
        progress=callbacks.ProgressCallback(state, device=device, verbose=verbose),
    )
