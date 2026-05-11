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
Build dispatcher
'''

# standard imports
import typing
# local imports
import landseg.session.instrumentation.callbacks as callbacks
import landseg.session.instrumentation.tracking as tracking

def build_dispatcher(
    trackers: list[typing.Literal['tb', 'mlflow']] | None = None,
    uri: str | None = None,
    # artifact_path: str | None = None, used by MLFlow
    verbose: bool = True
) -> callbacks.CallbackDispatcher:
    '''doc'''

    # trackers list
    tracker_list: list[tracking.BaseTracker] = []
    if trackers:
        if 'tb' in trackers:
            assert uri, 'URI not provided for TensorBoard tracker'
            tracker_list.append(tracking.TensorBoardTracker(uri))

    # callbacks list
    callbacks_list: list[callbacks.BaseCallback] = [
        callbacks.LoggingCallback(verbose=verbose),
        callbacks.TrackingCallback(trackers=tracker_list),
        callbacks.PreviewCallback(trackers=tracker_list)
    ]

    # return dispatcher
    return callbacks.CallbackDispatcher(callbacks_list)
