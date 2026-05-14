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
Callback dispatcher construction utilities.

Builds a ``CallbackDispatcher`` by assembling logging, tracking, and
visualization callbacks, optionally integrating external experiment
trackers such as TensorBoard or MLflow.

This module provides a simple entry point for configuring session
instrumentation.
'''

# standard imports
import typing
# local imports
import landseg.session.instrumentation.callbacks as callbacks
import landseg.session.instrumentation.tracking as tracking

def build_dispatcher(
    trackers: list[typing.Literal['tb', 'mlflow']] | None = None,
    uri: str | None = None,
    artifact_path: str | None = None,
    reclass_color_map: dict[int, list[int]] | None = None,
    verbose: bool = True
) -> callbacks.CallbackDispatcher:
    '''
    Construct a callback dispatcher with logging and tracking support.

    Builds a dispatcher composed of standard callbacks for logging,
    experiment tracking, and preview generation. Optionally attaches
    external trackers based on configuration.

    Args:
        trackers: List of tracker identifiers to enable (e.g., 'tb',
            'mlflow').
        uri: Tracking backend URI (required for enabled trackers).
        artifact_path: Artifact storage path (required for MLflow).
        reclass_color_map: Optional mapping for visualization
            reclassification.
        verbose: Whether logging callback should emit console output.

    Returns:
        CallbackDispatcher:
            Dispatcher managing all configured callbacks.

    Raises:
        AssertionError:
            If required tracker configuration is missing.

    Notes:
        - Trackers are optional and only initialized if specified.
        - All callbacks are registered in a fixed execution order:
          logging → tracking → preview.
    '''

    # trackers list
    tracker_list: list[tracking.BaseTracker] = []
    if trackers:
        if 'tb' in trackers:
            assert uri, 'URI not provided for TensorBoard tracker'
            tracker_list.append(tracking.TensorBoardTracker(uri))
        if 'mlflow' in trackers:
            assert uri and artifact_path
            tracker_list.append(tracking.MLFlowTracker(uri, artifact_path))

    # callbacks list
    callbacks_list: list[callbacks.BaseCallback] = [
        callbacks.LoggingCallback(verbose=verbose),
        callbacks.ScalarsCallback(trackers=tracker_list),
        callbacks.ImageCallback(
            trackers=tracker_list,
            reclass_color_map=reclass_color_map
        )
    ]

    # return dispatcher
    return callbacks.CallbackDispatcher(callbacks_list)
