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
Minimal MLFlow tracker
'''

# standard imports
import typing
# third-party imports
import mlflow
# local imports
import landseg.session.instrumentation.tracking as tracking

#
if typing.TYPE_CHECKING:
    import numpy.typing
    import torch

#
class MLFlowTracker(tracking.BaseTracker):
    '''Minimal MLflow tracker for experiment logging.'''

    def __init__(self, uri: str, artifact_path: str):
        super().__init__(uri, artifact_path)

        mlflow.set_tracking_uri(self.uri)
        if artifact_path:
            mlflow.set_experiment(artifact_path)
        # start run immediately (simple semantics)
        self._run = mlflow.start_run()

    def log_scalar(self, key: str, value: float, step: int):
        mlflow.log_metric(key, value, step)

    def log_params(self, key: str, value: typing.Any):
        mlflow.log_param(key, value)

    def log_image(self, key: str, image: 'numpy.typing.NDArray | torch.Tensor', step: int):
        # on ops for now
        pass

    def log_artifact(self, fpath: str):
        mlflow.log_artifact(fpath)

    def flush(self): ...
        # MLflow writes immediately; not required

    def close(self):
        mlflow.end_run()
