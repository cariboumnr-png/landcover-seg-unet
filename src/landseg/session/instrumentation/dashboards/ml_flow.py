# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''Minimal MLflow experiment dashboard tracker.'''

# standard imports
import pathlib
import typing
# third-party imports
import mlflow
# local imports
import landseg.session.instrumentation.dashboards.base as base

#
if typing.TYPE_CHECKING:
    import torch


# ----- public classes
class MLFlowTracker(base.BaseTracker):
    '''Minimal MLflow tracker for experiment logging.'''

    def __init__(
        self,
        uri: pathlib.Path | str,
        artifact_path: str | None = None,
    ):
        '''
        Initialize the MLflow tracker and ensure an active run exists.

        Args:
            uri:
                Tracking server URI or local tracking directory path.
            artifact_path:
                Optional MLflow experiment name to activate.
        '''
        super().__init__(uri, artifact_path)

        if self.uri:
            mlflow.set_tracking_uri(self.uri)
        if artifact_path:
            mlflow.set_experiment(artifact_path)
        active = mlflow.active_run()
        if active is not None:
            self._run = active
            self._owns_run = False
        else:
            self._run = mlflow.start_run()
            self._owns_run = True

    def log_scalar(self, key: str, value: float, step: int):
        '''
        Log a scalar metric to MLflow.

        Args:
            key:
                Metric name.
            value:
                Scalar metric value.
            step:
                Step index associated with the metric.
        '''
        mlflow.log_metric(key, value, step)

    def log_params(self, key: str, value: typing.Any):
        '''
        Log a parameter entry to MLflow.

        Args:
            key:
                Parameter name.
            value:
                Parameter value.
        '''
        mlflow.log_param(key, value)

    def log_image(self, key: str, image: 'torch.Tensor', step: int, **kwargs):
        '''
        Log an image tensor (currently a no-op for MLflow).

        Args:
            key:
                Image tag or identifier.
            image:
                Image tensor to log.
            step:
                Step index associated with the image.
            **kwargs:
                Additional keyword arguments.
        '''
        # no ops for now
        pass

    def log_artifact(self, fpath: str):
        '''
        Log a local file artifact to MLflow.

        Args:
            fpath:
                File path of the artifact to upload.
        '''
        mlflow.log_artifact(fpath)

    def log_text(self, key: str, text: str, step: int):
        '''
        Log a text entry (currently a no-op for MLflow).

        Args:
            key:
                Text entry tag or identifier.
            text:
                Text payload to log.
            step:
                Step index associated with the text entry.
        '''
        # no ops for now
        pass

    def flush(self):
        '''No-op flush because MLflow writes metrics immediately.'''
        # `mlflow` writes immediately; not required
        ...

    def close(self):
        '''End the active MLflow run if owned by this tracker.'''
        if getattr(self, '_owns_run', True):
            mlflow.end_run()
