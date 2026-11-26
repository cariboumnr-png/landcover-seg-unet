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

'''Abstract base class for experiment dashboard trackers.'''

# standard imports
import abc
import pathlib
import typing

#
if typing.TYPE_CHECKING:
    import torch


# ----- public classes
class BaseTracker(abc.ABC):
    '''Interface of the base tracker.'''

    def __init__(
        self,
        uri: pathlib.Path | str,
        artifact_path: str | None = None
    ):
        '''
        Initialize the dashboard tracker.

        Args:
            uri:
                Tracking directory path or backend tracking URI.
            artifact_path:
                Optional experiment or artifact storage identifier.
        '''
        if isinstance(uri, pathlib.Path):
            self.uri = uri.absolute().as_uri() # safe URI generation
        else:
            self.uri = str(uri)
        self.artifact_path = artifact_path

    @abc.abstractmethod
    def log_scalar(self, key: str, value: float, step: int):
        '''
        Log a scalar metric value at a given step.

        Args:
            key:
                Metric name or tag.
            value:
                Scalar metric value to log.
            step:
                Global or epoch step index.
        '''
        ...

    @abc.abstractmethod
    def log_params(self, key: str, value: typing.Any):
        '''
        Log a configuration parameter or metadata entry.

        Args:
            key:
                Parameter name.
            value:
                Parameter value to record.
        '''
        ...

    @abc.abstractmethod
    def log_image(self, key: str, image: 'torch.Tensor', step: int, **kwargs):
        '''
        Log an image tensor at a given step.

        Args:
            key:
                Image tag or identifier.
            image:
                Image tensor to log.
            step:
                Global or epoch step index.
            **kwargs:
                Additional backend-specific keyword arguments.
        '''
        ...

    @abc.abstractmethod
    def log_artifact(self, fpath: str):
        '''
        Log a local file artifact to the tracking backend.

        Args:
            fpath:
                File path of the artifact to log.
        '''
        ...

    @abc.abstractmethod
    def log_text(self, key: str, text: str, step: int):
        '''
        Log a text entry at a given step.

        Args:
            key:
                Text entry tag or identifier.
            text:
                Text payload to log.
            step:
                Global or epoch step index.
        '''
        ...

    @abc.abstractmethod
    def flush(self):
        '''Flush buffered events to the tracking backend.'''
        ...

    @abc.abstractmethod
    def close(self):
        '''Close the tracker and release underlying resources.'''
        ...
