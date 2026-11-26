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

'''TensorBoard experiment dashboard tracker.'''

# standard imports
import pathlib
import typing
# third-party imports
import torch.utils.tensorboard as tensorboard
# local imports
import landseg.session.instrumentation.dashboards.base as base

#
if typing.TYPE_CHECKING:
    import torch


# ----- public classes
class TensorBoardTracker(base.BaseTracker):
    '''TensorBoard tracker wrapping `SummaryWriter`.'''

    def __init__(self, uri: pathlib.Path):
        '''
        Initialize the TensorBoard summary writer.

        Args:
            uri:
                Directory path for TensorBoard event files.
        '''
        super().__init__(uri, None)
        self.writer = tensorboard.SummaryWriter(log_dir=self.uri)

    def log_scalar(self, key: str, value: float, step: int):
        '''
        Log a scalar metric to TensorBoard.

        Args:
            key:
                Scalar tag name.
            value:
                Scalar metric value.
            step:
                Step index associated with the scalar.
        '''
        self.writer.add_scalar(key, value, step)

    def log_params(self, key: str, value: typing.Any):
        '''
        Log a parameter entry (no-op because not natively supported).

        Args:
            key:
                Parameter name.
            value:
                Parameter value.
        '''
        # not natively supported in TensorBoard
        ...

    def log_image(self, key: str, image: 'torch.Tensor', step: int, **kwargs):
        '''
        Log an image tensor to TensorBoard.

        Args:
            key:
                Image tag name.
            image:
                Image tensor to log.
            step:
                Step index associated with the image.
            **kwargs:
                Additional keyword arguments forwarded to `add_image`.
        '''
        self.writer.add_image(key, image, step, **kwargs)

    def log_artifact(self, fpath: str):
        '''
        Log a file artifact (no-op in this tracker).

        Args:
            fpath:
                File path of the artifact.
        '''
        # not implemented here
        ...

    def log_text(self, key: str, text: str, step: int):
        '''
        Log a text entry to TensorBoard.

        Args:
            key:
                Text tag name.
            text:
                Text payload to log.
            step:
                Step index associated with the text.
        '''
        self.writer.add_text(key, text, step)

    def flush(self):
        '''Flush buffered TensorBoard events to disk.'''
        self.writer.flush()

    def close(self):
        '''Close the underlying TensorBoard summary writer.'''
        self.writer.close()
