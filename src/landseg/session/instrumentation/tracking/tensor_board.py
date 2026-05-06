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
Tensorboard tracker
'''

# standard imports
import typing
# third-party imports
import torch.utils.tensorboard as tensorbaord
# local imports
import landseg.session.instrumentation.tracking as tracking

#
if typing.TYPE_CHECKING:
    import numpy.typing
    import torch

#
class TensorBoardTracker(tracking.BaseTracker):
    '''Tensorboard tracker class.'''

    def __init__(self, uri: str):
        super().__init__(uri, None)
        self.writer = tensorbaord.SummaryWriter(log_dir=self.uri)

    def log_scalar(self, key: str, value: float, step: int):
        self.writer.add_scalar(key, value, step)

    def log_params(self, key: str, value: typing.Any): ...
        # not natively supported in TensorBoard

    def log_image(self, key: str, image: 'numpy.typing.NDArray | torch.Tensor', step: int):
        self.writer.add_image(key, image, step)

    def log_artifact(self, fpath: str): ...
        # not implemented here

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
