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

# pylint: disable=missing-function-docstring

'''
Image normalization and block materialization pipeline.

Consumes raw block split manifests, aggregates image statistics from
training blocks only, normalizes all splits using these statistics, and
writes normalized block artifacts along with updated split mappings.
'''

# standard imports
import typing

# private type
class TransformPaths(typing.Protocol):
    '''Typed pipeline-specific paths container.'''
    @property
    def train_blocks(self) -> str:...
    @property
    def val_blocks(self) -> str:...
    @property
    def test_blocks(self) -> str:...
    @property
    def splits_source_blocks(self) -> str:...
    @property
    def splits_summary(self) -> str:...
    @property
    def label_stats(self) -> str:...
    @property
    def image_stats(self) -> str:...
    @property
    def splits_transformed_blocks(self) -> str:...
    @property
    def schema(self) -> str:...
