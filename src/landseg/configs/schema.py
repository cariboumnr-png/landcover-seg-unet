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

# pylint: disable=too-many-instance-attributes

'''
This module mirrors the Hydra/YAML config tree using Python dataclasses,
suitable for OmegaConf structured configs.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# local imports
import landseg.configs._schema as s

# alias
field = dataclasses.field

# ------------------------------EXECUTION CONFIGS------------------------------
@dataclasses.dataclass
class _ExecutionContext:
    '''Mutable execution context.'''
    exp_root: str = './experiment' # root directory for this experiment run
    verbosity: str = 'full' # 'full', 'logging_only', 'silent'
    dev_settings: str | None = None # developer-only override config

# --------------------------------ROOT  CONFIGS--------------------------------
@dataclasses.dataclass
class RootConfig:
    '''Root structured config for landseg.'''

    # execution configs
    execution: _ExecutionContext = field(default_factory=_ExecutionContext)
    # raw input data and configs
    foundation: s.DataFoundation = field(default_factory=s.DataFoundation)
    # data preparation
    transform: s.DataTransform = field(default_factory=s.DataTransform)
    # data specfication
    dataspecs: s.DataSpecs = field(default_factory=s.DataSpecs)
    # model settings
    models: s.ModelsConfig = field(default_factory=s.ModelsConfig)
    # session settings
    session: s.SessionConfig = field(default_factory=s.SessionConfig)
    # study settings
    study: s.StudyConfig = field(default_factory=s.StudyConfig)
    # pipeline specific CLI flags
    pipeline: s.PipelineConfig = field(default_factory=s.PipelineConfig)

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Dict representation.'''
        return dataclasses.asdict(self)

    def validate_all(self) -> None:
        '''Validation'''
        # delegated to subtrees.
        self.foundation.validate()
        self.transform.validate()
        self.models.validate()
        # TBD

    # set parameters
    def set_lr(self, lr: float) -> None:
        '''Set learning rate.'''
        self.session.components.optimization.lr = lr

    def set_weight_decay(self, weight_decay: float) -> None:
        '''Set weight decay.'''
        self.session.components.optimization.weight_decay = weight_decay

    def set_patch_size(self, patch_size: int) -> None:
        '''Set patch size'''
        self.session.components.loader.patch_size = patch_size

    def set_batch_size(self, batch_size: int) -> None:
        '''Set batch size'''
        self.session.components.loader.batch_size = batch_size
