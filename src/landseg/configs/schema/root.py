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
import landseg.configs.schema.sections as sec

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
    foundation: sec.DataFoundation = field(default_factory=sec.DataFoundation)
    # data preparation
    transform: sec.DataTransform = field(default_factory=sec.DataTransform)
    # data specfication
    dataspecs: sec.DataSpecs = field(default_factory=sec.DataSpecs)
    # model settings
    models: sec.ModelsConfig = field(default_factory=sec.ModelsConfig)
    # session settings
    session: sec.SessionConfig = field(default_factory=sec.SessionConfig)
    # study settings
    study: sec.StudyConfig = field(default_factory=sec.StudyConfig)
    # pipeline specific CLI flags
    pipeline: sec.PipelineConfig = field(default_factory=sec.PipelineConfig)

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
        self.session.validate()
        # To be completed

    # set parameters
    def set_lr(self, lr: float) -> None:
        '''Set learning rate.'''
        self.session.engine_optim.lr = lr

    def set_weight_decay(self, weight_decay: float) -> None:
        '''Set weight decay.'''
        self.session.engine_optim.weight_decay = weight_decay

    def set_patch_size(self, patch_size: int) -> None:
        '''Set patch size'''
        self.session.data_loader.patch_size = patch_size

    def set_batch_size(self, batch_size: int) -> None:
        '''Set batch size'''
        self.session.data_loader.batch_size = batch_size
