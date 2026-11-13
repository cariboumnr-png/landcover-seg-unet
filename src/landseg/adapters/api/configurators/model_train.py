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

'''
Programmatic API entry
'''

# standard imports
import pathlib
import typing
# local imports
import landseg.adapters.api.configurators as configurators

class TrainingSessionConfigurator(configurators.BaseConfigurator):
    '''Configure a training session.'''

    def __init__(
        self,
        experiment_root: str | pathlib.Path,
        *,
        preparation_dpath: str | pathlib.Path | None = None,
        session_dpath: str | pathlib.Path | None = None,
    ):
        super().__init__(
            experiment_root,
            'model-train',
            preparation_dpath=preparation_dpath,
            session_dpath=session_dpath,
        )

    def set_model(
        self,
        body: str,
        bottleneck: str,
        base_channel: int
    ) -> typing.Self:
        '''Set model body.'''
        self._cfg.models.model_body = body
        self._cfg.models.bottleneck = bottleneck
        self._cfg.models.set_base_channel(base_channel)
        return self

    def set_optimization(
        self,
        optimizer: typing.Literal['AdamW'],
        learning_rate: float,
        weight_decay: float,
        scheduler: typing.Literal['CosAnneal']
    ) -> typing.Self:
        '''Set model optimization.'''
        engine_optim = self._cfg.session.engine_optim
        engine_optim.opt_cls = optimizer
        engine_optim.lr = learning_rate
        engine_optim.weight_decay = weight_decay
        engine_optim.sched_cls  = scheduler
        return self

    def set_device(self, device: str) -> typing.Self:
        '''Set target execution device.'''
        self._cfg.session.device = device
        return self

    def set_trackers(
        self,
        trackers: list[typing.Literal['tb', 'mlflow']],
        tracking_uri: str | pathlib.Path | None = None,
    ) -> typing.Self:
        '''Set experiment tracking backends.'''
        self._cfg.session.trackers = list(trackers)
        self._cfg.session.tracking_uri = str(tracking_uri) if tracking_uri else None
        return self
