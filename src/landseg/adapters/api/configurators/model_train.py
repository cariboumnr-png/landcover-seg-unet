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
Programmatic API entry
'''

# standard imports
import typing
# local imports
import landseg.adapters.api.configurators as configurators

class TrainingSessionConfigurator(configurators.BaseConfigurator):
    '''Configure a training session.'''

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

    def set_objectives(
        self,
        focal_loss_weight: float,
        dice_loss_weight: float,
        spectral_loss_weight: float,
        tv_loss_weight: float
    ) -> typing.Self:
        '''Set loss weights.'''
        loss_types = self._cfg.session.engine_tasks.loss_configs
        loss_types.focal.weight = focal_loss_weight
        loss_types.dice.weight = dice_loss_weight
        loss_types.spectral.weight = spectral_loss_weight
        loss_types.tv.weight = tv_loss_weight
        return self

    def set_runtime(
        self,
        max_epochs: int,
        patience_epoch: int | None,
        logit_adjust_alpha: float
    ) -> typing.Self:
        '''Set training runtime behaviour.'''
        orchestration = self._cfg.session.orchestration
        orchestration.curriculum.single.phases[0].num_epochs = max_epochs
        orchestration.monitor.patience = patience_epoch
        self._cfg.session.engine_exec.logit_adjust_alpha = logit_adjust_alpha
        return self
