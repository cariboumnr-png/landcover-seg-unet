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

'''Training session configurator.'''

# standard imports
import pathlib
import typing
# local imports
import landseg.adapters.api.configurators as configurators

class TrainingSessionConfigurator(configurators.BaseConfigurator):
    '''Configure a training session.'''

    def __init__(
        self,
        *,
        upstream_dpath: str | pathlib.Path | None = None,
        output_dpath: str | pathlib.Path | None = None,
        experiment_root: str | pathlib.Path | None = None,
    ):
        '''
        Initialize the training session configurator.

        Args:
            upstream_dpath:
                Optional directory path for upstream preparation
                outputs.
            output_dpath:
                Optional destination directory for training outputs.
            experiment_root:
                Optional experiment root directory for unified mode.
        '''
        super().__init__(
            'model-train',
            upstream_dpath=upstream_dpath,
            output_dpath=output_dpath,
            experiment_root=experiment_root,
        )

    def set_model(
        self,
        body: str,
        bottleneck: str,
        base_channel: int
    ) -> typing.Self:
        '''
        Set model architecture backbone, bottleneck, and base width.

        Args:
            body:
                Model backbone architecture identifier.
            bottleneck:
                Bottleneck module identifier.
            base_channel:
                Base channel count for the first encoder stage.

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        self.cfg.models.model_body = body
        self.cfg.models.bottleneck = bottleneck
        self.cfg.models.set_base_channel(base_channel)
        return self

    def set_optimization(
        self,
        optimizer: typing.Literal['AdamW'],
        learning_rate: float,
        weight_decay: float,
        scheduler: typing.Literal['CosAnneal']
    ) -> typing.Self:
        '''
        Set optimizer and learning-rate scheduler configuration.

        Args:
            optimizer:
                Optimizer class identifier (`'AdamW'`).
            learning_rate:
                Initial learning rate for the optimizer.
            weight_decay:
                Weight decay regularization coefficient.
            scheduler:
                Learning-rate scheduler identifier (`'CosAnneal'`).

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        engine_optim = self.cfg.session.engine_optim
        engine_optim.opt_cls = optimizer
        engine_optim.lr = learning_rate
        engine_optim.weight_decay = weight_decay
        engine_optim.sched_cls  = scheduler
        return self

    def set_device(self, device: str) -> typing.Self:
        '''Set target execution device.'''
        self.cfg.session.device = device
        return self

    def set_trackers(
        self,
        trackers: list[typing.Literal['tb', 'mlflow']],
        tracking_uri: str | pathlib.Path | None = None,
    ) -> typing.Self:
        '''
        Set experiment tracking backends and optional tracking URI.

        Args:
            trackers:
                Tracker backend identifiers (`'tb'` or `'mlflow'`).
            tracking_uri:
                Optional remote or local tracking server URI.

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        self.cfg.session.trackers = list(trackers)
        self.cfg.session.tracking_uri = str(tracking_uri) if tracking_uri else None
        return self
