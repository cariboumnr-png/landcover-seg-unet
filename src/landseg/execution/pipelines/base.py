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
Base pipeline ABC
'''

# standard imports
import abc
import time
import typing
# third-party imports
import torch
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.core as core
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session


T = typing.TypeVar('T')


class Pipeline(abc.ABC, typing.Generic[T]):
    '''Pipeline ABC'''

    def __init__(
        self,
        config: configs.RootConfig,
        *,
        artifact_paths: artifacts.ArtifactPaths | None = None,
        disable_console_logging: bool = False
    ) -> None:
        '''Init'''
        self.config = config

        if artifact_paths is None:
            self.artifact_paths = artifacts.ArtifactPaths.from_config(self.config)
        else:
            self.artifact_paths = artifact_paths

        self.console_level = (
            None
            if disable_console_logging
            else self.config.execution.console_level
        )

        self.logger: T

        self.timer: dict[str, float] = {}
        self.dataspecs: core.DataSpecs | None = None

    @abc.abstractmethod
    def run(self) -> typing.Any:
        '''Initialize a pipeline runner and run end-to-end process.'''

    @abc.abstractmethod
    def validate(self) -> None:
        '''Validate pipeline environment and upstream requirements.'''

    @typing.overload
    def build_session_runner(
        self,
        *,
        mode_override: typing.Literal['evaluate'],
        eval_split: typing.Literal['val', 'test'] = 'val',
    ) -> session.EpochRunner: ...

    @typing.overload
    def build_session_runner(
        self,
        *,
        mode_override: typing.Literal['continuous'],
        eval_split: typing.Literal['val', 'test'] = 'val',
    ) -> session.ContinuousRunner: ...

    @typing.overload
    def build_session_runner(
        self,
        *,
        mode_override: typing.Literal['curriculum'],
        eval_split: typing.Literal['val', 'test'] = 'val',
    ) -> session.CurriculumRunner: ...

    @typing.overload
    def build_session_runner(
        self,
        *,
        mode_override: None = None,
        eval_split: typing.Literal['val', 'test'] = 'val',
    ) -> session.ContinuousRunner | session.CurriculumRunner: ...

    def build_session_runner(
        self,
        *,
        mode_override: typing.Literal['continuous', 'curriculum', 'evaluate'] | None = None,
        eval_split: typing.Literal['val', 'test'] = 'val',
    ) -> session.EpochRunner | session.ContinuousRunner | session.CurriculumRunner :
        '''doc.'''
        if not isinstance(self.logger, session.SessionLogger):
            raise ValueError(
                '"build_session_runner()" method only works when the pipeline '
                'is either "model-train" or "model-evaluate".'
            )

        # collect artifacts and build `DataSpecs` if not already cached
        if self.dataspecs is not None:
            dataspecs = self.dataspecs
        else:
            self.logger.log('INFO', '[START] Data specifications setup')
            start_t = time.perf_counter()
            dataspecs = geopipe.build_dataspec(
                self.artifact_paths,
                mode='default',
                ids_domain_name=self.config.data.specification.domain_ids_name,
                vec_domain_name=self.config.data.specification.domain_vec_name
            )
            self.timer['data'] = time.perf_counter() - start_t
            self.logger.log('INFO', f'[COMPLETE] Data specs setup (D_{self.timer['data']:.2f}s)')
            self.dataspecs = dataspecs

        for s in dataspecs.summary:
            self.logger.log('INFO', s)
        self.logger.log_sep()

        # setup the model
        self.logger.log('INFO', '[START] Model assembly')
        start_t = time.perf_counter()
        model = models.build_multihead_unet(
            patch_size=self.config.session.dataloader.patch_size,
            dataspecs=dataspecs,
            unet_backbone_config=self.config.models.unet_backbone_config,
            conditioning_config=self.config.models.conditioning_config,
            enable_clamp=self.config.models.numeric_safety.enable_clamp,
            clamp_range=self.config.models.numeric_safety.clamp_range
        )
        self.timer['model'] = time.perf_counter() - start_t
        self.logger.log('INFO', f'[COMPLETE] Model assembly (D_{self.timer['model']:.2f}s)')

        # summarize inputs and log
        total_p = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.set_inputs({
            'system': {
                'device':c.DEVICE_NAME,
                'torch_version': torch.__version__
            },
            'model': {
                'backbone': 'unet',
                'total_parameters': total_p,
                'trainable_parameters': trainable_p,
                'heads': list(dataspecs.heads.class_counts.keys())
            },
            'data': {
                'patch_size': self.config.session.dataloader.patch_size
            },
            'dataspecs': dataspecs.to_dict()
        })
        self.logger.log_sep()

        # build and return the session runner
        match mode_override:
            case 'evaluate': session_type = 'evaluate'
            case 'continuous': session_type = 'continuous'
            case 'curriculum': session_type = 'curriculum'
            case None: session_type = self.config.session.training_mode

        runner = session.build_session_runner(
            dataspecs=dataspecs,
            model=model,
            config=self.config.session,
            context=session.SessionBuildContext(
                device=c.DEVICE,
                session_paths=self.artifact_paths.session,
                eval_dataset=eval_split,
                logger=self.logger
            ),
            session_type=session_type,
        )
        return runner
