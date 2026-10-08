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
Base pipeline abstractions.

Provides common foundation, geospatial data pipeline, and machine
learning session pipeline base classes.

Public APIs:
    - `BasePipeline`: Root abstract pipeline runner class.
    - `GeoPipeline`: Base class for geospatial data pipelines.
    - `SessionPipeline`: Base class for neural network session pipelines.
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
import landseg.execution.preflight as preflight
import landseg.models as models
import landseg.session as session


# ----- public classes
class BasePipeline(abc.ABC):
    '''Base pipeline runner class common to all pipelines.'''

    pipeline_name: str = 'unknown'
    context: typing.Any
    logger: typing.Any
    pipeline_paths: typing.Any

    def __init__(
        self,
        config: configs.RootConfig,
        *,
        artifact_paths: artifacts.ArtifactPaths | None = None,
        disable_console_logging: bool = False
    ) -> None:
        '''Initialize common pipeline configuration and attributes.'''
        self.config = config
        self.artifact_paths = (
            artifact_paths
            if artifact_paths is not None
            else artifacts.ArtifactPaths.from_config(self.config)
        )
        self.console_level = (
            None
            if disable_console_logging
            else self.config.execution.console_level
        )
        self.timer: dict[str, float] = {}

        self._resolve_pipeline_paths()

    @abc.abstractmethod
    def run(self) -> typing.Any:
        '''Initialize a pipeline runner and run end-to-end process.'''

    @abc.abstractmethod
    def _resolve_pipeline_paths(self) -> None:
        '''Resolve canonical artifact paths based on pipeline name.'''

    @abc.abstractmethod
    def _create_logger(self) -> typing.Any:
        '''Instantiate and configure the logger for this pipeline.'''

    @abc.abstractmethod
    def _build_context(self) -> typing.Any:
        '''Build the necessary pipeline running context object.'''

    def _initialize_run(self) -> None:
        '''Initialize run directories, persist config, and open logger.'''
        self._validate()

        assert isinstance(self.pipeline_paths, artifacts.PipelineArtifactsPaths)
        self.pipeline_paths.init_pipeline_folders()

        config_ctrl = artifacts.Controller[dict](self.pipeline_paths.config)
        config_ctrl.persist(self.config.as_dict)

        self.logger = self._create_logger()

    def _validate(self) -> typing.Any:
        '''Validate pipeline environment and upstream requirements.'''
        preflight.assert_target_prerequisites(
            self.pipeline_name,
            self.artifact_paths,
        )


class GeoPipeline(BasePipeline):
    '''Base class for geospatial data pipelines.'''

    def _resolve_pipeline_paths(self) -> None:
        '''Resolve canonical artifact paths for geospatial pipelines.'''
        match self.pipeline_name:
            case 'world-grid':
                self.pipeline_paths = self.artifact_paths.world_grid
            case 'data-harmonize':
                self.pipeline_paths = self.artifact_paths.data_harmonization
            case 'data-ingest':
                self.pipeline_paths = self.artifact_paths.data_ingestion
            case 'data-prepare':
                self.pipeline_paths = self.artifact_paths.data_preparation
            case _:
                raise ValueError(f'Invalid pipeline name: {self.pipeline_name}')


class SessionPipeline(BasePipeline):
    '''Base class for neural network session pipelines.'''

    context: core.DataSpecs
    logger: session.SessionLogger
    pipeline_paths: artifacts.SessionPaths

    @abc.abstractmethod
    def _build_context(self) -> core.DataSpecs:
        '''Build the necessary pipeline running context object.'''

    def _resolve_pipeline_paths(self) -> None:
        '''Resolve session artifact paths.'''
        self.pipeline_paths = self.artifact_paths.session

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
    ) -> session.EpochRunner | session.ContinuousRunner | session.CurriculumRunner:
        '''
        Assemble dataspecs, model, and return configured session runner.

        Args:
            mode_override:
                optional session mode to override configuration.
            eval_split:
                dataset split to use during evaluation.

        Returns:
            session.EpochRunner | session.ContinuousRunner | session.CurriculumRunner:
                instantiated session runner.
        '''
        dataspecs = self._build_context()

        for s in dataspecs.summary:
            self.logger.info(s)
        self.logger.log_sep()

        # setup the model
        self.logger.info('[START] Model assembly')
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
        self.logger.info(f'[COMPLETE] Model assembly (D_{self.timer['model']:.2f}s)')

        # summarize inputs and log
        total_p = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.set_inputs({
            'system': {
                'device': c.DEVICE_NAME,
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

        return session.build_session_runner(
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
