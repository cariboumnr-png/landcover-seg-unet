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
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session


# ----- public classes
class BasePipeline(abc.ABC):
    '''Base pipeline runner class common to all pipelines.'''

    pipeline_name: str = 'unknown'

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
        self.context: typing.Any = None
        self.logger: typing.Any = None
        self.pipeline_paths: typing.Any = self._resolve_pipeline_paths()

    @abc.abstractmethod
    def run(self) -> typing.Any:
        '''Initialize a pipeline runner and run end-to-end process.'''

    @abc.abstractmethod
    def validate(self) -> None:
        '''Validate pipeline environment and upstream requirements.'''

    @abc.abstractmethod
    def _create_logger(self) -> typing.Any:
        '''Instantiate and configure the logger for this pipeline.'''

    def _initialize_run(self) -> None:
        '''Initialize run directories, persist config, and open logger.'''
        if self.logger is not None:
            return

        if isinstance(self.pipeline_paths, artifacts.PipelineArtifactsPaths):
            self.pipeline_paths.init_pipeline_folders()
            config_ctrl = artifacts.Controller[dict](self.pipeline_paths.config)
            config_ctrl.persist(self.config.as_dict)

        self.logger = self._create_logger()

    def _resolve_pipeline_paths(self) -> typing.Any:
        '''Resolve canonical artifact paths based on pipeline name.'''
        return None


class GeoPipeline(BasePipeline):
    '''Base class for geospatial data pipelines.'''

    def _resolve_pipeline_paths(self) -> typing.Any:
        '''Resolve canonical artifact paths for geospatial pipelines.'''
        match self.pipeline_name:
            case 'world-grid':
                return None
            case 'data-harmonize':
                return self.artifact_paths.data_harmonization
            case 'data-ingest':
                return self.artifact_paths.data_ingestion
            case 'data-prepare':
                return self.artifact_paths.data_preparation
            case _:
                return None


class SessionPipeline(BasePipeline):
    '''Base class for neural network session pipelines.'''

    logger: session.SessionLogger | None
    pipeline_paths: artifacts.SessionPaths

    def __init__(
        self,
        config: configs.RootConfig,
        *,
        artifact_paths: artifacts.ArtifactPaths | None = None,
        disable_console_logging: bool = False
    ) -> None:
        '''Initialize session pipeline with dataspecs cache.'''
        super().__init__(
            config,
            artifact_paths=artifact_paths,
            disable_console_logging=disable_console_logging
        )
        self.dataspecs: core.DataSpecs | None = None

    def _resolve_pipeline_paths(self) -> artifacts.SessionPaths:
        '''Resolve session artifact paths.'''
        return self.artifact_paths.session

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
        self._initialize_run()
        assert isinstance(self.logger, session.SessionLogger)

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
