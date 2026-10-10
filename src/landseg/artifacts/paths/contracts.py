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
Typed I/O path contract dataclasses for pipeline execution.

This module provides explicit, strongly-typed input and output
artifact path contracts for each pipeline stage.

Public APIs:
    - `WorldGridIO`: I/O contract for world grid tiling.
    - `HarmonizationIO`: I/O contract for data harmonization.
    - `IngestionIO`: I/O contract for data ingestion.
    - `PreparationIO`: I/O contract for data preparation.
    - `TrainingIO`: I/O contract for neural network model training.
    - `EvaluationIO`: I/O contract for model evaluation.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.configs as configs
import landseg.artifacts.paths.data_harmonization as data_harmonization
import landseg.artifacts.paths.data_ingestion as data_ingestion
import landseg.artifacts.paths.data_preparation as data_preparation
import landseg.artifacts.paths.session as session_paths
import landseg.artifacts.paths.world_grid as world_grid


# ----- typing aliases
HarmonizationPaths = data_harmonization.HarmonizationPaths
IngestionPaths = data_ingestion.IngestionPaths
PreparationPaths = data_preparation.PreparationPaths
SessionPaths = session_paths.SessionPaths
WorldGridPaths = world_grid.WorldGridPaths


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class WorldGridIO:
    '''I/O path contract for `world-grid` pipeline.'''
    output: WorldGridPaths

    @classmethod
    def from_config(cls, cfg: 'configs.RootConfig') -> typing.Self:
        '''Construct `WorldGridIO` from root configuration.'''
        return cls(output=WorldGridPaths(cfg.data.world_grid.output_dpath))


@dataclasses.dataclass(frozen=True)
class HarmonizationIO:
    '''I/O path contract for `data-harmonize` pipeline.'''
    grid: WorldGridPaths
    output: HarmonizationPaths

    @classmethod
    def from_config(cls, cfg: 'configs.RootConfig') -> typing.Self:
        '''Construct `HarmonizationIO` from root configuration.'''
        grid_dpath = cfg.data.world_grid.output_dpath
        if not grid_dpath:
            raise ValueError(
                'HarmonizationIO requires upstream world grid output_dpath.'
            )
        return cls(
            grid=WorldGridPaths(grid_dpath),
            output=HarmonizationPaths(cfg.data.harmonization.output_dpath),
        )


@dataclasses.dataclass(frozen=True)
class IngestionIO:
    '''I/O path contract for `data-ingest` pipeline.'''
    harmonization: HarmonizationPaths
    output: IngestionPaths

    @classmethod
    def from_config(cls, cfg: 'configs.RootConfig') -> typing.Self:
        '''Construct `IngestionIO` from root configuration.'''
        hm_dpath = cfg.data.harmonization.output_dpath
        if not hm_dpath:
            raise ValueError(
                'IngestionIO requires upstream harmonization output_dpath.'
            )
        return cls(
            harmonization=HarmonizationPaths(hm_dpath),
            output=IngestionPaths(cfg.data.ingestion.output_dpath),
        )


@dataclasses.dataclass(frozen=True)
class PreparationIO:
    '''I/O path contract for `data-prepare` pipeline.'''
    ingestion: IngestionPaths
    output: PreparationPaths

    @classmethod
    def from_config(cls, cfg: 'configs.RootConfig') -> typing.Self:
        '''Construct `PreparationIO` from root configuration.'''
        ing_dpath = cfg.data.ingestion.output_dpath
        if not ing_dpath:
            raise ValueError(
                'PreparationIO requires upstream ingestion output_dpath.'
            )
        return cls(
            ingestion=IngestionPaths(ing_dpath),
            output=PreparationPaths(cfg.data.preparation.output_dpath),
        )


@dataclasses.dataclass(frozen=True)
class TrainingIO:
    '''I/O path contract for `model-train` pipeline.'''
    preparation: PreparationPaths
    output: SessionPaths

    @classmethod
    def from_config(cls, cfg: 'configs.RootConfig') -> typing.Self:
        '''Construct `TrainingIO` from root configuration.'''
        prep_dpath = cfg.data.preparation.output_dpath
        if not prep_dpath:
            raise ValueError(
                'TrainingIO requires upstream preparation output_dpath.'
            )
        return cls(
            preparation=PreparationPaths(prep_dpath),
            output=SessionPaths(root=cfg.session.output_dpath),
        )


@dataclasses.dataclass(frozen=True)
class EvaluationIO:
    '''I/O path contract for `model-evaluate` pipeline.'''
    preparation: PreparationPaths
    output: SessionPaths

    @classmethod
    def from_config(cls, cfg: 'configs.RootConfig') -> typing.Self:
        '''Construct `EvaluationIO` from root configuration.'''
        prep_dpath = cfg.data.preparation.output_dpath
        if not prep_dpath:
            raise ValueError(
                'EvaluationIO requires upstream preparation output_dpath.'
            )
        return cls(
            preparation=PreparationPaths(prep_dpath),
            output=SessionPaths(root=cfg.session.output_dpath),
        )
