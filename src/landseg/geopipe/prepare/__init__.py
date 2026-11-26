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
Top-level namespace for `landseg.geopipe.prepare`.

Exposes public dataset context builders, partitioners, materializers,
preparation schema, handlers, and schema generators via lazy resolution.

Public APIs:
    - `PreparationContext`: Resolved execution context for prepare.
    - `PreparationLogger`: Specialized logger for preparation runs.
    - `build_preparation_context`: Build context from catalog & schema.
    - `run_data_preparation`: Pipeline runner.
    - `validators`: Preflight validation namespace.
'''

# pylint: disable=import-outside-toplevel

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg._lazy as _lazy


class _Validators: # pylint: disable=R0903
    '''Validation namespace for preparation preflight checks.'''

    @staticmethod
    def dataset_semantics(
        schema: geo_core.DataSchema,
        config: contracts.PreparationDatasetViewConfig | None = None,
        *,
        features: dict[str, str] | list[str] | None = None,
        targets: dict[str, str | geo_core.LabelScheme] | None = None,
        focal_target: str | None = None,
    ) -> tuple[semantics.FeatureSelection, semantics.TargetHeadsContext, str]:
        '''
        Validate feature channels, target heads, and focal head.

        Args:
            schema:
                Ingested catalog data schema mapping bands and targets.
            config:
                Optional dataset view config protocol instance.
            features:
                Optional feature selection by band list or scheme.
            targets:
                Optional target hierarchy selection or label scheme.
            focal_target:
                Optional focal target head or class name.

        Returns:
            tuple[semantics.FeatureSelection, semantics.TargetHeadsContext, str]:
                Resolved features, target heads context, and focal head
                name.

        Raises:
            KeyError:
                If a requested feature, scheme, or focal target is
                unrecognized.
            TypeError:
                If a target configuration has an unexpected type.
            ValueError:
                If schemes or target heads are missing or malformed.
        '''
        import landseg.geopipe.prepare.dataset.semantics as semantics

        features_cfg = (
            features if features is not None
            else getattr(config, 'features', None)
        )
        targets_cfg = (
            targets if targets is not None
            else getattr(config, 'targets', None)
        )
        focal_cfg = (
            focal_target if focal_target is not None
            else getattr(config, 'focal_target', None)
        )

        sel_features = semantics.resolve_feature_channels(schema, features_cfg)
        sel_targets = semantics.resolve_target_heads(schema, targets_cfg)
        focal_head = semantics.resolve_focal_head(sel_targets, focal_cfg)
        return sel_features, sel_targets, focal_head

validators = _Validators()


__all__ = [
    'PreparationContext',
    'PreparationLogger',
    'build_preparation_context',
    'run_data_preparation',
    'validators',
]


# for static check
if typing.TYPE_CHECKING:
    import landseg.geopipe.contracts as contracts
    import landseg.geopipe.core as geo_core
    import landseg.geopipe.prepare.dataset.semantics as semantics

    from .context import (
        PreparationContext,
        build_preparation_context,
    )
    from .logger import PreparationLogger
    from .pipeline import run_data_preparation


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.context': [
        'PreparationContext',
        'build_preparation_context',
    ],
    '.logger': ['PreparationLogger'],
    '.pipeline': ['run_data_preparation'],
})
