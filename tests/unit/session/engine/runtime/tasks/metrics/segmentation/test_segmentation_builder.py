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

'''Unit tests for segmentation metrics builder module (builder.py).'''

# local imports
import landseg.session.engine.runtime.tasks.heads.specs as head_specs
import landseg.session.engine.runtime.tasks.metrics.segmentation.builder as builder_module
import landseg.session.engine.runtime.tasks.metrics.segmentation.confusion_matrix as cm_module


def test_headmetrics_wrapper():
    '''
    Given: A mapping of head names to `ConfusionMatrix` instances.
    When: Instantiating `HeadMetrics`.
    Then: Support dictionary key access `__getitem__`, `__len__`, and `as_dict`.
    '''
    cm_a = cm_module.ConfusionMatrix(
        num_classes=2,
        ignore_index=255,
        parent_class_1b=None,
        exclude_class_1b=None
    )
    cm_b = cm_module.ConfusionMatrix(
        num_classes=3,
        ignore_index=255,
        parent_class_1b=1,
        exclude_class_1b=(3,)
    )

    mapping = {'head_a': cm_a, 'head_b': cm_b}
    hmetrics = builder_module.HeadMetrics(mapping)

    assert len(hmetrics) == 2
    assert hmetrics['head_a'] is cm_a
    assert hmetrics['head_b'] is cm_b
    assert hmetrics.as_dict() == mapping


def test_build_headmetrics():
    '''
    Given: `HeadSpecs` describing prediction heads and an ignore index.
    When: Calling `build_headmetrics`.
    Then: Return a `HeadMetrics` container mapping head names to configured `ConfusionMatrix` objects.
    '''
    head_1 = head_specs.HeadSpec(
        name='head_1',
        count=[10, 20],
        loss_alpha=[0.5, 0.5],
        parent_head=None,
        parent_cls=None,
        weight=1.0,
        exclude_cls=None
    )
    head_2 = head_specs.HeadSpec(
        name='head_2',
        count=[5, 15, 25],
        loss_alpha=[0.3, 0.3, 0.4],
        parent_head='head_1',
        parent_cls=1,
        weight=1.0,
        exclude_cls=(3,)
    )

    hspecs = head_specs.HeadSpecs({'head_1': head_1, 'head_2': head_2})

    hmetrics = builder_module.build_headmetrics(hspecs, ignore_index=255)

    assert isinstance(hmetrics, builder_module.HeadMetrics)
    assert len(hmetrics) == 2
    assert 'head_1' in hmetrics.as_dict()
    assert 'head_2' in hmetrics.as_dict()

    cm1 = hmetrics['head_1']
    assert isinstance(cm1, cm_module.ConfusionMatrix)
    assert cm1.n_cls == 2
    assert cm1.ignore_index == 255
    assert cm1.parent_class_1b is None
    assert cm1.exclude_class_1b is None

    cm2 = hmetrics['head_2']
    assert isinstance(cm2, cm_module.ConfusionMatrix)
    assert cm2.n_cls == 3
    assert cm2.ignore_index == 255
    assert cm2.parent_class_1b == 1
    assert cm2.exclude_class_1b == (3,)
