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
import landseg.session.engine.runtime.tasks.metrics.segmentation.builder as builder
import landseg.session.engine.runtime.tasks.metrics.segmentation.confusion_matrix as cm_module


def test_build_headmetrics(dataspecs):
    '''
    Given: `HeadSpecs` describing prediction heads and an ignore index.
    When: Calling `build_headmetrics`.
    Then: Return a `HeadMetrics` container mapping head names to
        configured `ConfusionMatrix` objects.
    '''
    hspecs = head_specs.build_headspecs(dataspecs, alpha_fn='inverse')

    hmetrics = builder.build_headmetrics(hspecs, ignore_index=255)

    assert isinstance(hmetrics, builder.HeadMetrics)
    assert len(hmetrics) == 2
    assert isinstance(hmetrics.as_dict()['head_1'], cm_module.ConfusionMatrix)
    assert isinstance(hmetrics.as_dict()['head_2'], cm_module.ConfusionMatrix)
    assert isinstance(hmetrics['head_1'], cm_module.ConfusionMatrix)
    assert isinstance(hmetrics['head_2'], cm_module.ConfusionMatrix)
