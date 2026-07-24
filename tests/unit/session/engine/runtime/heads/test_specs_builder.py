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

# pylint: disable=protected-access

'''Unit tests for head specs module (specs.py).'''

# third-party imports
import pytest
# local imports
import landseg.session.engine.runtime.tasks.heads.specs as head_specs


def test_headspecs_build_raise_invalid_alpha_fn(dataspecs):
    '''
    Given: `DataSpecs` fixture invalid `alpha_fn` config.
    When: `build_headspecs()` is called.
    Then: Raise `ValueError`.
    '''
    with pytest.raises(ValueError, match='Invalid class weights function'):
        head_specs.build_headspecs(
            dataspecs,
            alpha_fn='invalid_function',
            en_beta=None,
            excluded_cls=None
        )


def test_headspecs_build_raise_effective_number_w_no_beta(dataspecs):
    '''
    Given: `DataSpecs` fixture and configs to use 'effective number' to
        calculate class weights but missing `en_beta` value.
    When: `build_headspecs()` is called.
    Then: Raise `ValueError`.
    '''
    with pytest.raises(ValueError, match='Beta parameter missing'):
        head_specs.build_headspecs(
            dataspecs,
            alpha_fn='effective_n',
            en_beta=None,
            excluded_cls=None
        )


@pytest.mark.parametrize('alpha_fn', ('effective_n', 'inverse'))
def test_headspecs_build_success(dataspecs, alpha_fn):
    '''
    Given: `DataSpecs` fixture and configs.
    When: `build_headspecs()` is called.
    Then: Successfully build and return a `HeadSpecs` object.
    '''
    specs = head_specs.build_headspecs(
        dataspecs,
        alpha_fn=alpha_fn,
        en_beta=0.999, # ignored for inverse weights
        excluded_cls=None
    )

    # here we test the wrapper class functionality as well
    assert len(specs) == 2
    assert isinstance(specs['head_1'], head_specs.HeadSpec)
    assert isinstance(specs['head_2'], head_specs.HeadSpec)
    assert isinstance(specs.as_dict()['head_1'], head_specs.HeadSpec)
    assert isinstance(specs.as_dict()['head_2'], head_specs.HeadSpec)


def test_count_to_inv_weights():
    '''
    Given: Class counts.
    When: Converting to inverse weight.
    Then: Weights are computed correctly.
    '''
    fn = head_specs._count_to_inv_weights

    # equal counts -> equal weights
    assert fn([1, 1]) == pytest.approx([0.5, 0.5])

    # smaller count -> larger weight
    assert fn([1, 2]) == pytest.approx([2 / 3, 1 / 3])

    # zero-count classes receive zero weight
    assert fn([1, 0]) == pytest.approx([1.0, 0.0])

    # all-zero counts are invalid
    with pytest.raises(AssertionError):
        fn([0, 0])


def test_count_to_effective_num():
    '''
    Given: Class counts,
    When: Converting to effective number weights.
    Then: Weights are computed correctly.
    '''
    fn = head_specs._count_to_effective_num

    # equal counts -> equal weights
    assert fn([1, 1], b=0.9) == pytest.approx([1.0, 1.0])

    # different counts (b=0.5 gives simple fractions)
    # raw weights: [1, 2/3]
    # normalized to sum=len(counts)=2 -> [1.2, 0.8]
    assert fn([1, 2], b=0.5) == pytest.approx([1.2, 0.8])

    # zero-count classes receive zero weight
    assert fn([1, 0], b=0.5) == pytest.approx([2.0, 0.0])

    # all-zero counts remain all zero
    assert fn([0, 0], b=0.5) == [0, 0]
