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

# pylint: disable=missing-function-docstring

'''Unit tests for `DataSpec` core contract (data_specs.py)'''

# local imports
import landseg.core as core

# ----- composition
def test_dataspecs_fixture_constructs_valid_instance(dataspecs):
    assert dataspecs.name == "test_dataset"
    assert dataspecs.mode == "default"

    assert isinstance(dataspecs.meta, core.Meta)
    assert isinstance(dataspecs.heads, core.Heads)
    assert isinstance(dataspecs.splits, core.Splits)
    assert isinstance(dataspecs.domains, core.Domains)


# ----- image specs
def test_image_collects_spectral_channels():
    image = core.Meta.Image(
        num_channels=4,
        height_width=256,
        array_key='image',
        band_map={
            'red': 0,
            'green': 1,
            'nir': 2,
            'foo': 3,
        },
    )

    assert image.spec_channels == [0, 1, 2]
    assert not image.topo_channels


def test_image_collects_topographic_channels():
    image = core.Meta.Image(
        num_channels=5,
        height_width=256,
        array_key='image',
        band_map={
            'dem': 0,
            'slope': 1,
            'cos_aspect': 2,
            'sin_aspect': 3,
            'tpi': 4,
        },
    )

    assert not image.spec_channels
    assert image.topo_channels == [0, 1, 2, 3, 4]


def test_image_band_names_are_case_insensitive():
    image = core.Meta.Image(
        num_channels=3,
        height_width=256,
        array_key='image',
        band_map={
            'Red': 0,
            'NIR': 1,
            'DEM': 2,
        },
    )

    assert image.spec_channels == [0, 1]
    assert image.topo_channels == [2]


def test_image_ignores_unknown_band_names():
    image = core.Meta.Image(
        num_channels=3,
        height_width=256,
        array_key='image',
        band_map={
            'foo': 0,
            'bar': 1,
            'baz': 2,
        },
    )

    assert not image.spec_channels
    assert not image.topo_channels


def test_image_preserves_channel_indices():
    image = core.Meta.Image(
        num_channels=10,
        height_width=256,
        array_key='image',
        band_map={
            'red': 7,
            'dem': 2,
        },
    )

    assert image.spec_channels == [7]
    assert image.topo_channels == [2]

# ----- domains
def test_domain_accept_expected_structure():
    domains = core.Domains(
        train=core.Domains.Dom(
            ids_domain={'blk1': 1},
            vec_domain={'blk1': [0.1, 0.2]}
        ),
        val=core.Domains.Dom(ids_domain=None, vec_domain=None),
        test=core.Domains.Dom(ids_domain=None, vec_domain=None),
        ids_num=3,
        vec_dim=2,
    )

    assert domains.train['ids_domain'] and domains.train['vec_domain']
    assert domains.train['ids_domain']['blk1'] == 1
    assert domains.train['vec_domain']['blk1'] == [0.1, 0.2]
    assert domains.ids_num == 3
    assert domains.vec_dim == 2
