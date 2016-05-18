import numpy

import pytest
import numpy.testing as nptest

from probscale.probscale import _minimal_norm
from probscale import transforms


def test__mask_out_of_bounds():
    known = numpy.array([numpy.nan, numpy.nan, 0.1, 0.5, 0.9, numpy.nan, numpy.nan])
    x = [-0.1, 0, 0.1, 0.5, 0.9, 1.0, 1.1]
    result = transforms._mask_out_of_bounds(x)
    nptest.assert_array_equal(known, result)


def test__clip_out_of_bounds():
    known = numpy.array([0.0, 0.0, 0.1, 0.5, 0.9, 1.0, 1.0])
    x = [-0.1, 0, 0.1, 0.5, 0.9, 1.0, 1.1]
    result = transforms._clip_out_of_bounds(x)
    diff = numpy.abs(result - known)
    assert numpy.all(diff < 0.0001)


class Mixin_Transform(object):
    known_input_dims = 1
    known_output_dims = 1
    known_is_separable = True
    known_has_inverse = True

    def test_input_dims(self):
        assert hasattr(self.trans, 'input_dims')
        assert self.trans.input_dims == self.known_input_dims

    def test_output_dims(self):
        assert hasattr(self.trans, 'output_dims')
        assert self.trans.output_dims == self.known_output_dims

    def test_is_separable(self):
        assert hasattr(self.trans, 'is_separable')
        assert self.trans.is_separable == self.known_is_separable

    def test_has_inverse(self):
        assert hasattr(self.trans, 'has_inverse')
        assert self.trans.has_inverse == self.known_has_inverse

    def test_dist(self):
        assert hasattr(self.trans, 'dist')
        assert self.trans.dist == _minimal_norm

    def test_transform_non_affine(self):
        assert hasattr(self.trans, 'transform_non_affine')
        diff = numpy.abs(self.trans.transform_non_affine([0.5]) - self.known_tras_na)
        assert numpy.all(diff < 0.0001)

    def test_inverted(self):
        assert hasattr(self.trans, 'inverted')

    def test_bad_non_pos(self):
        with pytest.raises(ValueError):
            self._trans(_minimal_norm, nonpos='junk')

    def test_non_pos_clip(self):
        self._trans(_minimal_norm, nonpos='clip')


class Test_ProbTransform(Mixin_Transform):
    def setup(self):
        self._trans = transforms.ProbTransform
        self.trans = transforms.ProbTransform(_minimal_norm)
        self.known_tras_na = [-2.569150498]

    def test_inverted(self):
        inv_trans = self.trans.inverted()
        assert self.trans.dist == inv_trans.dist
        assert self.trans.factor == inv_trans.factor
        assert self.trans.nonpos == inv_trans.nonpos


class Test_QuantileTransform(Mixin_Transform):
    def setup(self):
        self._trans = transforms.QuantileTransform
        self.trans = transforms.QuantileTransform(_minimal_norm)
        self.known_tras_na = [69.1464492]

    def test_inverted(self):
        inv_trans = self.trans.inverted()
        assert self.trans.dist == inv_trans.dist
        assert self.trans.factor == inv_trans.factor
        assert self.trans.nonpos == inv_trans.nonpos
