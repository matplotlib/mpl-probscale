import numpy
import matplotlib
matplotlib.use('agg')

import nose.tools as nt
import numpy.testing as nptest

from probscale.probscale import _minimal_norm
from probscale import transforms


def test__mask_out_of_bounds():
    x = [-0.1, 0, 0.1, 0.5, 0.9, 1.0, 1.1]
    known = numpy.array([numpy.nan, numpy.nan, 0.1, 0.5, 0.9, numpy.nan, numpy.nan])
    result = transforms._mask_out_of_bounds(x)
    nptest.assert_array_almost_equal(result, known)


def test__clip_out_of_bounds():
    x = [-0.1, 0, 0.1, 0.5, 0.9, 1.0, 1.1]
    known = numpy.array([0.0, 0.0, 0.1, 0.5, 0.9, 1.0, 1.0])
    result = transforms._clip_out_of_bounds(x)
    nptest.assert_array_almost_equal(result, known)


class Mixin_Transform(object):
    known_input_dims = 1
    known_output_dims = 1
    known_is_separable = True
    known_has_inverse = True

    def test_input_dims(self):
        nt.assert_true(hasattr(self.trans, 'input_dims'))
        nt.assert_equal(self.trans.input_dims, self.known_input_dims)

    def test_output_dims(self):
        nt.assert_true(hasattr(self.trans, 'output_dims'))
        nt.assert_equal(self.trans.output_dims, self.known_output_dims)

    def test_is_separable(self):
        nt.assert_true(hasattr(self.trans, 'is_separable'))
        nt.assert_equal(self.trans.is_separable, self.known_is_separable)

    def test_has_inverse(self):
        nt.assert_true(hasattr(self.trans, 'has_inverse'))
        nt.assert_equal(self.trans.has_inverse, self.known_has_inverse)

    def test_dist(self):
        nt.assert_true(hasattr(self.trans, 'dist'))
        nt.assert_equal(self.trans.dist, _minimal_norm)

    def test_transform_non_affine(self):
        nt.assert_true(hasattr(self.trans, 'transform_non_affine'))
        nptest.assert_almost_equal(self.trans.transform_non_affine([0.5]), self.known_tras_na)

    def test_inverted(self):
        nt.assert_true(hasattr(self.trans, 'inverted'))

    @nt.raises(ValueError)
    def test_bad_non_pos(self):
        self._trans(_minimal_norm, nonpos='junk')

    # def test_non_pos_default(self):
    #     x = [-0.1, 0, 0.1, 0.5, 0.99, 1, 1.1]

    def test_non_pos_clip(self):
        self._trans(_minimal_norm, nonpos='clip')


class Test_ProbTransform(Mixin_Transform):
    def setup(self):
        self._trans = transforms.ProbTransform
        self.trans = transforms.ProbTransform(_minimal_norm)
        self.known_tras_na = [-2.569150498]

    def test_inverted(self):
        inv_trans = self.trans.inverted()
        nt.assert_equal(self.trans.dist, inv_trans.dist)
        nt.assert_equal(self.trans.factor, inv_trans.factor)
        nt.assert_equal(self.trans.nonpos, inv_trans.nonpos)


class Test_QuantileTransform(Mixin_Transform):
    def setup(self):
        self._trans = transforms.QuantileTransform
        self.trans = transforms.QuantileTransform(_minimal_norm)
        self.known_tras_na = [69.1464492]

    def test_inverted(self):
        inv_trans = self.trans.inverted()
        nt.assert_equal(self.trans.dist, inv_trans.dist)
        nt.assert_equal(self.trans.factor, inv_trans.factor)
        nt.assert_equal(self.trans.nonpos, inv_trans.nonpos)
