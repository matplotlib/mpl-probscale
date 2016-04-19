import matplotlib
matplotlib.use('agg')

import nose.tools as nt

from probscale import probscale


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
        nt.assert_equal(self.trans.dist, probscale._minimal_norm)

    def test_transform_non_affine(self):
        nt.assert_true(hasattr(self.trans, 'transform_non_affine'))
        nt.assert_almost_equal(self.trans.transform_non_affine(0.5), self.known_tras_na)

    def test_inverted(self):
        nt.assert_true(hasattr(self.trans, 'inverted'))


class Test_ProbTransform(Mixin_Transform):
    def setup(self):
        self.trans = probscale.ProbTransform(probscale._minimal_norm)
        self.known_tras_na = -2.569150498


class Test_InvertedProbTransform(Mixin_Transform):
    def setup(self):
        self.trans = probscale.InvertedProbTransform(probscale._minimal_norm)
        self.known_tras_na = 69.1464492
