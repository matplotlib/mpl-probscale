import sys
PYTHON27 = sys.version_info.major == 2

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from matplotlib.testing.decorators import image_comparison, cleanup
import nose.tools as nt
import numpy.testing as nptest

from probscale import probscale


class Test__minimal_norm(object):
    def setup(self):
        self.mn = probscale._minimal_norm()
        self.known__A = 0.1400122886866665

        self.input = np.array([
            0.331, 0.742, 0.067, 0.826, 0.357, 0.089,
            0.754, 0.342, 0.762, 0.658, 0.239, 0.910,
        ])

        self.known_erf = np.array([
            0.36029027,  0.70598131,  0.07548843,  0.75724986,
            0.38635283,  0.10016122,  0.71371964,  0.37137355,
            0.71880142,  0.64791492,  0.26463458,  0.80188283,
        ])

        self.known_ppf = np.array([
            -0.43715354,  0.6495236 , -1.49851307,  0.93847570,
            -0.36648929, -1.34693863,  0.68713129, -0.40701088,
             0.71275076,  0.40701088, -0.70952297,  1.34075503,
        ])

        self.known_cdf = np.array([
            0.62967776, 0.77095633, 0.52670915,  0.79559795,
            0.63945410, 0.53545904, 0.77457539,  0.63382455,
            0.77697000, 0.74473093, 0.59444721,  0.81858875
        ])

    def test__A(self):
        nt.assert_true(hasattr(self.mn, '_A'))
        nt.assert_almost_equal(self.mn._A, self.known__A)

    def test__approx_erf(self):
        nptest.assert_array_almost_equal(
            self.mn._approx_erf(self.input),
            self.known_erf,
            decimal=3
        )

    def test__approx_inv_erf(self):
        nptest.assert_array_almost_equal(
            self.input,
            self.mn._approx_inv_erf(self.mn._approx_erf(self.input)),
            decimal=3
        )

    def test_ppf(self):
        nptest.assert_array_almost_equal(
            self.mn.ppf(self.input),
            self.known_ppf,
            decimal=3
        )

    def test_cdf(self):
        nptest.assert_array_almost_equal(
            self.mn.cdf(self.input),
            self.known_cdf,
            decimal=3
        )


class Mixin_Check_Formatter_sig_figs(object):
    def teardown(self):
        pass

    def test_baseline(self):
        nt.assert_equal(self.fmt._sig_figs(self.x, 3), self.known_3)
        nt.assert_equal(self.fmt._sig_figs(self.x, 4), self.known_4)

    def test_string(self):
        nt.assert_equal(self.fmt._sig_figs('1.23', 3), '1.23')

    def test_na_inf(self):
        nt.assert_equal(self.fmt._sig_figs(np.nan, 3), 'NA')
        nt.assert_equal(self.fmt._sig_figs(np.inf, 3), 'NA')

    def test_zero(self):
        nt.assert_equal(self.fmt._sig_figs(0, 3), '0')

    def test_trailing_zeros(self):
        nt.assert_equal(self.fmt._sig_figs(self.x, 8), self.known_8)

    @nptest.raises(ValueError)
    def test_sigFigs_zero_n(self):
        self.fmt._sig_figs(self.x, 0)

    @nptest.raises(ValueError)
    def test_sigFigs_negative_n(self):
        self.fmt._sig_figs(self.x, -1)

    def test_forceint(self):
        nt.assert_equal(
            self.fmt._sig_figs(self.x, 3, forceint=True),
            self.known_int
        )


class Mixin_Check_PctFormatter_sig_figs(Mixin_Check_Formatter_sig_figs):
    fmt = probscale.PctFormatter()
    def test__call__(self):
        nt.assert_equal(self.fmt(0.0301), '0.03')
        nt.assert_equal(self.fmt(0.2), '0.2')
        nt.assert_equal(self.fmt(0.1), '0.1')
        nt.assert_equal(self.fmt(10), '10')
        nt.assert_equal(self.fmt(5), '5')
        nt.assert_equal(self.fmt(50), '50')
        nt.assert_equal(self.fmt(99), '99')
        nt.assert_equal(self.fmt(99.1), '99.1')
        nt.assert_equal(self.fmt(99.99), '99.99')


class Mixin_Check_ProbFormatter_sig_figs(Mixin_Check_Formatter_sig_figs):
    fmt = probscale.ProbFormatter()
    def test__call__(self):
        nt.assert_equal(self.fmt(0.000301), '0.0003')
        nt.assert_equal(self.fmt(0.001), '0.001')
        nt.assert_equal(self.fmt(0.10), '0.10')
        nt.assert_equal(self.fmt(0.05), '0.05')
        nt.assert_equal(self.fmt(0.50), '0.50')
        nt.assert_equal(self.fmt(0.99), '0.99')
        nt.assert_equal(self.fmt(0.991), '0.991')
        nt.assert_equal(self.fmt(0.9999), '0.9999')


class Test_PctFormatter_sig_figs_gt1(Mixin_Check_PctFormatter_sig_figs):
    def setup(self):
        self.x = 1234.56
        self.known_3 = '1,230'
        self.known_4 = '1,235'
        self.known_8 = '1,234.5600'
        self.known_exp3 = '1.23e+08'
        self.known_int = '1,235'
        self.factor = 10**5


class Test_PctFormatter_sig_figs_lt1(Mixin_Check_PctFormatter_sig_figs):
    def setup(self):
        self.x = 0.123456
        self.known_3 = '0.123'
        self.known_4 = '0.1235'
        self.known_8 = '0.12345600'
        self.known_exp3 = '1.23e-07'
        self.known_int = '0'
        self.factor = 10**-6


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


class Test_QuantileTransform(Mixin_Transform):
    def setup(self):
        self.trans = probscale.QuantileTransform(probscale._minimal_norm)
        self.known_tras_na = 69.1464492


@image_comparison(baseline_images=['test_the_scale_default'], extensions=['png'])
@nptest.dec.skipif(PYTHON27)
def test_the_scale_default():
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_yscale('prob')
    ax.set_ylim(0.01, 99.99)
    fig.tight_layout()


@image_comparison(baseline_images=['test_the_scale_not_as_pct'], extensions=['png'])
def test_the_scale_not_as_pct():
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_yscale('prob', as_pct=False)
    ax.set_ylim(0.02, 0.98)


@image_comparison(baseline_images=['test_the_scale_beta'], extensions=['png'])
def test_the_scale_beta():
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_yscale('prob', as_pct=True, dist=stats.beta(3, 2))
    ax.set_ylim(1, 99)
    fig.tight_layout()


class Test_ProbFormatter_sig_figs(Mixin_Check_ProbFormatter_sig_figs):
    def setup(self):
        self.x = 0.123456
        self.known_3 = '0.123'
        self.known_4 = '0.1235'
        self.known_8 = '0.12345600'
        self.known_exp3 = '1.23e-07'
        self.known_int = '0'
        self.factor = 10**-6
