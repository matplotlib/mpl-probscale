import sys
PYTHON27 = sys.version_info.major == 2

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from matplotlib.testing.decorators import image_comparison, cleanup
import nose.tools as nt
import numpy.testing as nptest

import probscale
from probscale.probscale import _minimal_norm


class Test__minimal_norm(object):
    def setup(self):
        self.mn = _minimal_norm()
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
