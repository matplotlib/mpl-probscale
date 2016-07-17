import sys
from functools import wraps

import numpy
import matplotlib.pyplot as plt

PY27 = sys.version_info.major == 2
if PY27:  # pragma: no cover
    import mock
    TOLERANCE = 15
else:
    from unittest import mock
    TOLERANCE = 12
import pytest
import numpy.testing as nptest

try:
    from scipy import stats
except:  # pragma: no cover
    stats = None

from probscale import viz
from probscale.probscale import _minimal_norm


BASELINE_DIR = 'baseline_images/test_viz'


def seed(func):
    """ Decorator to seed the RNG before any function. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        numpy.random.seed(0)
        return func(*args, **kwargs)
    return wrapper


class Test_fit_line(object):
    def setup(self):
        self.data = numpy.array([
            2.00,   4.0 ,   4.62,   5.00,   5.00,   5.50,   5.57,   5.66,
            5.75,   5.86,   6.65,   6.78,   6.79,   7.50,   7.50,   7.50,
            8.63,   8.71,   8.99,   9.50,   9.50,   9.85,  10.82,  11.00,
           11.25,  11.25,  12.20,  14.92,  16.77,  17.81,  19.16,  19.19,
           19.64,  20.18,  22.97
        ])

        self.zscores = numpy.array([
            -2.06188401, -1.66883254, -1.4335397 , -1.25837339, -1.11509471,
            -0.99166098, -0.8817426 , -0.78156696, -0.68868392, -0.60139747,
            -0.51847288, -0.4389725 , -0.36215721, -0.28742406, -0.21426459,
            -0.14223572, -0.07093824,  0.00000000,  0.07093824,  0.14223572,
             0.21426459,  0.28742406,  0.36215721,  0.43897250,  0.51847288,
             0.60139747,  0.68868392,  0.78156696,  0.88174260,  0.99166098,
             1.11509471,  1.25837339,  1.43353970,  1.66883254,  2.06188401
        ])

        self.probs = _minimal_norm.cdf(self.zscores) * 100.

        self.y = numpy.array([
            0.07323274,  0.12319301,  0.16771455,  0.1779695 ,  0.21840761,
            0.25757016,  0.2740265 ,  0.40868106,  0.44872637,  0.5367353 ,
            0.55169933,  0.56211726,  0.62375442,  0.66631353,  0.68454978,
            0.72137134,  0.87602096,  0.94651962,  1.01927875,  1.06040448,
            1.07966792,  1.17969506,  1.21132273,  1.30751428,  1.45371899,
            1.76381932,  1.98832275,  2.09275652,  2.66552831,  2.86453334,
            3.23039631,  4.23953492,  4.25892247,  4.5834766 ,  6.53100725
        ])

        self.known_y_linlin_no_ci = numpy.array([-0.896506, 21.12622])
        self.known_y_linlin = numpy.array([-0.8965, 6.4370, 9.7360, 12.8837, 17.7706])
        self.known_y_linlog = numpy.array([2.8019, 6.0052, 8.4619, 11.7375, 19.5072])
        self.known_y_linprob = numpy.array([8.4762, 23.0079, 40.0813, 57.6156, 94.6629])
        self.known_y_loglin = numpy.array([-2.576205, -0.7402 , -0.034269, 0.426663, 1.395386])
        self.known_y_loglog = numpy.array([0.0468154, 0.37470676, 0.83369069, 1.40533704, 4.21100704])
        self.known_y_logprob = numpy.array([0.48982206, 22.957763, 48.63313552, 66.518853, 91.86591714])
        self.known_y_problin = numpy.array([-0.89650596, 6.43698357, 9.73601589, 12.88372926, 17.77058661])
        self.known_y_problog = numpy.array([2.80190754, 6.00524156, 8.46190468, 11.73746612, 19.50723532])
        self.known_y_probprob = numpy.array([2.106935, 24.925853, 47.268638, 69.562842, 92.127085])

        self.custom_xhat = [-2, -1, 0, 1, 2]
        self.known_custom_yhat = numpy.array([-0.56601826, 4.77441944, 10.11485714,
                                           15.45529485, 20.79573255])

    def check_res(self, res, known_res):
        assert abs(res['intercept'] - known_res['intercept']) < 0.000001
        assert abs(res['slope'] - known_res['slope']) < 0.000001
        if known_res['yhat_lo'] is None:
            assert res['yhat_hi'] is None
            assert res['yhat_lo'] is None
        else:
            nptest.assert_allclose(res['yhat_lo'], known_res['yhat_lo'], rtol=0.0001)
            nptest.assert_allclose(res['yhat_hi'], known_res['yhat_hi'], rtol=0.0001)

    @seed
    def test_xlinear_ylinear_no_ci(self):
        known_y_linlin_no_ci = numpy.array([
            -0.89650596,   1.20256093,   2.45912768,   3.39459245,
             4.15976331,   4.81895346,   5.40596572,   5.94094748,
             6.43698357,   6.90313142,   7.34598503,   7.77055185,
             8.18077912,   8.57988686,   8.97059045,   9.35525614,
             9.73601589,  10.11485714,  10.49369839,  10.87445814,
            11.25912384,  11.64982743,  12.04893516,  12.45916243,
            12.88372926,  13.32658287,  13.79273071,  14.2887668 ,
            14.82374857,  15.41076083,  16.06995097,  16.83512184,
            17.77058661,  19.02715336,  21.12622025
        ])
        scales = {'fitlogs': None, 'fitprobs': None}
        x, y = self.zscores, self.data
        x_, y_, res = viz.fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, known_y_linlin_no_ci)
        known_res = {
            'slope': 5.3404377026700995,
            'intercept': 10.114857142857147,
            'yhat_lo': None,
            'yhat_hi': None,
        }
        self.check_res(res, known_res)

    @seed
    def test_xlinear_ylinear(self):
        scales = {'fitlogs': None, 'fitprobs': None}
        x, y = self.zscores, self.data
        x_, y_, res = viz.fit_line(x, y, xhat=x[::8], estimate_ci=True, **scales)
        nptest.assert_allclose(y_, self.known_y_linlin, rtol=0.0001)
        known_res = {
            'slope': 5.3404377026700995,
            'intercept': 10.114857142857147,
            'yhat_lo': numpy.array([ -2.9223,   5.4807,   9.109 ,  12.0198,  16.2376]),
            'yhat_hi': numpy.array([  0.4983,   7.0448,  10.2715,  13.4877,  18.8306]),
        }
        self.check_res(res, known_res)

    @seed
    def test_xlinear_ylog(self):
        scales = {'fitlogs': 'y', 'fitprobs': None}
        x, y = self.zscores, self.data
        x_, y_, res = viz.fit_line(x, y, xhat=x[::8], estimate_ci=True, **scales)
        nptest.assert_allclose(y_, self.known_y_linlog, rtol=0.0001)
        known_res = {
            'slope': 0.55515014824534514,
            'intercept': 2.1749556618678434,
            'yhat_lo': numpy.array([  2.4355,   5.6436,   8.1653,  11.3136,  18.1000]),
            'yhat_hi': numpy.array([  3.1348,   6.3072,   8.7495,  12.2324,  21.2824]),
        }
        self.check_res(res, known_res)

    @seed
    def test_xlinear_yprob(self):
        scales = {'fitlogs': None, 'fitprobs': 'y'}
        x, y = self.data, self.probs
        x_, y_, res = viz.fit_line(x, y, xhat=x[::8], estimate_ci=True, **scales)
        nptest.assert_allclose(y_, self.known_y_linprob, rtol=0.0001)
        known_res = {
            'slope': 0.16920340891421964,
            'intercept': -1.7114683092517717,
            'yhat_lo': numpy.array([  5.6382,  18.9842,  36.0326,  54.0282,  92.8391]),
            'yhat_hi': numpy.array([ 12.6284,  28.2687,  44.6934,  61.8816,  97.1297]),
        }
        self.check_res(res, known_res)

    @seed
    def test_xlog_ylinear(self):
        scales = {'fitlogs': 'x', 'fitprobs': None}
        x, y = self.data, self.zscores
        x_, y_, res = viz.fit_line(x, y, xhat=x[::8], estimate_ci=True, **scales)
        nptest.assert_allclose(y_, self.known_y_loglin, rtol=0.0001)
        known_res = {
            'slope': 1.7385543724819053,
            'intercept': -3.7812786758946122,
            'yhat_lo': numpy.array([-2.88948 , -0.846565, -0.093696,  0.360738,  1.255963]),
            'yhat_hi': numpy.array([-2.310246, -0.63795 ,  0.024143,  0.494404,  1.561183]),
        }
        self.check_res(res, known_res)

    @seed
    def test_xlog_ylog(self):
        scales = {'fitlogs': 'both', 'fitprobs': None}
        x, y = self.data, self.y
        x_, y_, res = viz.fit_line(x, y, xhat=x[::8], estimate_ci=True, **scales)
        nptest.assert_allclose(y_, self.known_y_loglog, rtol=0.0001)
        known_res = {
            'slope': 1.9695339470891058,
            'intercept': -4.4267200322534261,
            'yhat_lo': numpy.array([ 0.033559,  0.32797 ,  0.777473,  1.331504,  3.811647]),
            'yhat_hi': numpy.array([ 0.061867,  0.422956,  0.892383,  1.48953 ,  4.842235]),
        }
        self.check_res(res, known_res)

    @seed
    def test_xlog_yprob(self):
        scales = {'fitlogs': 'x', 'fitprobs': 'y'}
        x, y = self.data, self.probs
        x_, y_, res = viz.fit_line(x, y, xhat=x[::8], estimate_ci=True, **scales)
        nptest.assert_allclose(y_, self.known_y_logprob, rtol=0.0001)
        known_res = {
            'slope': 1.7385543724819046,
            'intercept': -3.7812786758946113,
            'yhat_lo': numpy.array([0.187555, 19.859832, 46.267537, 64.085292, 89.551801]),
            'yhat_hi': numpy.array([1.030230, 26.174702, 50.963065, 68.949137, 94.089655]),
        }
        self.check_res(res, known_res)

    @seed
    def test_xprob_ylinear(self):
        scales = {'fitlogs': None, 'fitprobs': 'x'}
        x, y = self.probs, self.data
        x_, y_, res = viz.fit_line(x, y, xhat=x[::8], estimate_ci=True, **scales)
        nptest.assert_allclose(y_, self.known_y_problin, rtol=0.0001)
        known_res = {
            'slope': 5.3404377026700995,
            'intercept': 10.114857142857147,
            'yhat_lo': numpy.array([-2.92233134, 5.48065673,  9.1090198 , 12.01977856, 16.23762957]),
            'yhat_hi': numpy.array([ 0.49826723, 7.04480065, 10.27146083, 13.48770383, 18.83061329]),
        }
        self.check_res(res, known_res)

    @seed
    def test_xprob_ylog(self):
        scales = {'fitlogs': 'y', 'fitprobs': 'x'}
        x, y = self.probs, self.data
        x_, y_, res = viz.fit_line(x, y, xhat=x[::8], estimate_ci=True, **scales)
        nptest.assert_allclose(y_, self.known_y_problog, rtol=0.0001)
        known_res = {
            'intercept': 2.1749556618678434,
            'slope': 0.55515014824534525,
            'yhat_lo': numpy.array([2.43550106, 5.6436203 , 8.16525601, 11.31358231, 18.09998664]),
            'yhat_hi': numpy.array([3.13484803, 6.30722509, 8.74945323, 12.23244498, 21.28240831]),
        }
        self.check_res(res, known_res)

    @seed
    def test_xprob_yprob(self):
        p2 = self.probs + numpy.random.uniform(-1, 1, size=len(self.probs))

        scales = {'fitlogs': None, 'fitprobs': 'both'}
        x, y = self.probs, p2,
        x_, y_, res = viz.fit_line(x, y, xhat=x[::8], estimate_ci=True, **scales)
        nptest.assert_allclose(y_, self.known_y_probprob, rtol=0.0001)
        known_res = {
            'slope': 0.98467862838225351,
            'intercept': 0.0013327049076583583,
            'yhat_lo': numpy.array([1.96759603, 24.66922946, 46.88723664, 68.88913508, 91.58436332]),
            'yhat_hi': numpy.array([2.28593917, 25.24921351, 47.60781632, 70.11543855, 92.54803847]),
        }
        self.check_res(res, known_res)

    def test_bad_fitlogs(self):
        with pytest.raises(ValueError):
            x, y = self.zscores, self.data
            x_, y_, res = viz.fit_line(x, y, fitlogs='junk')

    def test_bad_fitprobs(self):
        with pytest.raises(ValueError):
            x, y = self.zscores, self.data
            x_, y_, res = viz.fit_line(x, y, fitprobs='junk')

    def test_custom_xhat(self):
        x, y = self.zscores, self.data
        x_, y_, res = viz.fit_line(x, y, xhat=self.custom_xhat)
        nptest.assert_array_almost_equal(y_, self.known_custom_yhat)


class Test__estimate_from_fit(object):
    def setup(self):
        self.x = numpy.arange(1, 11, 0.5)
        self.slope = 2
        self.intercept = 3.5

        self.known_ylinlin = numpy.array([
             5.5,   6.5,   7.5,   8.5,   9.5,  10.5,  11.5,  12.5,  13.5,
            14.5,  15.5,  16.5,  17.5,  18.5,  19.5,  20.5,  21.5,  22.5,
            23.5,  24.5
        ])


        self.known_yloglin = numpy.array([
            3.5       ,  4.31093022,  4.88629436,  5.33258146,  5.69722458,
            6.00552594,  6.27258872,  6.50815479,  6.71887582,  6.90949618,
            7.08351894,  7.24360435,  7.3918203 ,  7.52980604,  7.65888308,
            7.78013233,  7.89444915,  8.0025836 ,  8.10517019,  8.20275051
        ])

        self.known_yloglog = numpy.array([
              33.11545196,    74.50976691,   132.46180783,   206.97157474,
             298.03906763,   405.66428649,   529.84723134,   670.58790216,
             827.88629897,  1001.74242175,  1192.15627051,  1399.12784525,
            1622.65714598,  1862.74417268,  2119.38892536,  2392.59140402,
            2682.35160865,  2988.66953927,  3311.54519587,  3650.97857845
        ])

        self.known_ylinlog = numpy.array([
             2.44691932e+02,   6.65141633e+02,   1.80804241e+03,
             4.91476884e+03,   1.33597268e+04,   3.63155027e+04,
             9.87157710e+04,   2.68337287e+05,   7.29416370e+05,
             1.98275926e+06,   5.38969848e+06,   1.46507194e+07,
             3.98247844e+07,   1.08254988e+08,   2.94267566e+08,
             7.99902177e+08,   2.17435955e+09,   5.91052206e+09,
             1.60664647e+10,   4.36731791e+10
         ])

    def test_linlin(self):
        ylinlin = viz._estimate_from_fit(self.x, self.slope, self.intercept,
                                              xlog=False, ylog=False)
        nptest.assert_array_almost_equal(ylinlin, self.known_ylinlin)

    def test_loglin(self):
        yloglin = viz._estimate_from_fit(self.x, self.slope, self.intercept,
                                              xlog=True, ylog=False)
        nptest.assert_array_almost_equal(yloglin, self.known_yloglin)

    def test_loglog(self):
        yloglog = viz._estimate_from_fit(self.x, self.slope, self.intercept,
                                              xlog=True, ylog=True)
        nptest.assert_array_almost_equal(yloglog, self.known_yloglog)

    def test_linlog(self):
        ylinlog = viz._estimate_from_fit(self.x, self.slope, self.intercept,
                                              xlog=False, ylog=True)
        percent_diff = numpy.abs(ylinlog - self.known_ylinlog) / self.known_ylinlog
        nptest.assert_array_almost_equal(
            percent_diff,
            numpy.zeros(self.x.shape[0]),
            decimal=5
        )


class Test_plot_pos(object):
    def setup(self):
        self.data = numpy.arange(16)

        self.known_type4 = numpy.array([
            0.0625,  0.125 ,  0.1875,  0.25  ,  0.3125,  0.375 ,  0.4375,
            0.5   ,  0.5625,  0.625 ,  0.6875,  0.75  ,  0.8125,  0.875 ,
            0.9375,  1.
        ])

        self.known_type5 = numpy.array([
            0.03125,  0.09375,  0.15625,  0.21875,  0.28125,  0.34375,
            0.40625,  0.46875,  0.53125,  0.59375,  0.65625,  0.71875,
            0.78125,  0.84375,  0.90625,  0.96875
        ])

        self.known_type6 = numpy.array([
            0.05882353,  0.11764706,  0.17647059,  0.23529412,  0.29411765,
            0.35294118,  0.41176471,  0.47058824,  0.52941176,  0.58823529,
            0.64705882,  0.70588235,  0.76470588,  0.82352941,  0.88235294,
            0.94117647
        ])

        self.known_type7 = numpy.array([
            0.        ,  0.06666667,  0.13333333,  0.2       ,  0.26666667,
            0.33333333,  0.4       ,  0.46666667,  0.53333333,  0.6       ,
            0.66666667,  0.73333333,  0.8       ,  0.86666667,  0.93333333,
            1.
        ])

        self.known_type8 = numpy.array([
            0.04081633,  0.10204082,  0.16326531,  0.2244898 ,  0.28571429,
            0.34693878,  0.40816327,  0.46938776,  0.53061224,  0.59183673,
            0.65306122,  0.71428571,  0.7755102 ,  0.83673469,  0.89795918,
            0.95918367
        ])

        self.known_type9 = numpy.array([
            0.03846154,  0.1       ,  0.16153846,  0.22307692,  0.28461538,
            0.34615385,  0.40769231,  0.46923077,  0.53076923,  0.59230769,
            0.65384615,  0.71538462,  0.77692308,  0.83846154,  0.9       ,
            0.96153846
        ])

        self.known_weibull = self.known_type6

        self.known_median = numpy.array([
            0.04170486,  0.10281088,  0.1639169 ,  0.22502291,  0.28612893,
            0.34723495,  0.40834097,  0.46944699,  0.53055301,  0.59165903,
            0.65276505,  0.71387107,  0.77497709,  0.8360831 ,  0.89718912,
            0.95829514
        ])

        self.known_apl = numpy.array([
            0.0398773 ,  0.10122699,  0.16257669,  0.22392638,  0.28527607,
            0.34662577,  0.40797546,  0.46932515,  0.53067485,  0.59202454,
            0.65337423,  0.71472393,  0.77607362,  0.83742331,  0.89877301,
            0.9601227
        ])

        self.known_pwm = self.known_apl

        self.known_blom = self.known_type9

        self.known_hazen = self.known_type5

        self.known_cunnane = numpy.array([
            0.03703704,  0.09876543,  0.16049383,  0.22222222,  0.28395062,
            0.34567901,  0.40740741,  0.4691358 ,  0.5308642 ,  0.59259259,
            0.65432099,  0.71604938,  0.77777778,  0.83950617,  0.90123457,
            0.96296296
        ])

        self.known_gringorten = numpy.array([
            0.03473945,  0.09677419,  0.15880893,  0.22084367,  0.28287841,
            0.34491315,  0.40694789,  0.46898263,  0.53101737,  0.59305211,
            0.65508685,  0.71712159,  0.77915633,  0.84119107,  0.90322581,
            0.96526055
        ])

    def test_type4(self):
        pp, yy = viz.plot_pos(self.data, postype='type 4')
        nptest.assert_array_almost_equal(pp, self.known_type4)

    def test_type5(self):
        pp, yy = viz.plot_pos(self.data, postype='type 5')
        nptest.assert_array_almost_equal(pp, self.known_type5)

    def test_type6(self):
        pp, yy = viz.plot_pos(self.data, postype='type 6')
        nptest.assert_array_almost_equal(pp, self.known_type6)

    def test_type7(self):
        pp, yy = viz.plot_pos(self.data, postype='type 7')
        nptest.assert_array_almost_equal(pp, self.known_type7)

    def test_type8(self):
        pp, yy = viz.plot_pos(self.data, postype='type 8')
        nptest.assert_array_almost_equal(pp, self.known_type8)

    def test_type9(self):
        pp, yy = viz.plot_pos(self.data, postype='type 9')
        nptest.assert_array_almost_equal(pp, self.known_type9)

    def test_weibull(self):
        pp, yy = viz.plot_pos(self.data, postype='weibull')
        nptest.assert_array_almost_equal(pp, self.known_weibull)

    def test_median(self):
        pp, yy = viz.plot_pos(self.data, postype='median')
        nptest.assert_array_almost_equal(pp, self.known_median)

    def test_apl(self):
        pp, yy = viz.plot_pos(self.data, postype='apl')
        nptest.assert_array_almost_equal(pp, self.known_apl)

    def test_pwm(self):
        pp, yy = viz.plot_pos(self.data, postype='pwm')
        nptest.assert_array_almost_equal(pp, self.known_pwm)

    def test_blom(self):
        pp, yy = viz.plot_pos(self.data, postype='blom')
        nptest.assert_array_almost_equal(pp, self.known_blom)

    def test_hazen(self):
        pp, yy = viz.plot_pos(self.data, postype='hazen')
        nptest.assert_array_almost_equal(pp, self.known_hazen)

    def test_cunnane(self):
        pp, yy = viz.plot_pos(self.data, postype='cunnane')
        nptest.assert_array_almost_equal(pp, self.known_cunnane)

    def test_gringorten(self):
        pp, yy = viz.plot_pos(self.data, postype='gringorten')
        nptest.assert_array_almost_equal(pp, self.known_gringorten)

    def test_bad_postype(self):
        with pytest.raises(KeyError):
            viz.plot_pos(self.data, postype='junk')


def test__make_boot_index():
    result = viz._make_boot_index(5, 5000)
    assert result.shape == (5000, 5)
    assert result.min() == 0
    assert result.max() == 4


@pytest.fixture
def plot_data():
    data = numpy.array([
         3.113,   3.606,   4.046,   4.046,   4.710,   6.140,   6.978,
         2.000,   4.200,   4.620,   5.570,   5.660,   5.860,   6.650,
         6.780,   6.790,   7.500,   7.500,   7.500,   8.630,   8.710,
         8.990,   9.850,  10.820,  11.250,  11.250,  12.200,  14.920,
        16.770,  17.810,  19.160,  19.190,  19.640,  20.180,  22.970,
    ])
    return data


@pytest.mark.parametrize(('fitlogs', 'known_yhat'), [
    (None, numpy.array([0.7887, 3.8946, 7.0005, 10.1065, 13.2124, 16.3183])),
    ('x', numpy.array([0.2711, 1.2784, 1.5988, 1.7953, 1.9373, 2.0487])),
    ('y', numpy.array([2.2006e+00, 4.9139e+01, 1.0972e+03, 2.4501e+04, 5.4711e+05, 1.2217e+07])),
    ('both', numpy.array([1.3114, 3.5908, 4.9472, 6.0211, 6.9402, 7.7577])),
])
def test__fit_simple(plot_data, fitlogs, known_yhat):
    x = numpy.arange(1, len(plot_data)+1)
    known_results = {'slope': 0.5177, 'intercept': 0.2711}
    xhat = x[::6]
    yhat, results = viz._fit_simple(x, plot_data, xhat, fitlogs=fitlogs)
    assert abs(results['intercept'] - known_results['intercept']) < 0.0001
    assert abs(results['slope'] - known_results['slope']) < 0.0001
    nptest.assert_allclose(yhat, known_yhat, rtol=0.0001)


@pytest.mark.parametrize(('fitlogs', 'known_lo', 'known_hi'), [
    (None, numpy.array([-0.7944, 2.7051, 6.1974,  9.2612, 11.9382, 14.4290]),
           numpy.array([ 2.1447, 4.8360, 7.7140, 10.8646, 14.1014, 17.4432])),
    ('x', numpy.array([-1.4098, -0.2210, 0.1387, 0.3585, 0.5147, 0.6417]),
          numpy.array([ 1.7067,  2.5661, 2.8468, 3.0169, 3.1400, 3.2341])),
    ('y', numpy.array([4.5187e-01, 1.4956e+01, 4.9145e+02, 1.0522e+04, 1.5299e+05, 1.8468e+06]),
          numpy.array([8.5396e+00, 1.2596e+02, 2.2396e+03, 5.2290e+04, 1.3310e+06, 3.7627e+07])),
    ('both', numpy.array([0.2442,  0.8017,  1.1488,  1.4312,  1.6731,  1.8997]),
             numpy.array([5.5107, 13.0148 , 17.232, 20.4285, 23.1035, 25.3843])),
])
def test__fit_ci(plot_data, fitlogs, known_lo, known_hi):
    numpy.random.seed(0)
    x = numpy.arange(1, len(plot_data)+1)
    xhat = x[::6]
    yhat_lo, yhat_hi = viz._fit_ci(x, plot_data, xhat, fitlogs=fitlogs, niter=1000)

    nptest.assert_allclose(yhat_lo, known_lo, rtol=0.001)
    nptest.assert_allclose(yhat_hi, known_hi, rtol=0.001)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_prob(plot_data):
    fig, ax = plt.subplots()
    fig = viz.probplot(plot_data, ax=ax, problabel='Test xlabel', datascale='log')
    assert isinstance(fig, plt.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_qq(plot_data):
    fig, ax = plt.subplots()
    fig = viz.probplot(plot_data, ax=ax, plottype='qq', datalabel='Test label',
                       datascale='log', scatter_kws=dict(color='r'))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@pytest.mark.skipif(stats is None, reason="no scipy")
def test_probplot_qq_dist(plot_data):
    fig, ax = plt.subplots()
    norm = stats.norm(*stats.norm.fit(plot_data))
    fig = viz.probplot(plot_data, ax=ax, plottype='qq', dist=norm,
                       datalabel='Test label')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_pp(plot_data):
    fig, ax = plt.subplots()
    scatter_kws = dict(color='b', linestyle='--', markeredgecolor='g', markerfacecolor='none')
    fig = viz.probplot(plot_data, ax=ax, plottype='pp', datascale='linear',
                       datalabel='test x', problabel='test y',
                       scatter_kws=scatter_kws)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_prob_bestfit(plot_data):
    fig, ax = plt.subplots()
    fig = viz.probplot(plot_data, ax=ax, datalabel='Test xlabel', bestfit=True,
                       datascale='log', estimate_ci=True)
    assert isinstance(fig, plt.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_qq_bestfit(plot_data):
    fig, ax = plt.subplots()
    fig = viz.probplot(plot_data, ax=ax, plottype='qq', bestfit=True,
                       problabel='Test label', datascale='log',
                       estimate_ci=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_pp_bestfit(plot_data):
    fig, ax = plt.subplots()
    scatter_kws = {'marker': 's', 'color': 'red'}
    line_kws = {'linestyle': '--', 'linewidth': 3}
    fig = viz.probplot(plot_data, ax=ax, plottype='pp', datascale='linear',
                       datalabel='test x', bestfit=True, problabel='test y',
                       scatter_kws=scatter_kws, line_kws=line_kws,
                       estimate_ci=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_prob_probax_y(plot_data):
    fig, ax = plt.subplots()
    fig = viz.probplot(plot_data, ax=ax, datalabel='Test xlabel', datascale='log', probax='y')
    assert isinstance(fig, plt.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_qq_probax_y(plot_data):
    fig, ax = plt.subplots()
    fig = viz.probplot(plot_data, ax=ax, plottype='qq', problabel='Test label', probax='y',
                       datascale='log', scatter_kws=dict(color='r'))
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_pp_probax_y(plot_data):
    fig, ax = plt.subplots()
    scatter_kws = dict(color='b', linestyle='--', markeredgecolor='g', markerfacecolor='none')
    fig = viz.probplot(plot_data, ax=ax, plottype='pp', datascale='linear', probax='y',
                       datalabel='test x', problabel='test y', scatter_kws=scatter_kws)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_prob_bestfit_probax_y(plot_data):
    fig, ax = plt.subplots()
    fig = viz.probplot(plot_data, ax=ax, datalabel='Test xlabel', bestfit=True,
                       datascale='log', probax='y', estimate_ci=True)
    assert isinstance(fig, plt.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_qq_bestfit_probax_y(plot_data):
    fig, ax = plt.subplots()
    fig = viz.probplot(plot_data, ax=ax, plottype='qq', bestfit=True, problabel='Test label',
                       datascale='log', probax='y', estimate_ci=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_pp_bestfit_probax_y(plot_data):
    fig, ax = plt.subplots()
    scatter_kws = {'marker': 's', 'color': 'red'}
    line_kws = {'linestyle': '--', 'linewidth': 3}
    fig = viz.probplot(plot_data, ax=ax, plottype='pp', datascale='linear', probax='y',
                       datalabel='test x', bestfit=True, problabel='test y',
                       scatter_kws=scatter_kws, line_kws=line_kws, estimate_ci=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@pytest.mark.skipif(stats is None, reason="no scipy")
def test_probplot_beta_dist_best_fit_y(plot_data):
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    dist = stats.beta(3, 3)
    fig = viz.probplot(plot_data, dist=dist, ax=ax1, problabel='Beta scale',
                       bestfit=True, datascale='log', probax='y')
    ax1.set_ylim(bottom=0.5, top=98)

    fig = viz.probplot(plot_data, ax=ax2, datalabel='Default (norm)',
                       bestfit=True, datascale='log', probax='y',
                       estimate_ci=True)
    ax2.set_ylim(bottom=0.5, top=98)

    assert isinstance(fig, plt.Figure)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@pytest.mark.skipif(stats is None, reason="no scipy")
def test_probplot_beta_dist_best_fit_x(plot_data):
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    dist = stats.beta(3, 3)
    fig = viz.probplot(plot_data, dist=dist, ax=ax1, problabel='Beta scale',
                       bestfit=True, datascale='log', probax='x')
    ax1.set_xlim(left=0.5, right=98)

    fig = viz.probplot(plot_data, ax=ax2, problabel='Default (norm)',
                       bestfit=True, datascale='log', probax='x',
                       estimate_ci=True)
    ax2.set_xlim(left=0.5, right=98)

    assert isinstance(fig, plt.Figure)
    return fig


def test_probplot_test_results(plot_data):
    fig, ax = plt.subplots()
    fig, results = viz.probplot(plot_data, return_best_fit_results=True)

    assert isinstance(results, dict)
    known_keys = sorted(['q', 'x', 'y', 'xhat', 'yhat', 'res'])
    assert sorted(list(results.keys())) == known_keys
    return fig


@pytest.mark.parametrize('probax', ['x', 'y'])
@pytest.mark.parametrize(('N', 'minval', 'maxval'), [
    (8, 10, 90),
    (37, 1, 99),
    (101, 0.1, 99.9),
    (10001, 0.001, 99.999)
])
def test__set_prob_limits_x(probax, N, minval, maxval):
    from probscale import validate
    ax = mock.Mock()
    with mock.patch.object(validate, 'axes_object', return_value=[None, ax]):
        viz._set_prob_limits(ax, probax, N)
        if probax == 'x':
            ax.set_xlim.assert_called_once_with(left=minval, right=maxval)
        elif probax == 'y':
            ax.set_ylim.assert_called_once_with(bottom=minval, top=maxval)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_probplot_color_and_label(plot_data):
    fig, ax = plt.subplots()
    fig = viz.probplot(plot_data, ax=ax, color='pink', label='A Top-Level Label')
    ax.legend(loc='lower right')
    return fig