import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import nose.tools as nt
import numpy.testing as nptest
from matplotlib.testing.decorators import image_comparison, cleanup

from probscale import viz


@nt.nottest
def setup_plot_data():
    data = np.array([
         3.113,   3.606,   4.046,   4.046,   4.710,   6.140,   6.978,
         2.000,   4.200,   4.620,   5.570,   5.660,   5.860,   6.650,
         6.780,   6.790,   7.500,   7.500,   7.500,   8.630,   8.710,
         8.990,   9.850,  10.820,  11.250,  11.250,  12.200,  14.920,
        16.770,  17.810,  19.160,  19.190,  19.640,  20.180,  22.970,
    ])
    return data


@cleanup
class Test__check_ax_obj(object):
    @nt.raises(ValueError)
    def test_bad_value(self):
        viz._check_ax_obj('junk')

    def test_with_ax(self):
        fig, ax = plt.subplots()
        fig1, ax1 = viz._check_ax_obj(ax)
        nt.assert_true(isinstance(ax1, plt.Axes))
        nt.assert_true(isinstance(fig1, plt.Figure))
        nt.assert_true(ax1 is ax)
        nt.assert_true(fig1 is fig)

    def test_with_None(self):
        fig1, ax1 = viz._check_ax_obj(None)
        nt.assert_true(isinstance(ax1, plt.Axes))
        nt.assert_true(isinstance(fig1, plt.Figure))


class Test__check_fig_arg(object):
    @nt.raises(ValueError)
    def test_bad_value(self):
        viz._check_fit_arg('junk', 'fitprobs')

    def test_x(self):
        nt.assert_equal('x', viz._check_fit_arg('x', 'fitprobs'))
        nt.assert_equal('x', viz._check_fit_arg('x', 'fitlogs'))

    def test_y(self):
        nt.assert_equal('y', viz._check_fit_arg('y', 'fitprobs'))
        nt.assert_equal('y', viz._check_fit_arg('y', 'fitlogs'))

    def test_both(self):
        nt.assert_equal('both', viz._check_fit_arg('both', 'fitprobs'))
        nt.assert_equal('both', viz._check_fit_arg('both', 'fitlogs'))

    def test_None(self):
        nt.assert_true(viz._check_fit_arg(None, 'fitprobs') is None)
        nt.assert_true(viz._check_fit_arg(None, 'fitlogs') is None)


class Test__check_ax_name(object):
    @nt.raises(ValueError)
    def test_bad_value(self):
        viz._check_fit_arg('junk', 'axname')

    def test_x(self):
        nt.assert_equal('x', viz._check_fit_arg('x', 'axname'))

    def test_y(self):
        nt.assert_equal('y', viz._check_fit_arg('y', 'axname'))


class Test__fit_line(object):
    def setup(self):
        self.data = np.array([
            2.00,   4.0 ,   4.62,   5.00,   5.00,   5.50,   5.57,   5.66,
            5.75,   5.86,   6.65,   6.78,   6.79,   7.50,   7.50,   7.50,
            8.63,   8.71,   8.99,   9.50,   9.50,   9.85,  10.82,  11.00,
           11.25,  11.25,  12.20,  14.92,  16.77,  17.81,  19.16,  19.19,
           19.64,  20.18,  22.97
        ])

        self.zscores = np.array([
            -2.06188401, -1.66883254, -1.4335397 , -1.25837339, -1.11509471,
            -0.99166098, -0.8817426 , -0.78156696, -0.68868392, -0.60139747,
            -0.51847288, -0.4389725 , -0.36215721, -0.28742406, -0.21426459,
            -0.14223572, -0.07093824,  0.00000000,  0.07093824,  0.14223572,
             0.21426459,  0.28742406,  0.36215721,  0.43897250,  0.51847288,
             0.60139747,  0.68868392,  0.78156696,  0.88174260,  0.99166098,
             1.11509471,  1.25837339,  1.43353970,  1.66883254,  2.06188401
        ])

        self.probs = stats.norm.cdf(self.zscores) * 100.

        self.y = np.array([
            0.07323274,  0.12319301,  0.16771455,  0.1779695 ,  0.21840761,
            0.25757016,  0.2740265 ,  0.40868106,  0.44872637,  0.5367353 ,
            0.55169933,  0.56211726,  0.62375442,  0.66631353,  0.68454978,
            0.72137134,  0.87602096,  0.94651962,  1.01927875,  1.06040448,
            1.07966792,  1.17969506,  1.21132273,  1.30751428,  1.45371899,
            1.76381932,  1.98832275,  2.09275652,  2.66552831,  2.86453334,
            3.23039631,  4.23953492,  4.25892247,  4.5834766 ,  6.53100725
        ])

        self.known_y_linlin = np.array([-0.896506, 21.12622])
        self.known_y_linlog = np.array([2.801908, 27.649589])
        self.known_y_linprob = np.array([8.491444, 98.528266])
        self.known_y_loglin = np.array([-2.57620461, 1.66767934])
        self.known_y_loglog = np.array([ 0.0468154, 5.73261406])
        self.known_y_logprob = np.array([0.492579, 95.233708])
        self.known_y_problin = np.array([-0.887084, 21.116798])
        self.known_y_problog = np.array([2.804758, 27.621489])
        self.known_y_probprob = np.array([1.96093902, 98.03906098])

        self.custom_xhat = [-2, -1, 0, 1, 2]
        self.known_custom_yhat = np.array([-0.56601826, 4.77441944, 10.11485714,
                                           15.45529485, 20.79573255])

    def test_xlinear_ylinear(self):
        scales = {'fitlogs': None, 'fitprobs': None}
        x, y = self.zscores, self.data
        x_, y_, res = viz._fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_linlin)
        nt.assert_true(isinstance(res, np.ndarray))

    def test_xlinear_ylog(self):
        scales = {'fitlogs': 'y', 'fitprobs': None}
        x, y = self.zscores, self.data
        x_, y_, res = viz._fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_linlog)
        nt.assert_true(isinstance(res, np.ndarray))

    def test_xlinear_yprob(self):
        scales = {'fitlogs': None, 'fitprobs': 'y'}
        x, y = self.data, self.probs
        x_, y_, res = viz._fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_linprob)
        nt.assert_true(isinstance(res, np.ndarray))

    def test_xlog_ylinear(self):
        scales = {'fitlogs': 'x', 'fitprobs': None}
        x, y = self.data, self.zscores
        x_, y_, res = viz._fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_loglin)
        nt.assert_true(isinstance(res, np.ndarray))

    def test_xlog_ylog(self):
        scales = {'fitlogs': 'both', 'fitprobs': None}
        x, y = self.data, self.y
        x_, y_, res = viz._fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_loglog)
        nt.assert_true(isinstance(res, np.ndarray))

    def test_xlog_yprob(self):
        scales = {'fitlogs': 'x', 'fitprobs': 'y'}
        x, y = self.data, self.probs
        x_, y_, res = viz._fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_logprob)
        nt.assert_true(isinstance(res, np.ndarray))

    def test_xprob_ylinear(self):
        scales = {'fitlogs': None, 'fitprobs': 'x'}
        x, y = self.probs, self.data
        x_, y_, res = viz._fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_problin)
        nt.assert_true(isinstance(res, np.ndarray))

    def test_xprob_ylog(self):
        scales = {'fitlogs': 'y', 'fitprobs': 'x'}
        x, y = self.probs, self.data
        x_, y_, res = viz._fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_problog)
        nt.assert_true(isinstance(res, np.ndarray))

    def test_xprob_yprob(self):
        z2, _y = stats.probplot(self.y, fit=False)
        p2 = stats.norm.cdf(z2) * 100

        scales = {'fitlogs': None, 'fitprobs': 'both'}
        x, y = self.probs, p2,
        x_, y_, res = viz._fit_line(x, y, **scales)
        nptest.assert_array_almost_equal(y_, self.known_y_probprob)
        nt.assert_true(isinstance(res, np.ndarray))

    @nt.raises(ValueError)
    def test_bad_fitlogs(self):
        x, y = self.zscores, self.data
        x_, y_, res = viz._fit_line(x, y, fitlogs='junk')

    @nt.raises(ValueError)
    def test_bad_fitprobs(self):
        x, y = self.zscores, self.data
        x_, y_, res = viz._fit_line(x, y, fitprobs='junk')

    def test_custom_xhat(self):
        x, y = self.zscores, self.data
        x_, y_, res = viz._fit_line(x, y, xhat=self.custom_xhat)
        nptest.assert_array_almost_equal(y_, self.known_custom_yhat)


class Test__estimate_from_fit(object):
    def setup(self):
        self.x = np.arange(1, 11, 0.5)
        self.slope = 2
        self.intercept = 3.5

        self.known_ylinlin = np.array([
             5.5,   6.5,   7.5,   8.5,   9.5,  10.5,  11.5,  12.5,  13.5,
            14.5,  15.5,  16.5,  17.5,  18.5,  19.5,  20.5,  21.5,  22.5,
            23.5,  24.5
        ])


        self.known_yloglin = np.array([
            3.5       ,  4.31093022,  4.88629436,  5.33258146,  5.69722458,
            6.00552594,  6.27258872,  6.50815479,  6.71887582,  6.90949618,
            7.08351894,  7.24360435,  7.3918203 ,  7.52980604,  7.65888308,
            7.78013233,  7.89444915,  8.0025836 ,  8.10517019,  8.20275051
        ])

        self.known_yloglog = np.array([
              33.11545196,    74.50976691,   132.46180783,   206.97157474,
             298.03906763,   405.66428649,   529.84723134,   670.58790216,
             827.88629897,  1001.74242175,  1192.15627051,  1399.12784525,
            1622.65714598,  1862.74417268,  2119.38892536,  2392.59140402,
            2682.35160865,  2988.66953927,  3311.54519587,  3650.97857845
        ])

        self.known_ylinlog = np.array([
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
        percent_diff = np.abs(ylinlog - self.known_ylinlog) / self.known_ylinlog
        nptest.assert_array_almost_equal(
            percent_diff,
            np.zeros(self.x.shape[0]),
            decimal=5
        )


@image_comparison(baseline_images=['test_probplot_prob'], extensions=['png'])
def test_probplot_prob():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    fig = viz.probplot(data, ax=ax, xlabel='Test xlabel', otherscale='log')
    nt.assert_true(isinstance(fig, plt.Figure))


@image_comparison(baseline_images=['test_probplot_qq'], extensions=['png'])
def test_probplot_qq():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    fig = viz.probplot(data, ax=ax, axtype='qq', ylabel='Test label',
                       otherscale='log', scatter_kws=dict(color='r'))


@image_comparison(baseline_images=['test_probplot_pp'], extensions=['png'])
def test_probplot_pp():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    scatter_kws = dict(color='b', linestyle='--', markeredgecolor='g', markerfacecolor='none')
    fig = viz.probplot(data, ax=ax, axtype='pp', otherscale='linear',
                       xlabel='test x', ylabel='test y', scatter_kws=scatter_kws)


@image_comparison(baseline_images=['test_probplot_prob_bestfit'], extensions=['png'])
def test_probplot_prob_bestfit():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    fig = viz.probplot(data, ax=ax, xlabel='Test xlabel', bestfit=True, otherscale='log')
    nt.assert_true(isinstance(fig, plt.Figure))


@image_comparison(baseline_images=['test_probplot_qq_bestfit'], extensions=['png'])
def test_probplot_qq_bestfit():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    fig = viz.probplot(data, ax=ax, axtype='qq', bestfit=True, ylabel='Test label', otherscale='log')


@image_comparison(baseline_images=['test_probplot_pp_bestfit'], extensions=['png'])
def test_probplot_pp_bestfit():
    fig, ax = plt.subplots()
    data = setup_plot_data()
    scatter_kws = {'marker': 's', 'color': 'red'}
    line_kws = {'linestyle': '--', 'linewidth': 3}
    fig = viz.probplot(data, ax=ax, axtype='pp', otherscale='linear',
                       xlabel='test x', bestfit=True, ylabel='test y',
                       scatter_kws=scatter_kws, line_kws=line_kws)
