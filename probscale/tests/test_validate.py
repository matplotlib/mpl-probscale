from matplotlib import pyplot

import nose.tools as nt
from matplotlib.testing.decorators import cleanup

from probscale import validate


class Test_axes_object(object):
    @nt.raises(ValueError)
    def test_bad_value(self):
        validate.axes_object('junk')

    @cleanup
    def test_with_ax(self):
        fig, ax = pyplot.subplots()
        fig1, ax1 = validate.axes_object(ax)
        nt.assert_true(isinstance(ax1, pyplot.Axes))
        nt.assert_true(isinstance(fig1, pyplot.Figure))
        nt.assert_true(ax1 is ax)
        nt.assert_true(fig1 is fig)

    @cleanup
    def test_with_None(self):
        fig1, ax1 = validate.axes_object(None)
        nt.assert_true(isinstance(ax1, pyplot.Axes))
        nt.assert_true(isinstance(fig1, pyplot.Figure))


class Test_fit_argument(object):
    @nt.raises(ValueError)
    def test_bad_fitarg(self):
        validate.fit_argument('junk', 'fitprobs')

    def test_x(self):
        nt.assert_equal('x', validate.fit_argument('x', 'fitprobs'))
        nt.assert_equal('x', validate.fit_argument('x', 'fitlogs'))

    def test_y(self):
        nt.assert_equal('y', validate.fit_argument('y', 'fitprobs'))
        nt.assert_equal('y', validate.fit_argument('y', 'fitlogs'))

    def test_both(self):
        nt.assert_equal('both', validate.fit_argument('both', 'fitprobs'))
        nt.assert_equal('both', validate.fit_argument('both', 'fitlogs'))

    def test_None(self):
        nt.assert_true(validate.fit_argument(None, 'fitprobs') is None)
        nt.assert_true(validate.fit_argument(None, 'fitlogs') is None)


class Test_axis_name(object):
    @nt.raises(ValueError)
    def test_bad_name(self):
        validate.axis_name('junk', 'axname')

    def test_x(self):
        nt.assert_equal('x', validate.axis_name('x', 'axname'))

    def test_y(self):
        nt.assert_equal('y', validate.axis_name('y', 'axname'))


class Test_axis_type(object):
    @nt.raises(ValueError)
    def test_bad_value(self):
        validate.axis_type("JUNK")

    def test_upper(self):
        nt.assert_equal('pp', validate.axis_type('PP'))
        nt.assert_equal('qq', validate.axis_type('QQ'))
        nt.assert_equal('prob', validate.axis_type('ProB'))

class Test_other_options(object):
    def test_None(self):
        nt.assert_dict_equal(validate.other_options(None), {})

    def test_dict(self):
        x = {'a': 1, 'b': 'test'}
        nt.assert_dict_equal(validate.other_options(x), x)