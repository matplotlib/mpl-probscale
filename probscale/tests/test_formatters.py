import sys
PYTHON27 = sys.version_info.major == 2

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from matplotlib.testing.decorators import image_comparison, cleanup
import nose.tools as nt
import numpy.testing as nptest

from probscale import formatters


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
    fmt = formatters.PctFormatter()
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
    fmt = formatters.ProbFormatter()
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


class Test_ProbFormatter_sig_figs(Mixin_Check_ProbFormatter_sig_figs):
    def setup(self):
        self.x = 0.123456
        self.known_3 = '0.123'
        self.known_4 = '0.1235'
        self.known_8 = '0.12345600'
        self.known_exp3 = '1.23e-07'
        self.known_int = '0'
        self.factor = 10**-6
