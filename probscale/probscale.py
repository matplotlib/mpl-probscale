import numpy as np
import matplotlib
from matplotlib.transforms import Transform
from matplotlib.scale import ScaleBase
from matplotlib.ticker import (
    FixedLocator,
    NullLocator,
    Formatter,
    NullFormatter,
    FuncFormatter,
)


def _mask_non_positives(a):
    """
    Return a Numpy array where all values outside ]0, 1[ are
    replaced with NaNs. If all values are inside ]0, 1[, the original
    array is returned.
    """
    mask = (a <= 0.0) | (a >= 1.0)
    if mask.any():
        return np.where(mask, np.nan, a)
    return a


def _clip_non_positives(a):
    a = np.array(a, float)
    a[a <= 0.0] = 1e-300
    a[a >= 1.0] = 1 - 1e-300
    return a


class _minimal_norm(object):
    _A = -(8 * (np.pi - 3.0) / (3.0 * np.pi * (np.pi - 4.0)))

    @classmethod
    def _approx_erf(cls, x):
        """ Approximate solution to the error function

        http://en.wikipedia.org/wiki/Error_function

        """

        guts = -x**2 * (4.0 / np.pi + cls._A * x**2) / (1.0 + cls._A * x**2)
        return np.sign(x) * np.sqrt(1.0 - np.exp(guts))

    @classmethod
    def _approx_inv_erf(cls, z):
        """ Approximate solution to the inverse error function

        http://en.wikipedia.org/wiki/Error_function

        """

        _b = (2 / np.pi / cls._A) + (0.5 * np.log(1 - z**2))
        _c = np.log(1 - z**2) / cls._A
        return np.sign(z) * np.sqrt(np.sqrt(_b**2 - _c) - _b)

    @classmethod
    def ppf(cls, q):
        """ Percent point function (inverse of cdf)

        Wikipedia: https://goo.gl/Rtxjme

        """
        return np.sqrt(2) * cls._approx_inv_erf(2*q - 1)

    @classmethod
    def cdf(cls, x):
        """ Cumulative density function

        Wikipedia: https://goo.gl/ciUNLx

        """
        return 0.5 * (1 + cls._approx_erf(x/np.sqrt(2)))


class _FormatterMixin(Formatter):
    @classmethod
    def _sig_figs(cls, x, n, expthresh=5, forceint=False):
        """ Formats a number with the correct number of sig figs.

        Parameters
        ----------
        x : int or float
            The number you want to format.
        n : int
            The number of significan figures it should have.
        expthresh : int, optional (default = 5)
            The absolute value of the order of magnitude at which numbers
            are formatted in exponential notation.
        forceint : bool, optional (default is False)
            If true, simply returns int(x)

        Returns
        -------
        formatted : str
            The formatted number as a string

        Examples
        --------
        >>> print(_sig_figs(1247.15, 3))
               1250
        >>> print(_sig_figs(1247.15, 7))
               1247.150

        """

        # return a string value unaltered
        if isinstance(x, str) or x == 0.0:
            out = str(x)

        # check on the number provided
        elif x is not None and not np.isinf(x) and not np.isnan(x):

            # check on the _sig_figs
            if n < 1:
                raise ValueError("number of sig figs (n) must be greater than zero")

            elif forceint:
                out = '{:,.0f}'.format(x)

            # logic to do all of the rounding
            else:
                order = np.floor(np.log10(np.abs(x)))

                if (-1.0 * expthresh <= order <= expthresh):
                    decimal_places = int(n - 1 - order)

                    if decimal_places <= 0:
                        out = '{0:,.0f}'.format(round(x, decimal_places))

                    else:
                        fmt = '{0:,.%df}' % decimal_places
                        out = fmt.format(x)

                else:
                    decimal_places = n - 1
                    fmt = '{0:.%de}' % decimal_places
                    out = fmt.format(x)

        # with NAs and INFs, just return 'NA'
        else:
            out = 'NA'

        return out

    def __call__(self, x, pos=None):
        if x < (10 / self.factor):
            out = self._sig_figs(x, 1)
        elif x <= (99 / self.factor):
            out =  self._sig_figs(x, 2)
        else:
            order = np.ceil(np.round(np.abs(np.log10(self.top - x)), 6))
            out = self._sig_figs(x, order + self.offset)

        return '{}'.format(out)


class PctFormatter(_FormatterMixin):
    factor = 1.0
    offset = 2
    top = 100


class ProbFormatter(_FormatterMixin):
    factor = 100.0
    offset = 0
    top = 1


class _ProbTransformMixin(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, dist, as_pct=True, nonpos='mask'):
        Transform.__init__(self)
        self.dist = dist
        if as_pct:
            self.factor = 100.0
        else:
            self.factor = 1.0

        if nonpos == 'mask':
            self._handle_nonpos = _mask_non_positives
        elif nonpos == 'clip':
            self._handle_nonpos = _clip_non_positives
        else:
            raise ValueError("`nonpos` muse be either 'mask' or 'clip'")


class ProbTransform(_ProbTransformMixin):
    def transform_non_affine(self, prob):
        q = self.dist.ppf(prob / self.factor)
        return q

    def inverted(self):
        return QuantileTransform(self.dist, as_pct=self.as_pct, nonpos=self.nonpos)


class QuantileTransform(_ProbTransformMixin):
    def transform_non_affine(self, q):
        prob = self.dist.cdf(q) * self.factor
        return prob

class InvertedProbTransform(_ProbTransformMixin):
    def transform_non_affine(self, a):
        return self.dist.cdf(a) * self.factor


class InvertedProbTransform(_ProbTransformMixin):
    def transform_non_affine(self, q):
        prob = self.dist.cdf(q) * self.factor
        return prob

    def inverted(self):
        return ProbTransform(self.dist, as_pct=self.as_pct, nonpos=self.nonpos)


class ProbScale(ScaleBase):
    """ A probability scale for matplotlib Axes.

    Parameters
    ----------
    axis : a matplotlib axis artist
        The axis whose scale will be set.
    dist : scipy.stats probability distribution, optional
        The distribution whose ppf/cdf methods should be used to compute
        the tick positions. By default, a minimal implimentation of the
        ``scipy.stats.norm`` class is used so that scipy is not a
        requirement.

    Examples
    --------
    >>> from matplotlib import pyplot
    >>> import probscale
    >>> fig, ax = pyplot.subplots()
    >>> ax.set_scale('prob')

    """

    name = 'prob'

    def __init__(self, axis, **kwargs):
        self.dist = kwargs.pop('dist', _minimal_norm)
        self.as_pct = kwargs.pop('as_pct', True)
        self.nonpos = kwargs.pop('nonpos', 'mask')
        self._transform = ProbTransform(self.dist, as_pct=self.as_pct)

    @classmethod
    def _get_probs(cls, nobs, as_pct):
        """ Returns the x-axis labels for a probability plot based on
        the number of observations (`nobs`).
        """
        if as_pct:
            factor = 1.0
        else:
            factor = 100.0

        order = int(np.floor(np.log10(nobs)))
        base_probs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

        axis_probs = base_probs.copy()
        for n in range(order):
            if n <= 2:
                lower_fringe = np.array([1, 2, 5])
                upper_fringe = np.array([5, 8, 9])
            else:
                lower_fringe = np.array([1])
                upper_fringe = np.array([9])

            new_lower = lower_fringe / 10**(n)
            new_upper = upper_fringe / 10**(n) + axis_probs.max()
            axis_probs = np.hstack([new_lower, axis_probs, new_upper])
        locs = axis_probs / factor
        return locs

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        log scaling.
        """

        axis.set_major_locator(FixedLocator(self._get_probs(1e8, self.as_pct)))
        if self.as_pct:
            axis.set_major_formatter(FuncFormatter(PctFormatter()))
        else:
            axis.set_major_formatter(FuncFormatter(ProbFormatter()))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance
        appropriate for the given logarithm base.
        """
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to positive values.
        """
        return (vmin <= 0.0 and minpos or vmin, vmax <= 0.0 and minpos or vmax)
