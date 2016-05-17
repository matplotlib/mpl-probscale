import numpy
from matplotlib.transforms import Transform


def _mask_out_of_bounds(a):
    """
    Return a Numpy array where all values outside ]0, 1[ are
    replaced with NaNs. If all values are inside ]0, 1[, the original
    array is returned.
    """
    a = numpy.array(a, float)
    mask = (a <= 0.0) | (a >= 1.0)
    if mask.any():
        return numpy.where(mask, numpy.nan, a)
    return a


def _clip_out_of_bounds(a):
    """
    Return a Numpy array where all values outside ]0, 1[ are
    replaced with eps or 1 - eps. If all values are inside ]0, 1[
    the original array is returned. (eps = 1e-300)
    """
    a = numpy.array(a, float)
    a[a <= 0.0] = 1e-300
    a[a >= 1.0] = 1 - 1e-300
    return a


class _ProbTransformMixin(Transform):
    """
    Mixin for MPL axes transform for quantiles/probabilities or
    percentages.

    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, dist, as_pct=True, nonpos='mask'):
        Transform.__init__(self)
        self.dist = dist
        self.as_pct = as_pct
        self.nonpos = nonpos
        if self.as_pct:
            self.factor = 100.0
        else:
            self.factor = 1.0

        if self.nonpos == 'mask':
            self._handle_out_of_bounds = _mask_out_of_bounds
        elif self.nonpos == 'clip':
            self._handle_out_of_bounds = _clip_out_of_bounds
        else:
            raise ValueError("`nonpos` muse be either 'mask' or 'clip'")


class ProbTransform(_ProbTransformMixin):
    """
    MPL axes tranform class to convert quantiles to probabilities
    or percents.

    """

    def transform_non_affine(self, prob):
        prob = self._handle_out_of_bounds(numpy.asarray(prob) / self.factor)
        q = self.dist.ppf(prob)
        return q

    def inverted(self):
        return QuantileTransform(self.dist, as_pct=self.as_pct, nonpos=self.nonpos)


class QuantileTransform(_ProbTransformMixin):
    """
    MPL axes tranform class to convert probabilities or percents to
    quantiles.

    """

    def transform_non_affine(self, q):
        prob = self.dist.cdf(q) * self.factor
        return prob

    def inverted(self):
        return ProbTransform(self.dist, as_pct=self.as_pct, nonpos=self.nonpos)
