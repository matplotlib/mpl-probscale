import numpy
from matplotlib.transforms import Transform


def _mask_non_positives(a):
    """
    Return a Numpy array where all values outside ]0, 1[ are
    replaced with NaNs. If all values are inside ]0, 1[, the original
    array is returned.
    """
    mask = (a <= 0.0) | (a >= 1.0)
    if mask.any():
        return numpy.where(mask, numpy.nan, a)
    return a


def _clip_non_positives(a):
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

    def inverted(self):
        return ProbTransform(self.dist, as_pct=self.as_pct, nonpos=self.nonpos)