from __future__ import annotations

from collections.abc import Callable

import numpy
from matplotlib.transforms import Transform
from numpy.typing import ArrayLike

from ._typing import DistLike, FloatingArray

#: A callable that handles values outside the ]0, 1[ range.
BoundsHandler = Callable[[ArrayLike], FloatingArray]


def _mask_out_of_bounds(a: ArrayLike) -> FloatingArray:
    """Return a Numpy array where all values outside ]0, 1[ are
    replaced with NaNs.

    Parameters
    ----------
    a : array-like
        The input values.

    Returns
    -------
    a : numpy array
        The input values with values outside ]0, 1[ replaced with NaN.
        If all values are inside ]0, 1[, the original array is returned.

    """
    a = numpy.array(a, float)
    mask = (a <= 0.0) | (a >= 1.0)
    if mask.any():
        return numpy.where(mask, numpy.nan, a)
    return a


def _clip_out_of_bounds(a: ArrayLike) -> FloatingArray:
    """Return a Numpy array where all values outside ]0, 1[ are
    replaced with eps or 1 - eps (eps = 1e-300).

    Parameters
    ----------
    a : array-like
        The input values.

    Returns
    -------
    a : numpy array
        The input values with values <= 0 replaced by 1e-300 and values
        >= 1 replaced by 1 - 1e-300.

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

    def __init__(
        self, dist: DistLike, as_pct: bool = True, out_of_bounds: str = "mask"
    ) -> None:
        Transform.__init__(self)
        self.dist: DistLike = dist
        self.as_pct: bool = as_pct
        self.out_of_bounds: str = out_of_bounds
        if self.as_pct:
            self.factor: float = 100.0
        else:
            self.factor = 1.0

        if self.out_of_bounds == "mask":
            self._handle_out_of_bounds: BoundsHandler = _mask_out_of_bounds
        elif self.out_of_bounds == "clip":
            self._handle_out_of_bounds = _clip_out_of_bounds
        else:
            raise ValueError("`out_of_bounds` muse be either 'mask' or 'clip'")


class ProbTransform(_ProbTransformMixin):
    """
    MPL axes transform class to convert quantiles to probabilities
    or percents.

    Parameters
    ----------
    dist : scipy.stats distribution
        The distribution whose ``ppf`` and ``cdf`` methods will set the
        scale of the axis.
    as_pct : bool, optional (True)
        Toggles the formatting of the probabilities associated with the
        tick labels as percentages (0 - 100) or fractions (0 - 1).
    out_of_bounds : string, optional ('mask' or 'clip')
        Determines how data outside the range of valid values is
        handled. The default behavior is to mask the data.
        Alternatively, the data can be clipped to values arbitrarily
        close to the limits of the scale.

    """

    def transform_non_affine(self, values: ArrayLike) -> FloatingArray:
        with numpy.errstate(divide="ignore", invalid="ignore"):
            prob = self._handle_out_of_bounds(numpy.asarray(values) / self.factor)
            q = self.dist.ppf(prob)
        return q

    def inverted(self) -> QuantileTransform:
        return QuantileTransform(
            self.dist, as_pct=self.as_pct, out_of_bounds=self.out_of_bounds
        )


class QuantileTransform(_ProbTransformMixin):
    """
    MPL axes transform class to convert probabilities or percents to
    quantiles.

    Parameters
    ----------
    dist : scipy.stats distribution
        The distribution whose ``ppf`` and ``cdf`` methods will set the
        scale of the axis.
    as_pct : bool, optional (True)
        Toggles the formatting of the probabilities associated with the
        tick labels as percentages (0 - 100) or fractions (0 - 1).
    out_of_bounds : string, optional ('mask' or 'clip')
        Determines how data outside the range of valid values is
        handled. The default behavior is to mask the data.
        Alternatively, the data can be clipped to values arbitrarily
        close to the limits of the scale.

    """

    def transform_non_affine(self, values: ArrayLike) -> FloatingArray:
        with numpy.errstate(divide="ignore", invalid="ignore"):
            prob = self.dist.cdf(values) * self.factor
        return prob

    def inverted(self) -> ProbTransform:
        return ProbTransform(
            self.dist, as_pct=self.as_pct, out_of_bounds=self.out_of_bounds
        )
