"""Shared type definitions for the probscale package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias, TypedDict

import numpy
from numpy.typing import ArrayLike, NDArray

#: The axes on which a probability/quantile transform is applied.
FitAxis: TypeAlias = Literal["x", "y", "both"]

#: A one-dimensional array of floating point values.
FloatingArray: TypeAlias = NDArray[numpy.floating]


PlotType: TypeAlias = Literal["prob", "pp", "qq"]


class DistLike(Protocol):
    """The subset of a scipy-style distribution used by the scale.

    Implementations must provide ``ppf`` and ``cdf`` methods that accept
    array-like values and return arrays of floating point values. Both
    distribution instances (e.g. ``scipy.stats.norm``) and classes that
    implement these methods (e.g. ``probscale.probscale._minimal_norm``)
    are compatible.
    """

    def ppf(self, q: ArrayLike) -> FloatingArray: ...

    def cdf(self, x: ArrayLike) -> FloatingArray: ...


class FitResults(TypedDict, total=False):
    """Coefficients and optional confidence band of a best-fit line."""

    slope: float
    intercept: float
    yhat_lo: FloatingArray | None
    yhat_hi: FloatingArray | None


class ProbPlotResults(TypedDict):
    """Results returned by ``viz.probplot`` when requested."""

    q: FloatingArray
    x: FloatingArray
    y: FloatingArray

    #: Model estimates of the x-data used to draw the best-fit line.
    xhat: FloatingArray | None
    #: Model estimates of the y-data used to draw the best-fit line.
    yhat: FloatingArray | None
    #: Coefficients (and optional confidence band) of the best-fit line.
    res: FitResults | None


#: A function that estimates a best-fit line with a percentile bootstrap.
BestFitEstimator = Callable[
    [ArrayLike, ArrayLike, ArrayLike, FitAxis | None],
    tuple[FloatingArray, FloatingArray],
]
