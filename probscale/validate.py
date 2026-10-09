from __future__ import annotations

from typing import Any, cast

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._typing import BestFitEstimator, FitTransform
from .algo import _bs_fit


def axes_object(ax: Axes | None) -> tuple[Figure, Axes]:
    """Checks if a value if an Axes. If None, a new one is created.
    Both the figure and axes are returned (in that order).

    """

    if ax is None:
        ax = pyplot.gca()
        fig = cast(Figure, ax.figure)
    elif isinstance(ax, pyplot.Axes):
        fig = cast(Figure, ax.figure)
    else:
        msg = "`ax` must be a matplotlib Axes instance or None"
        raise ValueError(msg)

    return fig, ax


def axis_name(axis: str, axname: str) -> str:
    """
    Checks that an axis name is in ``{'x', 'y'}``. Raises an error on
    an invalid value. Returns the lower case version of valid values.

    """

    valid_args = ["x", "y"]
    if axis.lower() not in valid_args:
        msg = "Invalid value for {} ({}). Must be on of {}."
        raise ValueError(msg.format(axname, axis, valid_args))

    return axis.lower()


def fit_argument(arg: str | None, argname: str) -> FitTransform | None:
    """
    Checks that an axis options is in ``{'x', y', 'both', None}``.
    Raises an error on an invalid value. Returns the lower case version
    of valid values.

    """

    valid_args = ["x", "y", "both", None]
    if arg not in valid_args:
        msg = "Invalid value for {} ({}). Must be on of {}."
        raise ValueError(msg.format(argname, arg, valid_args))
    elif arg is not None:
        arg = arg.lower()

    return cast(FitTransform | None, arg)


def axis_type(axtype: str) -> str:
    """
    Checks that a valid axis type is requested.

      - *pp* - percentile axis
      - *qq* - quantile axis
      - *prob* - probability axis

    Raises an error on an invalid value. Returns the lower case version
    of valid values.

    """

    if axtype.lower() not in ["pp", "qq", "prob"]:
        raise ValueError(f"invalid axtype: {axtype}")
    return axtype.lower()


def axis_label(label: str | None) -> str:
    """
    Replaces None with an empty string for axis labels.

    """

    return "" if label is None else label


def other_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """
    Replaces None with an empty dict for plotting options.

    """

    return {} if options is None else options.copy()


def estimator(value: str) -> BestFitEstimator:
    value = value.lower()
    if value in ["res", "resid", "resids", "residual", "residuals"]:
        msg = "Bootstrapping the residuals is not ready yet"
        raise NotImplementedError(msg)
    elif value in ["fit", "values"]:
        est: BestFitEstimator = _bs_fit
    else:
        raise ValueError('estimator must be either "resid" or "fit".')

    return est
