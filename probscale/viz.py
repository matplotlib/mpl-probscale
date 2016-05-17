﻿import numpy
from matplotlib import pyplot

from .probscale import _minimal_norm
from . import validate


def probplot(data, ax=None, plottype='prob', dist=None, probax='x',
             color=None, label=None, datascale='linear', xlabel=None,
             ylabel=None, bestfit=False, return_results=False,
             scatter_kws=None, line_kws=None, pp_kws=None):
    """ Probability, percentile, and quantile plots.

    Parameters
    ----------
    data : array-like
        1-dimensional data to be plotted
    ax : matplotlib axes, optional
        The Axes on which to plot. If one is not provided, a new Axes
        will be created.
    plottype : string (default = 'prob')
        Type of plot to be created. Options are:
           - 'prob': probabilty plot
           - 'pp': percentile plot
           - 'qq': quantile plot
    dist : scipy distribution, optional
        A distribtion to compute the scale's tick positions. If not
        specified, a normal distribution will be used.
    probax : string, optional (default = 'x')
        The axis ('x' or 'y') that will serve as the probability (or
        quantile) axis.
    color : valid matplotlib color specification, optional
        If provided, this value will be added to the ``scatter_kws``
        and ``line_kws`` dictionary under the "color" key.
    label : string, optional
        If provided, this legend label is applied to the scatter series
        of the probability plot.
    datascale : string, optional (default = 'log')
        Scale for the other axis that is not
    xlabel, ylabel : string, optional
        Axis labels for the plot.
    bestfit : bool, optional (default is False)
        Specifies whether a best-fit line should be added to the
        plot.
    return_results : bool (default = False)
        If True a dictionary of results of is returned along with the
        figure.
    scatter_kws, line_kws : dictionary, optional
        Dictionary of keyword arguments passed directly to ``ax.plot``
        when drawing the scatter points and best-fit line, respectively.
    pp_kws : dictionary, optional
        Dictionary of keyword arguments passed directly to
        ``viz.plot_pos``.

    Returns
    -------
    fig : matplotlib.Figure
        The figure on which the plot was drawn.
    result : dictionary of linear fit results, optional
        Keys are:
           - q : array of quantiles
           - x, y : arrays of data passed to function
           - xhat, yhat : arrays of modeled data plotted in best-fit line
           - res : array of coeffcients of the best-fit line.

    See also
    --------
    viz.plot_pos
    numpy.polyfit
    scipy.stats.probplot
    scipy.stats.mstats.plotting_positions

    """

    if dist is None:
        dist = _minimal_norm

    # check input values
    fig, ax = validate.axes_object(ax)
    probax = validate.axis_name(probax, 'x')

    # default values for plotting options
    scatter_kws = validate.other_options(scatter_kws)
    line_kws = validate.other_options(line_kws)
    pp_kws = validate.other_options(pp_kws)

    if color is not None:
        scatter_kws['color'] = color
        line_kws['color'] = color

    if label is not None:
        scatter_kws['label'] = label

    # check plottype
    plottype = validate.axis_type(plottype)

    # compute the plotting positions and sort the data
    probs, datavals = plot_pos(data, **pp_kws)
    qntls = dist.ppf(probs)

    # determine how the probability values should be expressed
    if plottype == 'qq':
        probvals = qntls
    else:
        probvals = probs * 100

    # set up x, y, Axes for probabilities on the x
    if probax == 'x':
        x, y = probvals, datavals
        if plottype == 'prob':
            ax.set_xscale('prob', dist=dist)
            fitprobs = 'x'
        else:
            fitprobs = None
            if plottype == 'pp':
                ax.set_xlim(left=0, right=100)

        ax.set_yscale(datascale)
        fitlogs = 'y' if datascale == 'log' else None

    # setup x, y, Axes for probabilities on the y
    elif probax == 'y':
        y, x = probvals, datavals
        if plottype == 'prob':
            ax.set_yscale('prob', dist=dist)
            fitprobs = 'y'
        else:
            fitprobs = None
            if plottype == 'pp':
                ax.set_ylim(bottom=0, top=100)

        ax.set_xscale(datascale)
        fitlogs = 'x' if datascale == 'log' else None

    # plot the final ROS data versus the Z-scores
    linestyle = scatter_kws.pop('linestyle', 'none')
    marker = scatter_kws.pop('marker', 'o')
    ax.plot(x, y, linestyle=linestyle, marker=marker, **scatter_kws)

    # maybe label the x-axis
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    # maybe label the y-axis
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # maybe do a best-fit and plot
    if bestfit:
        xhat, yhat, modelres = _fit_line(x, y, fitprobs=fitprobs, fitlogs=fitlogs, dist=dist)
        ax.plot(xhat, yhat, **line_kws)
    else:
        xhat, yhat, modelres = (None, None, None)

    # return the figure and maybe results of the best-fit
    if return_results:
        results = dict(q=qntls, x=x, y=y, xhat=xhat, yhat=yhat, res=modelres)
        return fig, results
    else:
        return fig


def plot_pos(data, postype=None, alpha=None, beta=None):
    """
    Compute the plotting positions for a dataset. Heavily borrows from
    ``scipy.stats.mstats.plotting_positions``.

    A plottiting position is defined as: ``(i-alpha)/(n+1-alpha-beta)``
    where:

        - ``i`` is the rank order
        - ``n`` is the size of the dataset
        - ``alpha`` and ``beta`` are parameters used to adjust the
          positions.

    The values of ``alpha`` and ``beta`` can be explicitly set. Typical
    values can also be access via the ``postype`` parameter. Available
    ``postype`` values (alpha, beta) are:

        "type 4" : (0, 1)
            Linear interpolation of the empirical CDF.
        "type 5" or "hazen" : (0.5, 0.5)
            Piecewise linear interpolation.
        "type 6" or "weibull" : (0, 0)
            Weibull plotting positions. Unbiased exceedance probability
            for all distributions. This is will be the default value.
        "type 7" : (1, 1)
            The default values in R.
        "type 8" : (1/3, 1/3)
            Approximately median-unbiased.
        "type 9" or "blom" : (0.375, 0.375)
            Approximately unbiased positions if the data are normally
            distributed.
        "median" : (0.3175, 0.3175)
            Median exceedance probabilities for all distributions
            (used in ``scipy.stats.probplot``).
        "apl" or "pwm" : (0.35, 0.35)
            Used with probability-weighted moments.
        "cunnane" : (0.4, 0.4)
            Nearly unbiased quantiles for normally distributed data.
        "gringorten" : (0.44, 0.44)
            Used for Gumble distributions.

    Parameters
    ----------
    data : array-like
        The values whose plotting positions need to be computed.
    postype : string, optional (default: "cunnane")
    alpha, beta : float, optional
        Custom plotting position parameters is the options available
        through the `postype` parameter are insufficient.

    Returns
    -------
    plot_pos : numpy.array
        The computed plotting positions, sorted.
    data_sorted : numpy.array
        The original data values, sorted.

    References
    ----------
    http://artax.karlin.mff.cuni.cz/r-help/library/lmomco/html/pp.html
    http://astrostatistics.psu.edu/su07/R/html/stats/html/quantile.html
    http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.probplot.html
    http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.mstats.plotting_positions.html

    """

    pos_params = {
        'type 4': (0, 1),
        'type 5': (0.5, 0.5),
        'type 6': (0, 0),
        'type 7': (1, 1),
        'type 8': (1/3., 1/3.),
        'type 9': (0.375, 0.375),
        'weibull': (0, 0),
        'median': (0.3175, 0.3175),
        'apl': (0.35, 0.35),
        'pwm': (0.35, 0.35),
        'blom': (0.375, 0.375),
        'hazen': (0.5, 0.5),
        'cunnane': (0.4, 0.4),
        'gringorten': (0.44, 0.44), # Gumble
    }

    postype = 'cunnane' if postype is None else postype
    if alpha is None and beta is None:
        alpha, beta = pos_params[postype.lower()]

    data = numpy.asarray(data, dtype=float).flatten()
    n = data.shape[0]
    pos = numpy.empty_like(data)
    pos[n:] = 0

    sorted_index = data.argsort()
    pos[sorted_index[:n]] = (numpy.arange(1, n+1) - alpha) / (n + 1.0 - alpha - beta)

    return pos[sorted_index], data[sorted_index]


def _fit_line(x, y, xhat=None, fitprobs=None, fitlogs=None, dist=None):
    """
    Fits a line to x-y data in various forms (linear, log, prob scales).

    Parameters
    ----------
    x, y : array-like
        Independent and dependent data, respectively.
    xhat : array-like, optional
        The values at which ``yhat`` should should be estimated. If
        not provided, falls back to the sorted values of ``x``.
    fitprobs, fitlogs : str, optional.
        Defines how data should be transformed. Valid values are
        'x', 'y', or 'both'. If using ``fitprobs``, variables should
        be expressed as a percentage, i.e.,
        Probablility transform = lambda x: ``dist``.ppf(x / 100.).
        Log transform = lambda x: numpy.log(x).
        Take care to not pass the same value to both ``fitlogs`` and
        ``figprobs`` as both transforms will be applied.
    dist : distribution, optional
        A fully-spec'd scipy.stats distribution-like object
        such that ``dist.ppf`` and ``dist.cdf`` can be called. If not
        provided, defaults to a minimal implementation of
        scipt.stats.norm.

    Returns
    -------
    xhat, yhat : numpy arrays
        Linear model estimates of ``x`` and ``y``.
    results : a statmodels result object
        The object returned by numpy.polyfit

    """

    fitprobs = validate.fit_argument(fitprobs, "fitprobs")
    fitlogs = validate.fit_argument(fitlogs, "fitlogs")

    # maybe set xhat to default values
    if xhat is None:
        xhat = numpy.array([numpy.min(x), numpy.max(x)])

    # maybe set dist to default value
    if dist is None:
        dist = _minimal_norm

    # maybe compute ppf of x
    if fitprobs in ['x', 'both']:
        x = dist.ppf(x/100.)
        xhat = dist.ppf(numpy.array(xhat)/100.)

    # maybe compute ppf of y
    if fitprobs in ['y', 'both']:
        y  = dist.ppf(y/100.)

    # maybe compute log of x
    if fitlogs in ['x', 'both']:
        x = numpy.log(x)

    # maybe compute log of y
    if fitlogs in ['y', 'both']:
        y = numpy.log(y)

    # do the best-fit
    coeffs = numpy.polyfit(x, y, 1)

    # estimate y values
    yhat = _estimate_from_fit(xhat, coeffs[0], coeffs[1],
                                  xlog=fitlogs in ['x', 'both'],
                                  ylog=fitlogs in ['y', 'both'])

    # maybe undo the ppf transform
    if fitprobs in ['y', 'both']:
        yhat = 100.* dist.cdf(yhat)

    # maybe undo ppf transform
    if fitprobs in ['x', 'both']:
        xhat = 100.* dist.cdf(xhat)

    return xhat, yhat, coeffs


def _estimate_from_fit(xdata, slope, intercept, xlog=False, ylog=False):
    """ Estimate the dependent variables of a linear fit given x-data
    and linear parameters.

    Parameters
    ----------
    xdata : numpy array or pandas Series/DataFrame
        The input independent variable of the fit
    slope : float
        Slope of the best-fit line
    intercept : float
        y-intercept of the best-fit line
    xlog, ylog : bool (default = False)
        Toggles whether or not the logs of the x- or y- data should be
        used to perform the regression.

    Returns
    -------
    yhat : same type as xdata
        Estimate of the dependent variable.

    """

    x = numpy.array(xdata)
    if ylog:
        if xlog:
            yhat = numpy.exp(intercept) * x  ** slope
        else:
            yhat = numpy.exp(intercept) * numpy.exp(slope) ** x

    else:
        if xlog:
            yhat = slope * numpy.log(x) + intercept

        else:
            yhat = slope * x + intercept

    return yhat
