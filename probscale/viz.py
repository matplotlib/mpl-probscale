import numpy
from matplotlib import pyplot
from matplotlib import scale
from scipy import stats

from .probscale import ProbScale, _minimal_norm


scale.register_scale(ProbScale)


def _check_ax_obj(ax):
    """ Checks if a value if an Axes. If None, a new one is created.

    """

    if ax is None:
        fig, ax = pyplot.subplots()
    elif isinstance(ax, pyplot.Axes):
        fig = ax.figure
    else:
        msg = "`ax` must be a matplotlib Axes instance or None"
        raise ValueError(msg)

    return fig, ax


def _check_fit_arg(arg, argname):
    valid_args = ['x', 'y', 'both', None]
    if arg not in valid_args:
        msg = 'Invalid value for {} ({}). Must be on of {}.'
        raise ValueError(msg.format(argname, arg, valid_args))

    return arg


def _check_ax_name(axname, argname):
    valid_args = ['x', 'y']
    if axname not in valid_args:
        msg = 'Invalid value for {} ({}). Must be on of {}.'
        raise ValueError(msg.format(argname, arg, valid_args))

    return axname


def probplot(data, ax=None, axtype='prob', probax='x', otherscale='log',
             xlabel=None, ylabel=None, bestfit=False,
             scatter_kws=None, line_kws=None, return_results=False):
    """ Probability, percentile, and quantile plots.

    Parameters
    ----------
    data : array-like
        1-dimensional data to be plotted
    ax : optional matplotlib axes object or None (default).
        The Axes on which to plot. If None is provided, one will be
        created.
    axtype : string (default = 'pp')
        Type of plot to be created. Options are:
            - 'prob': probabilty plot
            - 'pp': percentile plot
            - 'qq': quantile plot
    yscale : string (default = 'log')
        Scale for the y-axis. Use 'log' for logarithmic (default) or
        'linear'.
    xlabel, ylabel : string or None (default)
        Axis labels for the plot.
    bestfit : bool, optional (default is False)
        Specifies whether a best-fit line should be added to the
        plot.
    scatter_kws, line_kws : dictionary
        Dictionary of keyword arguments passed directly to `pyplot.plot`
        when drawing the scatter points and best-fit line, respectively.
    return_results : bool (default = False)
        If True a dictionary of results of is returned along with the
        figure. Keys are:
            q - array of quantiles
            x, y - arrays of data passed to function
            xhat, yhat - arrays of modeled data plotted in best-fit line
            res - a statsmodels Result object.

    Returns
    -------
    fig : matplotlib.Figure
    result : dictionary of linear fit results.

    """

    fig, ax = _check_ax_obj(ax)
    _check_ax_name(probax, 'probax')

    scatter_kws = {} if scatter_kws is None else scatter_kws.copy()
    line_kws = {} if line_kws is None else line_kws.copy()

    if axtype not in ['pp', 'qq', 'prob']:
        raise ValueError("invalid axtype: {}".format(axtype))

    qntls, datavals = stats.probplot(data, fit=False)
    if axtype == 'qq':
        probvals = qntls
    else:
        probvals = stats.norm.cdf(qntls) * 100

    if probax == 'x':
        x, y = probvals, datavals
        if axtype == 'prob':
            ax.set_xscale('prob')
            fitprobs = 'x'
        else:
            fitprobs = None

        ax.set_yscale(otherscale)
        fitlogs = 'y' if otherscale == 'log' else None

    elif probax == 'y':
        y, x = probvals, datavals
        if axtype == 'prob':
            ax.set_yscale('prob')
            fitprobs = 'y'
        else:
            fitprobs = None

        ax.set_xscale(otherscale)
        fitlogs = 'x' if otherscale == 'log' else None

    # plot the final ROS data versus the Z-scores
    linestyle = scatter_kws.pop('linestyle', 'none')
    marker = scatter_kws.pop('marker', 'o')
    ax.plot(x, y, linestyle=linestyle, marker=marker, **scatter_kws)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if bestfit:
        xhat, yhat, modelres = _fit_line(x, y, fitprobs=fitprobs, fitlogs=fitlogs)
        ax.plot(xhat, yhat, **line_kws)
    else:
        xhat, yhat, modelres = (None, None, None)

    if return_results:
        return fig, dict(q=qntls, x=x, y=y, xhat=xhat, yhat=yhat, res=modelres)
    else:
        return fig


def _fit_line(x, y, xhat=None, fitprobs=None, fitlogs=None, dist=None):
    """ Fits a line to x-y data in various forms (raw, log, prob scales)

    Parameters
    ----------
    x, y : array-like
        Independent and dependent data, respectively.
    xhat : array-like or None, optional
        The values at which yhat should should be estimated. If
        not provided, falls back to the sorted values of ``x``.
    fitprobs, fitlogs : str, options.
        Defines how data should be transformed. Valid values are
        'x', 'y', or 'both'. If using ``fitprobs``, variables should
        be expressed as a percentage, i.e.,
        Probablility transform = lambda x: ``dist``.ppf(x / 100.).
        Log transform = lambda x: numpy.log(x).
        Take care to not pass the same value to both ``fitlogs`` and
        ``figprobs`` as both transforms will be applied.
    dist : scipy.stats distribution or None, optional
        A fully-spec'd scipy.stats distribution such that ``dist.ppf``
        and ``dist.cdf`` can be called. If not provided, defaults to a
        minimal implementation of scipt.stats.norm.

    Returns
    -------
    xhat, yhat : numpy arrays
        Linear model estimates of ``x`` and ``y``.
    results : a statmodels result object
        The object returned by statsmodels.OLS.fit()

    """

    _check_fit_arg(fitprobs, "fitprobs")
    _check_fit_arg(fitlogs, "fitlogs")

    if xhat is None:
        xhat = numpy.array([numpy.min(x), numpy.max(x)])

    if dist is None:
        dist = _minimal_norm

    if fitprobs in ['x', 'both']:
        x = dist.ppf(x/100.)
        xhat = dist.ppf(numpy.array(xhat)/100.)

    if fitprobs in ['y', 'both']:
        y  = dist.ppf(y/100.)

    if fitlogs in ['x', 'both']:
        x = numpy.log(x)
    if fitlogs in ['y', 'both']:
        y = numpy.log(y)

    coeffs = numpy.polyfit(x, y, 1)

    yhat = _estimate_from_fit(xhat, coeffs[0], coeffs[1],
                                  xlog=fitlogs in ['x', 'both'],
                                  ylog=fitlogs in ['y', 'both'])

    if fitprobs in ['y', 'both']:
        yhat = 100.* dist.cdf(yhat)
    if fitprobs in ['x', 'both']:
        xhat = 100.* dist.cdf(xhat)

    return xhat, yhat, coeffs


def _estimate_from_fit(xdata, slope, intercept, xlog=False, ylog=False):
    """ Estimate the dependent of a linear fit given x-data and linear
    parameters.

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
