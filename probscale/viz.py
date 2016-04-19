import numpy
from matplotlib import pyplot

from .probscale import _minimal_norm
from . import validate


def probplot(data, ax=None, axtype='prob', probax='x',
             otherscale='linear', xlabel=None, ylabel=None,
             bestfit=False, return_results=False,
             scatter_kws=None, line_kws=None):
    """ Probability, percentile, and quantile plots.

    Parameters
    ----------
    data : array-like
        1-dimensional data to be plotted
    ax : matplotlib axes, optional
        The Axes on which to plot. If one is not provided, a new Axes
        will be created.
    axtype : string (default = 'prob')
        Type of plot to be created. Options are:
           - 'prob': probabilty plot
           - 'pp': percentile plot
           - 'qq': quantile plot
    probax : string, optional (default = 'x')
        The axis ('x' or 'y') that will serve as the probability (or
        quantile) axis.
    otherscale : string, optional (default = 'log')
        Scale for the other axis that is not
    xlabel, ylabel : string, optional
        Axis labels for the plot.
    bestfit : bool, optional (default is False)
        Specifies whether a best-fit line should be added to the
        plot.
    scatter_kws, line_kws : dictionary, optional
        Dictionary of keyword arguments passed directly to ``ax.plot``
        when drawing the scatter points and best-fit line, respectively.
    return_results : bool (default = False)
        If True a dictionary of results of is returned along with the
        figure.

    Returns
    -------
    fig : matplotlib.Figure
        The figure on which the plot was drawn.
    result : dictionary of linear fit results, optional
        Keys are:
           - q: array of quantiles
           - x, y: arrays of data passed to function
           - xhat, yhat: arrays of modeled data plotted in best-fit line
           - res: array of coeffcients of the best-fit line.


    """

    # check input values
    fig, ax = validate.axes_object(ax)
    probax = validate.axis_name(probax, 'x')

    # default values for plotting options
    scatter_kws = {} if scatter_kws is None else scatter_kws.copy()
    line_kws = {} if line_kws is None else line_kws.copy()

    # check axtype
    axtype = validate.axis_type(axtype)

    # compute the plotting positions and sort the data
    probs, datavals = _plot_pos(data)
    qntls = _minimal_norm.ppf(probs)

    # determine how the probability values should be expressed
    if axtype == 'qq':
        probvals = qntls
    else:
        probvals = probs * 100

    # set up x, y, Axes for probabilities on the x
    if probax == 'x':
        x, y = probvals, datavals
        if axtype == 'prob':
            ax.set_xscale('prob')
            fitprobs = 'x'
        else:
            fitprobs = None
            if axtype == 'pp':
                ax.set_xlim(left=0, right=100)

        ax.set_yscale(otherscale)
        fitlogs = 'y' if otherscale == 'log' else None

    # setup x, y, Axes for probabilities on the y
    elif probax == 'y':
        y, x = probvals, datavals
        if axtype == 'prob':
            ax.set_yscale('prob')
            fitprobs = 'y'
        else:
            fitprobs = None
            if axtype == 'pp':
                ax.set_ylim(bottom=0, top=100)

        ax.set_xscale(otherscale)
        fitlogs = 'x' if otherscale == 'log' else None

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
        xhat, yhat, modelres = _fit_line(x, y, fitprobs=fitprobs, fitlogs=fitlogs)
        ax.plot(xhat, yhat, **line_kws)
    else:
        xhat, yhat, modelres = (None, None, None)

    # return the figure and maybe results of the best-fit
    if return_results:
        return fig, dict(q=qntls, x=x, y=y, xhat=xhat, yhat=yhat, res=modelres)
    else:
        return fig


def _plot_pos(data, postype='cunnane', alpha=None, beta=None):
    pos_params = {
        'type 4':  (0, 1),
        'type 5':  (0.5, 0.5),
        'type 6':  (0, 0),
        'type 7':  (1, 1),
        'type 8':  (1/3., 1/3.),
        'type 9':  (3/8., 3/8.),
        'weibull': (0, 0),
        'median':  (0.3175, 0.3175),
        'apl':  (0.35, 0.35),
        'pwm':  (0.35, 0.35),
        'blom':  (0.375, 0.375),
        'hazen':  (0.5, 0.5),
        'cunnane':  (0.4, 0.4),
        'gringorten':  (0.44, 0.44), # Gumble
    }

    if alpha is None and beta is None:
        alpha, beta = pos_params.get(postype.lower(), pos_params['cunnane'])

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
    dist : scipy.stats distribution, optional
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
