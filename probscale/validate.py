from matplotlib import pyplot


def axes_object(ax):
    """ Checks if a value if an Axes. If None, a new one is created.

    """

    if ax is None:
        ax = pyplot.gca()
        fig = ax.figure
    elif isinstance(ax, pyplot.Axes):
        fig = ax.figure
    else:
        msg = "`ax` must be a matplotlib Axes instance or None"
        raise ValueError(msg)

    return fig, ax


def axis_name(axname, argname):
    valid_args = ['x', 'y']
    if axname.lower() not in valid_args:
        msg = 'Invalid value for {} ({}). Must be on of {}.'
        raise ValueError(msg.format(argname, axname, valid_args))

    return axname.lower()


def fit_argument(arg, argname):
    valid_args = ['x', 'y', 'both', None]
    if arg not in valid_args:
        msg = 'Invalid value for {} ({}). Must be on of {}.'
        raise ValueError(msg.format(argname, arg, valid_args))

    return arg


def axis_type(axtype):
    if axtype.lower() not in ['pp', 'qq', 'prob']:
        raise ValueError("invalid axtype: {}".format(axtype))
    return axtype.lower()


def other_options(options):
    return dict() if options is None else options.copy()