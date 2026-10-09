from typing import Self

import numpy
from matplotlib.ticker import Formatter


class _FormatterMixin(Formatter):
    """A mpl-axes formatter mixin class"""

    factor: float
    top: float
    offset: int

    @classmethod
    def _sig_figs(
        cls: type[Self],
        x: float | str | None,
        n: int,
        expthresh: int = 5,
        forceint: bool = False,
    ) -> str:
        """Format a number with the correct number of significant digits.

        Parameters
        ----------
        x : int, float, or str
            The number you want to format. Strings are converted to
            floating point values before formatting. None, NaN, and
            infinite values return 'NA'.
        n : int
            The number of significant figures it should have.
        expthresh : int, optional (default = 5)
            The absolute value of the order of magnitude at which
            numbers are formatted in exponential notation.
        forceint : bool, optional (default is False)
            If True, the formatted value is rounded to an integer.

        Returns
        -------
        formatted : str
            The formatted number as a string.

        Examples
        --------
        >>> from probscale.formatters import PctFormatter
        >>> fmt = PctFormatter()
        >>> fmt._sig_figs(1247.15, 3)
        '1,250'
        >>> fmt._sig_figs(1247.15, 7)
        '1,247.150'

        """

        # return a string value unaltered
        if isinstance(x, str):
            out = cls._sig_figs(float(x), n, expthresh=expthresh, forceint=forceint)

        elif x == 0.0:
            out = "0"

        # check on the number provided
        elif x is not None and numpy.isfinite(x):
            # check on the _sig_figs
            if n < 1:
                raise ValueError("number of sig figs (n) must be greater than zero")

            elif forceint:
                out = f"{x:,.0f}"

            # logic to do all of the rounding
            else:
                order = numpy.floor(numpy.log10(numpy.abs(x)))

                if -1.0 * expthresh <= order <= expthresh:
                    decimal_places = int(n - 1 - order)

                    if decimal_places <= 0:
                        out = f"{round(x, decimal_places):,.0f}"

                    else:
                        fmt = "{0:,.%df}" % decimal_places  # noqa: UP031
                        out = fmt.format(x)

                else:
                    decimal_places = n - 1
                    fmt = "{0:.%de}" % decimal_places  # noqa: UP031
                    out = fmt.format(x)

        # with NAs and INFs, just return 'NA'
        else:
            out = "NA"

        return out

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format the value of a tick label.

        Parameters
        ----------
        x : float
            The value to format.
        pos : int, optional
            The tick position (unused).

        Returns
        -------
        formatted : str
            The formatted tick label.

        """
        if x < (10 / self.factor):
            out = self._sig_figs(x, 1)
        elif x <= (99 / self.factor):
            out = self._sig_figs(x, 2)
        else:
            order = int(
                numpy.ceil(numpy.round(numpy.abs(numpy.log10(self.top - x)), 6))
            )
            out = self._sig_figs(x, order + self.offset)

        return f"{out}"


class PctFormatter(_FormatterMixin):
    """
    Formatter class for MPL axes to display probabilities as percentages.

    Examples
    --------
    >>> from probscale import formatters
    >>> fmt = formatters.PctFormatter()
    >>> fmt(0.2)
    '0.2'
    >>> fmt(10)
    '10'
    >>> fmt(99.999)
    '99.999'

    """

    factor = 1.0
    offset = 2
    top = 100


class ProbFormatter(_FormatterMixin):
    """
    Formatter class for MPL axes to display probabilities as decimals.

    Examples
    --------
    >>> from probscale import formatters
    >>> fmt = formatters.ProbFormatter()
    >>> fmt(0.01)
    '0.01'
    >>> fmt(0.2)
    '0.20'
    >>> try:
    ...    fmt(10.5)
    ... except(ValueError):
    ...     print('formatter out of bounds')
    formatter out of bounds
    """

    factor = 100.0
    offset = 0
    top = 1
