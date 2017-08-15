# mpl-probscale
Real probability scales for matplotlib

[![Build Status](https://travis-ci.org/matplotlib/mpl-probscale.svg)](https://travis-ci.org/matplotlib/mpl-probscale)
[![codecov](https://codecov.io/gh/matplotlib/mpl-probscale/branch/master/graph/badge.svg)](https://codecov.io/gh/matplotlib/mpl-probscale)

[Sphinx Docs](http://matplotlib.org/mpl-probscale/)

## Installation

### Official releases

Official releases are available through the conda-forge channel or pip

`conda install mpl-probscale --channel=conda-forge`

`pip install probscale`

### Development builds

This is a pure-python package, so building from source is easy on all platforms:

```
git clone git@github.com:matplotlib/mpl-probscale.git
cd mpl-probscale
pip install -e .
```

## Quick start

Simply importing `probscale` lets you use probability scales in your matplotlib figures:

```python
import matplotlib.pyplot as plt
import probscale
import seaborn
clear_bkgd = {'axes.facecolor':'none', 'figure.facecolor':'none'}
seaborn.set(style='ticks', context='notebook', rc=clear_bkgd)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_ylim(1e-2, 1e2)
ax.set_yscale('log')

ax.set_xlim(0.5, 99.5)
ax.set_xscale('prob')
seaborn.despine(fig=fig)
```

![Alt text](docs/img/example.png "Example axes")

## Testing

Testing is generally done via the ``pytest`` and ``numpy.testing`` modules.
The best way to run the tests is in an interactive python session:

```python
import matplotlib
matplotib.use('agg')
from probscale import tests
tests.test()
```
