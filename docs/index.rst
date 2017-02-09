.. probscale documentation master file, created by
   sphinx-quickstart on Thu Nov 19 23:14:08 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


mpl-probscale: Real probability scales for matplotlib
=====================================================

.. image:: https://travis-ci.org/matplotlib/mpl-probscale.svg?branch=master
    :target: https://travis-ci.org/matplotlib/mpl-probscale

.. image:: https://coveralls.io/repos/matplotlib/mpl-probscale/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/matplotlib/mpl-probscale?branch=master

https://github.com/matplotlib/mpl-probscale

Installation
------------

Official releases
~~~~~~~~~~~~~~~~~

Official releases are available through the conda-forge channel or pip:

``conda install mpl-probscale --channel=conda-forge``

or

``pip install probscale``

Development builds
~~~~~~~~~~~~~~~~~~

This is a pure-python package, so building from source is easy on all platforms:

::

    git clone git@github.com:matplotlib/mpl-probscale.git
    cd mpl-probscale
    pip install -e .


Quickstart
----------

Simply importing ``probscale`` lets you use probability scales in your matplotlib figures:

.. code-block:: python

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


.. image:: /img/example.png


Tutorials
=========

.. toctree::
   :maxdepth: 2

   tutorial/getting_started.rst
   tutorial/closer_look_at_viz.rst
   tutorial/closer_look_at_plot_pos.rst

Testing
=======

It's easiest to run the tests from an interactive python session:

.. code-block:: python

    import matplotlib
    matplotlib.use('agg')
    import probscale
    probscale.test()

API References
==============

.. toctree::
   :maxdepth: 2

   api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
