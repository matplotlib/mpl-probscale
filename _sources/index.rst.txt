.. probscale documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mpl-probscale: Real probability scales for matplotlib
=====================================================

.. image:: https://github.com/matplotlib/mpl-probscale/actions/workflows/python-runtests-all.yml/badge.svg
    :target: https://github.com/matplotlib/mpl-probscale/actions/workflows/python-runtests-all.yml

.. image:: https://github.com/matplotlib/mpl-probscale/actions/workflows/ruff_ty.yml/badge.svg
    :target: https://github.com/matplotlib/mpl-probscale/actions/workflows/ruff_ty.yml

Installation
------------

Install the latest release with ``pip install probscale`` (also available on
conda-forge as ``mpl-probscale``). See the :doc:`installation` page for
details.

Quickstart
----------

Simply importing ``probscale`` lets you use probability scales in your
matplotlib figures:

.. code-block:: python

    import matplotlib.pyplot as plt
    import probscale

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0.5, 99.5)
    ax.set_xscale("prob")
    ax.set_ylim(1e-2, 1e2)
    ax.set_yscale("log")

.. image:: /img/example.png

Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorial/getting_started.ipynb
   tutorial/closer_look_at_viz.ipynb
   tutorial/closer_look_at_plot_pos.ipynb

Examples
--------

.. toctree::
   :maxdepth: 2

   auto_examples/index

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api.rst

Testing
-------

Run the test suite (including the image comparison tests) from the repository
root:

.. code-block:: console

    $ uv run pytest --mpl

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Project

   readme.md
   installation.rst
   authors.rst
   contributing.rst