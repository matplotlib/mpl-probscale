mpl-probscale: Real probability scales for matplotlib
=====================================================

.. image:: https://travis-ci.org/phobson/watershed.svg?branch=master
    :target: https://travis-ci.org/phobson/watershed

.. image:: https://coveralls.io/repos/phobson/mpl-probscale/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/phobson/mpl-probscale?branch=master

https://github.com/phobson/mpl-probscale

Simply importing ``probscale`` let's you use probability scales in your matplotlib figures:

.. code-block:: python

    import matplotlib.pyplot as plt
    import probscale

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_ylim(1e-2, 1e2)
    ax.set_yscale('log')

    ax.set_xlim(0.5, 99.5)
    ax.set_xscale('prob')


.. image:: /img/example.png

