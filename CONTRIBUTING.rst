.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/phobson/probscale/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

mpl-probscale could always use more documentation, whether as part of the
official mpl-probscale docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/phobson/probscale/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `probscale` for local development.

1. Fork the `probscale` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/probscale.git

3. Install your local copy into a conda environment. Assuming you have conda installed, this is how you set up your fork for local development::

    $ conda config --add channels conda-forge
    $ conda create --name=probscale python=3.5 numpy matplotlib pytest pytest-cov pytest-pep8 pytest-mpl
    $ cd probscale/
    $ pip install -e .

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox::

    $ python check_probscale.py --mpl --pep8 --cov

6. Commit your changes and push your branch to GitHub::

    $ git add <files you want to stage>
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Matplotlib has good info on working with `source code`_ using `git and GitHub`_.

.. _source code: http://matplotlib.org/devel/coding_guide.html`
.. _git and GitHub: http://matplotlib.org/devel/gitwash/development_workflow.html

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.4 and higher. Check
   https://travis-ci.org/phobson/probscale/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ py.test tests.test_probscale


Configuring Sublime Text 3 to run the tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Sublime, got to Tools -> Build System -> New Build System.
Then add the following configuration and save as "wqio.sublime-build"::

    {
        "working_dir": "<path to the git repository>",
        "cmd": "<full path of the python executable> check_probscale.py --verbose <other pytest options>",
    }


Configuring Atom to run the tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Atom, install the build_ package, create a new file called ".atom-build.yml" in the
top level of the project directory, and add the following contents::

    cmd: "<full path of the python executable>"
    name: "wqio"
    args:
      - check_probscale.py
      - --verbose
      - <other pytest options ...>
    cwd: <path to the git repository>
    sh: false
    keymap: ctrl-b
    atomCommandName: namespace:testprobscale

After this, hitting ctrl+b in either text editor will run the test suite.

.. _build: https://atom.io/packages/build
