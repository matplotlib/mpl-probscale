# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/matplotlib/mpl-probscale/issues>.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

mpl-probscale could always use more documentation, whether as part of the
official mpl-probscale docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/matplotlib/mpl-probscale/issues>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `probscale` for local development.

1. Fork the `probscale` repo on GitHub.
2. Clone your fork locally:

   ```console
   $ git clone git@github.com:your_name_here/probscale.git
   ```

3. Install your local copy into a virtual environment with
   [uv](https://docs.astral.sh/uv/):

   ```console
   $ cd probscale/
   $ uv sync
   ```

4. Create a branch for local development:

   ```console
   $ git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the tests:

   ```console
   $ uv run pytest --mpl
   ```

6. Commit your changes and push your branch to GitHub:

   ```console
   $ git add <files you want to stage>
   $ git commit -m "Your detailed description of your changes."
   $ git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request through the GitHub website.

Matplotlib has good info on working with [source code] using [git and GitHub].

[source code]: https://matplotlib.org/devdocs/devel/coding_guide.html
[git and GitHub]: https://matplotlib.org/devdocs/devel/development_workflow.html

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in the README.
3. The pull request should work for Python 3.11 and higher. Check the
   GitHub Actions results and make sure that the tests pass for all
   supported Python versions.

## Tips

To run a subset of tests:

```console
$ uv run pytest probscale/tests/test_probscale.py
```
