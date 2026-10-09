"""Sphinx configuration for the probscale documentation."""

from importlib.metadata import version as _distribution_version

# -- Project information -----------------------------------------------------

project = "probscale"
author = "Paul Hobson (Herrera Environmental Consultants)"
copyright = f"2015-2026 {author}"

# The short X.Y version and the full version, including alpha/beta/rc tags.
release = _distribution_version("probscale")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

root_doc = "index"
language = "en"
# auto_examples/*.ipynb are the per-example notebooks sphinx-gallery writes
# for download; exclude them so myst-nb doesn't re-execute them (and so they
# don't collide with the gallery's own .rst pages).
exclude_patterns = [
    "_build",
    ".jupyter_cache",
    "**.ipynb_checkpoints",
    "auto_examples/*.ipynb",
    "auto_examples/**/*.ipynb",
]
templates_path = []
html_static_path = []

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # Renders the `.. plot::` blocks embedded in the API docstrings.
    "matplotlib.sphinxext.plot_directive",
    # numpydoc renders the docstrings (all numpydoc-style after the 2025
    # clean-up) into the API reference pages.
    "numpydoc",
    # myst-nb executes the tutorial notebooks during the build.
    "myst_nb",
    # sphinx-gallery turns examples/*.py into an image gallery.
    "sphinx_gallery.gen_gallery",
]

numpydoc_show_class_members = False
autodoc_member_order = "bysource"
todo_include_todos = True

# Options for the matplotlib .. plot:: directive (used in API docstrings)
plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

# -- myst-nb (tutorial notebooks) --------------------------------------------

nb_execution_mode = "auto"
nb_execution_timeout = 600
nb_execution_raise_on_error = True

# -- sphinx-gallery (example gallery) ----------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py$",
    "remove_config_comments": True,
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/matplotlib/mpl-probscale",
    "logo": {"text": "mpl-probscale"},
}
htmlhelp_basename = "probscaledoc"

# -- Options for LaTeX output ------------------------------------------------

latex_documents = [
    (
        root_doc,
        "probscale.tex",
        "probscale Documentation",
        "Paul Hobson (Herrera Environmental Consultants)",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (root_doc, "probscale", "probscale Documentation", [author], 1),
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        root_doc,
        "probscale",
        "probscale Documentation",
        author,
        "probscale",
        "Probability scales for matplotlib.",
        "Miscellaneous",
    ),
]
