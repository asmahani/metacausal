"""Sphinx configuration for MetaCausal's API reference.

Built with autodoc + napoleon (existing docstrings are a mix of
Google-style and NumPy-style; napoleon parses both) and autosummary
(one stub page per public class/function, generated from each
subpackage's ``__all__`` -- see api.md). Narrative pages, if added
later, are authored in MyST Markdown via myst_parser.
"""

from __future__ import annotations

import metacausal

project = "MetaCausal"
copyright = "2026, Mansour T. A. Sharabiani, Alireza S. Mahani"
author = "Mansour T. A. Sharabiani, Alireza S. Mahani"
version = metacausal.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "myst_parser",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
]

# Embedded plot images (metacausal.plots.* docstrings): PNG only, no PDF/
# hi-res variant -- these are small illustrative figures, not publication
# plates, so skip the extra build time and repo/artifact size.
plot_formats = ["png"]

# Strip the >>> / ... REPL prompts (and any un-prompted output lines) from
# what actually gets copied, so a pasted example is directly runnable --
# matches the convention numpy/scipy/pandas/scikit-learn all use for their
# doctest-style examples.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}
autodoc_typehints = "description"
# "groupwise" documents methods, then attributes/properties (each block
# alphabetized), matching the Methods/Attributes autosummary tables above
# them -- those tables are always alphabetical internally and aren't
# affected by this setting, so "bysource" here would make the full member
# docs below disagree with the tables above them.
autodoc_member_order = "groupwise"
# Editable install of a 0.x package under active development; annotations use
# `from __future__ import annotations` (PEP 563) throughout, so autodoc reads
# them as strings and doesn't need to import forward-referenced names (e.g.
# the plots methods' `Axes` type, which only exists under TYPE_CHECKING) --
# unresolved cross-references just render as plain text rather than failing.
autodoc_type_aliases = {}

# napoleon (Google-style in ensemble.py/estimators.py/aggregation/weights.py;
# NumPy-style in plots/*.py -- both enabled since both are in active use)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

myst_enable_extensions = ["colon_fence"]
# Auto-generate slugged anchors for headings up to h3, so in-page links like
# `[Quick start](#quick-start)` (e.g. from the mirrored README) resolve.
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "MetaCausal"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
