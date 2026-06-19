import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Genformer"
author = (
    "Rajdeep Pathak, Rahul Goswami, Madhurima Panja, "
    "Palash Ghosh, Tanujit Chakraborty, Donia Besher"
)
copyright = f"{datetime.now():%Y}, the Genformer authors"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Autodoc / autosummary
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}

# Allow the docs to build even when the heavy runtime deps aren't installed
# (e.g. minimal CI). Autodoc still reads the docstrings without importing these.
# NOTE: numpy/pandas are intentionally *not* mocked — they are lightweight and
# the source uses PEP 604 unions (e.g. ``np.ndarray | None``) that fail against
# a mock object at import time.
autodoc_mock_imports = [
    "torch",
    "darts",
    "lightning",
    "pytorch_lightning",
    "gluonts",
]

# Napoleon (Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_param = True

# Cross-project references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# nbsphinx: execute notebooks at build time if they don't already have outputs
nbsphinx_execute = "auto"
nbsphinx_allow_errors = True
suppress_warnings = ["nbsphinx"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "Genformer"
html_favicon = "_static/favicon.svg"
html_show_sourcelink = False

_github_url = "https://github.com/yuvrajiro/Genformer"

html_theme_options = {
    "logo": {
        "text": "Genformer",
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
    },
    "show_nav_level": 1,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "header_links_before_dropdown": 5,
    "navbar_align": "left",
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    "icon_links": [
        {
            "name": "GitHub",
            "url": _github_url,
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/genformer/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
}

html_context = {
    "github_user": "yuvrajiro",
    "github_repo": "Genformer",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}

html_sidebars = {
    "index": [],
}

# Copybutton: strip prompts so users copy clean code
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regex = True
