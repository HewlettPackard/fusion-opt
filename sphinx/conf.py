# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../'))


project = 'fusion-opt'
copyright = '2024, HPE'
author = 'HPEr'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinxcontrib.bibtex', # for citations
    'sphinxemoji.sphinxemoji', # for emojis
    'sphinx_copybutton', # to copy code block
    'sphinx_panels', # for backgrounds
    'sphinx.ext.autosectionlabel', #for reference sections using its title
    'sphinx_multitoc_numbering', #numbering sections
]

# source for bib references
bibtex_bibfiles = ['references.bib']

# citation style
bibtex_default_style = 'plain'
bibtex_reference_style = 'label'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


source_suffix = '.rst'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []
