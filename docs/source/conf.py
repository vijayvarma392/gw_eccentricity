# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sphinx_rtd_theme
import pathlib
import sys
import os
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.path.insert(0, os.path.abspath('../gw_eccentricity'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gw_eccentricity'
copyright = '2023, Md Arif Shaikh, Vijay Varma, Harald Pfeiffer'
author = 'Md Arif Shaikh, Vijay Varma, Harald Pfeiffer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'numpydoc',
              'nbsphinx',
              'sphinx.ext.autosectionlabel',
              'sphinx_tabs.tabs',
              "sphinx.ext.viewcode",
              'sphinx.ext.doctest',
              'sphinx.ext.napoleon',
              'myst_parser'
              ]
autosummary_generate = True
numpydoc_show_class_members = False
source_suffix = ['.rst', '.md', '.txt']
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = []
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
htmlhelp_basename = 'gw_eccentricitydoc'
