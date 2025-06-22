"""Sphinx configuration."""

project = "MatrixGen"
author = "Yousif Suhail"
copyright = "2025, Yousif Suhail"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
