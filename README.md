# MatrixGen

[![PyPI](https://img.shields.io/pypi/v/matrixgen.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/matrixgen.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/matrixgen)][pypi status]
[![License](https://img.shields.io/pypi/l/matrixgen)][license]

[![Read the documentation at https://matrixgen.readthedocs.io/](https://img.shields.io/readthedocs/matrixgen/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/He-Is-HaZaRdOuS/matrixgen/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/He-Is-HaZaRdOuS/matrixgen/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/matrixgen/
[read the docs]: https://matrixgen.readthedocs.io/
[tests]: https://github.com/He-Is-HaZaRdOuS/matrixgen/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/He-Is-HaZaRdOuS/matrixgen
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Overview
- MatrixGen is a modular Python package for generating, resizing, and analyzing sparse matrices. It supports multiple synthesis methods and provides tools for structure-aware scaling, useful in scientific computing, numerical simulations, and ML workloads where realistic matrix patterns are essential.

## Features

- **Modular Sparse Matrix Generation**
Generate structurally realistic sparse matrices of arbitrary size from a given base matrix using a unified and extensible framework.

- **Multiple Scaling Techniques**
Includes support for adapted resizing methods such as:

    - Nearest Neighbor

    - Bilinear Interpolation

    - Lanczos Resampling

    - Discrete Fourier Transform (DFT)

    - Discrete Cosine Transform (DCT)

    - Wavelet Transforms

- **Preservation of Structure and Sparsity**
Each method is designed to maintain key structural properties including sparsity patterns, symmetry, and bandwidth.

- **Controlled Randomization**
Allows slight variations during generation to support realistic data augmentation while retaining essential characteristics.

- **Feature-Based Evaluation**
Measures similarity between original and synthetic matrices using structural features like:

    - Nonzero density

    - Bandwidth and profile

    - Symmetry (Psym)

    - Diagonal spread and entropy

    - Row/column distribution metrics

    Cosine similarity is used to quantify structural preservation.

- **Scalable Generation**
Capable of generating matrices at much larger scales than the input while preserving core traits — useful for performance benchmarking and large-scale ML testing.

- **Fills a Sparse Matrix Data Gap**
Addresses the lack of diverse, realistic sparse matrices for algorithm validation, benchmarking, and ML applications.

## Requirements

- Handled automatically via Poetry. Requires Python 3.11 or higher.

## Installation

You can install _MatrixGen_ via [pip] from [PyPI]:

```console
$ pip install matrixgen
```

## Usage

You can use MatrixGen via:
- Python API: `from matrixgen import RESIZE_METHODS, resize_matrix, load_matrix, save_matrix`
- Command Line: Run `matrixgen --help` to see options
- See the examples/ folder for usage patterns.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [Apache 2.0 license][license],
_MatrixGen_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

This project is based off of [MatrixGen-d](https://github.com/He-Is-HaZaRdOuS/matrixgen-d) which in itself is a fork of the original [MatrixGen](https://github.com/aliemrepmk/MatrixGen---A-Realistic-Sparse-Matrix-Generator) by [Ali Emre Pamuk](https://github.com/aliemrepmk), [Mert Altekin](https://github.com/AltekinMert), and [Faruk Kaplan](https://github.com/farukaplan). They are the original creators of the matrix synthesis/generation/resizing logic and without them, this project wouldn't exist.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/He-Is-HaZaRdOuS/matrixgen/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/He-Is-HaZaRdOuS/matrixgen/blob/main/LICENSE
[contributor guide]: https://github.com/He-Is-HaZaRdOuS/matrixgen/blob/main/CONTRIBUTING.md
[command-line reference]: https://matrixgen.readthedocs.io/en/latest/usage.html
