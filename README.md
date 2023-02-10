[![github](https://img.shields.io/badge/GitHub-gw_eccentricity-blue.svg)](https://github.com/vijayvarma392/gw_eccentricity)
[![PyPI version](https://badge.fury.io/py/gw_eccentricity.svg)](https://pypi.org/project/gw_eccentricity)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/vijayvarma392/gw_eccentricity/blob/main/LICENSE)
[![Build Status](https://github.com/vijayvarma392/gw_eccentricity/actions/workflows/test.yml/badge.svg)](https://github.com/vijayvarma392/gw_eccentricity/actions/workflows/test.yml)


# Welcome to gw_eccentricity!
**gw_eccentricity** provides methods to measure eccentricity and mean anomaly
from gravitational waveforms.

These methods are described in the following paper: <br/>
- [1] Md Arif Shaikh, Vijay Varma, Harald Pfeiffer, Antoni Ramos-Buades and Maarten van de Meent,
"Defining eccentricity for gravitational wave astronomy", (2022). [Add arXiv number]

If you find this package useful in your work, please cite reference [1] and
this package: <br>
- Md Arif Shaikh, Vijay Varma, Harald Pfeiffer, Antoni Ramos-Buades and Maarten van de Meent,
"gw_eccentricity: A Python package for measuring eccentricity from gravitational waves",
https://pypi.org/project/gw-eccentricity/

This package lives on
[GitHub](https://github.com/vijayvarma392/gw_eccentricity), is compatible with
`python3`, and is tested every week. You can see the current build status of
the main branch at the top of this page.


## Installation

### PyPI
**gw_eccentricity** is available through [PyPI](https://pypi.org/project/gw_eccentricity/):

```shell
pip install gw_eccentricity
```

### From source

```shell
git clone git@github.com:vijayvarma392/gw_eccentricity.git
cd gw_eccentricity
python setup.py install
```

If you do not have root permissions, replace the last step with
`python setup.py install --user`

### Dependencies

All of these can be installed through pip or conda.
* [numpy](https://docs.scipy.org/doc/numpy/user/install.html)
* [scipy](https://www.scipy.org/install.html)
* [h5py](http://docs.h5py.org/en/latest/build.html)
* [lalsuite](https://pypi.org/project/lalsuite)


## Usage
See the example notebook [here](https://github.com/vijayvarma392/gw_eccentricity/blob/main/examples/gw_eccentricity_demo.ipynb).

## Credits
The main contributors to this code are [Md Arif Shaikh](https://md-arif-shaikh.github.io/), [Vijay
Varma](https://vijayvarma.com), and [Harald Pfeiffer](https://www.aei.mpg.de/person/54205/2784). You can find the full list of contributors
[here](https://github.com/vijayvarma392/gw_eccentricity/graphs/contributors).
Please report bugs by raising an issue on our
[GitHub](https://github.com/vijayvarma392/gw_eccentricity) repository.

### Making contributions
See this
[README](https://github.com/vijayvarma392/gw_eccentricity/blob/main/README_developers.md)
for instructions on how to make contributions to this package.
