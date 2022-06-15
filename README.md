[![github](https://img.shields.io/badge/GitHub-Eccentricity-blue.svg)](https://github.com/vijayvarma392/Eccentricity)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/vijayvarma392/Eccentricity/blob/master/LICENSE)
[![Build Status](https://github.com/vijayvarma392/Eccentricity/actions/workflows/test.yml/badge.svg)](https://github.com/vijayvarma392/Eccentricity/actions/workflows/test.yml)


# Eccentricity
Defining eccentricity for GW applications. See [wiki](https://github.com/vijayvarma392/Eccentricity/wiki).

## Automated tests
Continuous Integration tests are automatically run (using Github Actions) for every pull request.    
Before pushing, you should run these tests locally by doing the following from the base directory of the repo:
```
py.test test
```

## Waveform data
Follow the instructions in `data/README.md`

## Paper
To get the paper do:
```
git clone git@github.com:vijayvarma392/measure_eccentricity_paper.git paper
```
Note that this directory is not tracked in this repo as the directory `paper`
is ignored from `.gitignore`. However, the `paper` directory is a repo on its
own, so you can use it like a normal repo. Let's try to avoid submodules :)

For quick access to the paper: [![Latest build](https://img.shields.io/badge/PDF-latest-orange.svg?style=flat)](https://github.com/vijayvarma392/measure_eccentricity_paper/blob/pdflatex/paper.pdf)
