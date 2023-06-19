# Instructions for developers
See [wiki](https://github.com/vijayvarma392/gw_eccentricity/wiki).

## Contributing to gw_eccentricity

The preferred method of making contributions is to
[fork](https://help.github.com/articles/fork-a-repo/) + [pull
request](https://help.github.com/articles/about-pull-requests/) from the main
[repo](https://github.com/vijayvarma392/gw_eccentricity).

Before doing a pull request, you should check that your changes don't break
anything by running the following from the root directory of your check-out.
Every pull request will be automatically tested by github.
```
./run_tests
```

## Adding a new eccentricity definition
For adding a new definition of eccentricity follow the guidelines [here](https://github.com/vijayvarma392/gw_eccentricity/wiki/Adding-new-eccentricity-definitions)

## Waveform data
Follow the instructions in `data/README.md`

## PyPI release
Note: Currently, only Vijay and Arif can do this, as they are the only ones with maintainer access to the PyPI package.
1. Update the version number at the top of gw_eccentricity/gw_eccentricity.py and commit all changes. This version number gets propagated to setup.py.
2. Do the following to release the new version:
```shell
python -m build
twine upload dist/*
```
