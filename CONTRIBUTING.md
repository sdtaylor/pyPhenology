# Contributing to pyPhenology

We welcome contributions of all kinds including improvements to the core code,
addition new models, improvements to the documentation, bug reports, or
anything else you can think of. We strive to be supportive of anyone who wants
to contribute, so don't be shy, give it a go, and we'll do our best to help.

## Code of Conduct

This project and everyone participating in it is governed by a
[Code of Conduct](https://github.com/sdtaylor/pyPhenology/blob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Process for contributing changes

We use a standard
[GitHub flow](https://guides.github.com/introduction/flow/index.html) for
development and reviewing contributions. Fork the repository. Make changes to a
branch of your fork and then submit a pull request.

## Addition of new models

New models must be implemented using Numpy arrays. See the basic
[Thermal Time](http://pyphenology.readthedocs.io/en/master/generated/pyPhenology.models.ThermalTime.html)
model for an example. Note that model classes only need to describe the actual
math behind the model and a few requirements for running it,
such as the model parameters and predictor data required.

If you wish to contribute a model that uses predictors other than temperature
you will need to implement two extra methods to process the predictors. See
the [M1](http://pyphenology.readthedocs.io/en/master/generated/pyPhenology.models.M1.html)
for an example, which uses daylength as a predictor.

Please feel free to open a GitHub issue for help and clarification on model
additions.

## Running the tests

We use [pytest](https://docs.pytest.org) for testing. To run the
tests first install nose using pip:

`pip install pytest`

Then from the root of the repository install pyPhenology:

`python setup.py install`

and run the tests:

`py.test`

You should see a bunch of output followed by something like:

```
========================= 220 passed in 91.93 seconds =========================

```

## Continuous integration

We use [Travis CI](https://travis-ci.org/) for continuous integration
testing. All pull requests will automatically report whether the tests are
passing.
