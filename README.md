# pyPhenology [![Build Status](https://travis-ci.org/sdtaylor/pyPhenology.svg?branch=master)](https://travis-ci.org/sdtaylor/pyPhenology)
Plant phenology models in python with a scikit-learn inspired API

## Installation
Requires: scipy, pandas, and numpy

Install from Github  

```
pip install git+git://github.com/sdtaylor/pyPhenology
```

## Usage  

An example for now:

```
>>> from pyPhenology import models,utils
>>> doy, temp = utils.load_test_data(name='vaccinium')
>>> model = models.Thermal_Time()
>>> model.fit(doy, temp)
>>> model.get_params()
{'t1': 85.704951490688927, 'T': 7.0814430573372666, 'F': 185.36866570243012}
>>> 
```

Future notes on:
How the data needs to be structured
Making predictions
