# pyPhenology [![Build Status](https://travis-ci.org/sdtaylor/pyPhenology.svg?branch=master)](https://travis-ci.org/sdtaylor/pyPhenology) [![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/sdtaylor/phPhenology/master/LICENSE)
Plant phenology models in python with a scikit-learn inspired API

## Installation
Requires: scipy, pandas, and numpy

Install from Github  

```
pip install git+git://github.com/sdtaylor/pyPhenology
```

## Usage  

An example for now. The classic Thermal Time growing degree day model:

```
from pyPhenology import models, utils
doy, temp = utils.load_test_data(name='vaccinium')
model = models.Thermal_Time()
model.fit(doy, temp)
model.get_params()
{'t1': 85.704951490688927, 'T': 7.0814430573372666, 'F': 185.36866570243012}
```

Any of the parameters in a model can be set to a fixed value. For example the thermal time model with the threshold T set to 0 degrees C

```
model = models.Thermal_Time(parameters={'T':0})
model.fit(doy, temp)
model.get_params()
{'t1': 26.369813953905265, 'F': 333.76534368004388, 'T': 0}
```


Future notes on:  
How the data needs to be structured   
Making predictions  
