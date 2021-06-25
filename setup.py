from setuptools import setup, find_packages

NAME ='pyPhenology'
DESCRIPTION = 'Plant phenology models in python'
URL = 'https://github.com/sdtaylor/pyPhenology'
AUTHOR = 'Shawn Taylor'
LICENCE = 'MIT'
LONG_DESCRIPTION = """
# pyPhenology  
pyPheonology is a software package for building plant phenology models. It has
numpy at it’s core, making model building and prediction extremely fast. The
core code was written to model phenology observations from the National
Phenology Network, where abundant species have several thousand observations.
The API is inspired by scikit-learn, so all models can work interchangeably
with the same code. pyPhenology is currently used to build the continental
scale phenology forecasts on http://phenology.naturecast.org

## Full documentation  

[http://pyphenology.readthedocs.io/en/master/](http://pyphenology.readthedocs.io/en/master/)


## Installation
Requires: scipy, pandas, joblib, and numpy

Install via pip

```
pip install pyPhenology
```

Or install the latest version from Github  

```
pip install git+git://github.com/sdtaylor/pyPhenology
```

## Get in touch
See the [GitHub Repo](https://github.com/sdtaylor/pyPhenology) to see the 
source code or submit issues and feature requests.

## Citation

If you use this software in your research please cite it as:

Taylor, S. D. (2018). pyPhenology: A python framework for plant phenology 
modelling. Journal of Open Source Software, 3(28), 827. https://doi.org/10.21105/joss.00827

Bibtex:
```
@article{Taylor2018,
author = {Taylor, Shawn David},
doi = {10.21105/joss.00827},
journal = {Journal of Open Source Software},
mendeley-groups = {Software/Data},
month = {aug},
number = {28},
pages = {827},
title = {{pyPhenology: A python framework for plant phenology modelling}},
url = {http://joss.theoj.org/papers/10.21105/joss.00827},
volume = {3},
year = {2018}
}

```

## Acknowledgments

Development of this software was funded by
[the Gordon and Betty Moore Foundation's Data-Driven Discovery Initiative](http://www.moore.org/programs/science/data-driven-discovery) through
[Grant GBMF4563](http://www.moore.org/grants/list/GBMF4563) to Ethan P. White.

"""

# Set the version number in pyPhenology/version.py
version = {}
with open('pyPhenology/version.py') as fp:
    exec(fp.read(), version)
VERSION = version['__version__']

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url=URL,
      author=AUTHOR,
      license=LICENCE,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
