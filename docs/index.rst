.. pyPhenology documentation master file, created by
   sphinx-quickstart on Fri Feb  9 10:52:03 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyPhenology
=======================================

pyPheonology is a software package for building plant phenology models. It has numpy
at it's core, making model building and prediction extremely fast. The core code was written to model 
phenology observations from the National Phenology Network, where abundant species have several
thousand observations. The API is inspired by scikit-learn, so all models can work interchangeably
with the same code (eg: :ref:`example_model_selection_aic`). pyPhenology is currently used to build the
continental scale phenology forecasts on http://phenology.naturecast.org


.. toctree::
   :maxdepth: 1
   :caption: Documentation:
   
   install
   quickstart
   data_structures
   models
   control_parameters
   saving_loading
   optimizers
   utils
   examples
   api

Get in touch
============
See the `GitHub Repo <https://github.com/sdtaylor/pyPhenology>`__ to see the source code or submit issues and feature requests.

License
=======
pyPhenology uses the open source `MIT License <https://opensource.org/licenses/MIT>`__

Acknowledgments
===============
Development of this software was funded by `the Gordon and Betty Moore Foundation's Data-Driven Discovery Initiative <http://www.moore.org/programs/science/data-driven-discovery>`__ through `Grant GBMF4563 <http://www.moore.org/grants/list/GBMF4563>`__ to Ethan P. White.
