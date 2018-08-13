


Quickstart
====================

Comparison of Thermal Time Models
---------------------------------

This short overview will fit a basic :any:`ThermalTime` model, which has 3 parameters.
We will fit the model using data of flowering observations of blueberry (Vaccinium corymbosum)
from the Harvard Forest, which can be loaded from the package::

    from pyPhenology import models, utils
    import numpy as np

    observations, predictors = utils.load_test_data(name='vaccinium', phenophase='flowers')

The two objects (observations and predictors) are pandas ``data.frames``. The observations ``data.frame`` contains the direct
phenology observations as well as an associated site and year for each. The predictors ``data.frame`` has several
variables used to predict phenology, such as daily mean temperature, latitude and longitude, and daylength.
They are listed for each site and year represented in observations. Both of these ``data.frames`` are required for building models.
Read more about how phenology data is structured in this package :ref:`here <data_structure>`.

Next load the model::

    model = models.ThermalTime()

Then fit the model using the vaccinium flowering data::

    model.fit(observations, predictors)

The models estimated parameters can be viewed via a method call::

    model.get_params()
    {'F': 356.9379498434921, 'T': 5.507956473407982, 't1': 27.33480003499163}

Here, the fitted vaccinium flower model has a temperature threshold ``T`` of
5.5 degrees C, a starting accumulation day ``t1`` of julian day 27, and total degree day
requirement ``F`` of 357 units.

The fitted model can also be used to predict the day of flowering based off the
fitted parameters. By default it will make  predictions on the data used for
fitting::

    model.predict()
    array([126, 126, 127, 127, 126, 129, 129, 127, 132, 132, 133, 133, 132,
      132, 130, 130, 130, 129, 127, 126, 132, 130, 129, 132, 132, 133,
      133, 137, 137, 141, 141, 142, 132, 141, 141, 139, 139, 139, 139,
      137, 137, 141, 141, 141, 141, 142, 142, 142])

Note that model predictions are different than the actual observed predictions.
For example the root mean square error of the the predictions is about 3 days::

    np.sqrt(((model.predict() - observations.doy)**2).mean())
    2.846781808756454

This value can also be calculated by the model directly using :any:`Model.score`::

    model.score()
    2.846781808756454

It's common to fix 1 or more of the parameters in phenology models. For example
setting the starting day of warming accumulation, ``t1``, in the Thermal Time
model to January 1 (julian day 1). This is done in the model initialization. By default all
parameters in a model are estimated. Specifying one as fixed is done with the
``parameters`` argument::

    model = models.ThermalTime(parameters={'t1':1})

Note that the exact parameters are different for each model. See the details of
them in the :ref:`Primary Models` section.

The model can then be fit as before::

    model_fixed_t1 = models.ThermalTime(parameters={'t1':1})
    model_fixed_t1.get_params()
    {'F': 293.4456066438384, 'T': 7.542323552813556, 't1': 1}

This model has a slightly worse error than the  prior model, as ``t1=1`` is not
optimal for this particular dataset::

    model_fixed_t1.score()
    3.003470215156683

Finally to save the  model for later analysis use ``save_params()``::

    model.save_params(filename='blueberry_model.json')

The model can be loaded again later using :any:`utils.load_saved_model`::

    model = utils.load_saved_model(filename='blueberry_model.json')

Note that ``predict()`` and ``score()`` do not work on a loaded model as only
the model parameters, and not the data use to fit the model, are saved::

    model.predict()
    TypeError: No to_predict + temperature passed, and no fitting done. Nothing to predict

But given the data again, or new data, the loaded model can still be used to
make predictions::

    model.predict(to_predict=observations, predictors=predictors)
    array([126, 126, 127, 127, 126, 129, 129, 127, 132, 132, 133, 133, 132,
       132, 130, 130, 130, 129, 127, 126, 132, 130, 129, 132, 132, 133,
       133, 137, 137, 141, 141, 142, 132, 141, 141, 139, 139, 139, 139,
       137, 137, 141, 141, 141, 141, 142, 142, 142])
