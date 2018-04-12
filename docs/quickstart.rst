


Quickstart
====================

A Thermal Time Model
--------------------

An example dataset of blueberry (Vaccinium corymbosum) flower and leaf phenology from Harvard forest is available.::

    from pyPhenology import models, utils
    observations, predictors = utils.load_test_data(name='vaccinium')

The two objects (observations and predictors) are pandas ``data.frames``. The observations ``data.frame`` contains the direct
phenology observations as well as an associated site and year for each. predictors ``data.frame`` has several
variables used to predict phenology, such as daily temperature measurements, latitude and longitude, and daylength.
They listed for each site and year represented in observations. Both of these ``data.frames`` are required for building models.
Read more about how phenology data is structured in this package :ref:`here <data_structure>`.

Initialize and fit a :any:`ThermalTime` model, which has 3 parameters::

    model = models.ThermalTime()
    model.fit(observations, predictors)
    model.get_params()

A model can also be loaded via a text string with :any:`load_model`::

    Model = utils.load_model('ThermalTime')
    model = Model()
    model.fit(observations, predictors)

Using predict will give predictions of the same data as was used for fitting::

    model.predict()

New predictions can be made by passing a new observations and temp data.frames, where the `doy` column in
observations is not required. For example here we fit the model and prediction on held out data::

    observations_test = observations[1:10]
    observations_train = observations[10:]

    model.fit(observations_train, predictors)
    observations_test['prediction_doy'] = model.predict(observations_test, predictors)
