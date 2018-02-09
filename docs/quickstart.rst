


Quickstart
====================

A Thermal Time Model
--------------------

An example dataset of blueberry (Vaccinium corymbosum) flower and leaf phenology from Harvard forest is available.::

    from pyPhenology import models, utils
    observations, temp = utils.load_test_data(name='vaccinium')

The two objects observations and temp are pandas data.frames. The observations data.frame contains the direct
phenology observations as well as an associated site and year for each. temp is the daily temperature measurements
for each site and year represented in observations. Both of these are required for building models.
Read more about how phenology data is structured in this package here.

Initialize and fit a Thermal Time (link to model) model, which has 3 parameters::

    model = models.ThermalTime()
    model.fit(observations, temp)
    model.get_params()

A model can also be loaded via a text string::

    Model = utils.load_model('ThermalTime')
    model = Model()
    model.fit(observations, temp)

Using predict will give predictions of the same data as was used for fitting::

    model.predict()

New predictions can be made by passing a new observations and temp data.frames, where the ‘doy’ column in
observations is not required. For example here we fit the model and prediction on held out data::

    observations_test = observations[1:10]
    observations_train = observations[10:]

    model.fit(observations_train, temp)
    observations_test['prediction_doy'] = model.predict(observations_test, temp)
