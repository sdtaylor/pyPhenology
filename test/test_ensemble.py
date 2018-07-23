from pyPhenology import models, utils
import numpy as np
import pytest

obs, predictors = utils.load_test_data(name='vaccinium', phenophase='budburst')

m1 = models.ThermalTime(parameters={'T':0})
m2 = models.ThermalTime(parameters={'T':5})
m3 = models.Uniforc(parameters={'t1':1})
m4 = models.Linear()

core_models = [m1,m2,m3,m4]

model = models.Ensemble(core_models=core_models)
model.fit(obs, predictors, optimizer_params='testing')

def test_Ensemble_initialize():
    """WeightedEnsemble model requires core_models set"""
    with pytest.raises(TypeError):
        models.Ensemble()

def test_Ensemble_predict_default():
    """Predict with no new data should return 1D array"""
    assert model.predict().shape == obs.doy.values.shape

def test_Ensemble_score():
    """The score method should return a number"""
    assert isinstance(model.score(), float)

def test_Ensemble_parameters():
    """Parameters should be the same when loaded again"""
    model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    assert loaded_model.get_params() == model.get_params()

def test_Ensemble_save_load():
    """"Save and load a model"""
    model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    assert len(loaded_model.predict(obs, predictors)) == len(obs)

def test_Ensemble_do_not_predict_without_data():
    """Should not predict when no fitting was done and no new data passed """
    model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    with pytest.raises(TypeError):
        loaded_model.predict()

def test_Ensemble_prediction_is_stable_after_saving():
    """Predictions shouldn't change after the model is saved and re-loaded"""
    predictions1 = model.predict()
    model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    predictions2 = loaded_model.predict(to_predict = obs, predictors=predictors)
    assert np.all(predictions1 == predictions2)

def test_Ensemble_aggregation():
    """It's theoretically possible for the median to equal the mean, but
    chances are that something is off internally if that happens.
    """
    mean_prediction = model.predict(aggregation='mean')
    median_prediction = model.predict(aggregation='median')
    assert np.all(~(mean_prediction==median_prediction))