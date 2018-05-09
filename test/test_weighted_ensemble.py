from pyPhenology import models, utils
import pytest

obs, predictors = utils.load_test_data(name='vaccinium', phenophase='budburst')

m1 = models.ThermalTime(parameters={'T':0})
m2 = models.ThermalTime(parameters={'T':5})
m3 = models.Uniforc(parameters={'t1':1})
m4 = models.Linear()

core_models = [m1,m2,m3,m4]
stacking_iterations = 10

weighted_model = models.WeightedEnsemble(core_models=core_models)
weighted_model.fit(obs, predictors, optimizer_params='testing', 
                   iterations=stacking_iterations)

def test_WeightedEnsemble_initialize():
    """WeightedEnsemble model requires core_models set"""
    with pytest.raises(TypeError):
        models.BootstrapModel()

def test_WeightedEnsemble_weight_shape():
    """Array of fitted weights should be this shape"""
    assert weighted_model.fitted_weights.shape==(stacking_iterations, len(core_models))

def test_WeightedEnsemble_weight_shape2():
    """Array of mean fitted weights should be this shape"""
    assert weighted_model.weights.shape==(len(core_models),)
    
def test_WeightedEnsemble_predict_default():
    """Predict with no new data should return 1D array"""
    assert weighted_model.predict().shape == obs.doy.values.shape

def test_WeightedEnsemble_parameters():
    """Parameters should be the same when loaded again"""
    weighted_model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    assert loaded_model.get_params() == weighted_model.get_params()

def test_WeightedEnsemble_save_load():
    """"Save and load a bootstrap model"""
    weighted_model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    assert len(loaded_model.predict(obs, predictors)) == len(obs)

def test_WeightedEnsemble_do_not_predict_without_data():
    """Should not predict when no fitting was done and no new data passed """
    weighted_model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    with pytest.raises(TypeError):
        loaded_model.predict()