from pyPhenology import models, utils
import pytest

obs, predictors = utils.load_test_data()

n_bootstraps=5

bootstrap_model = models.BootstrapModel(core_model=models.ThermalTime,
                                        num_bootstraps=n_bootstraps)
bootstrap_model.fit(obs, predictors, optimizer_params='testing')

def test_bootstrap_initialize():
    """Bootstrap model requires core_model and num_bootstraps set"""
    with pytest.raises(TypeError):
        models.BootstrapModel()
        
def test_bootstrap_model_predict_default():
    """Predict with no new data should return 1D array"""
    assert len(bootstrap_model.predict().shape) == 1
    
def test_bootstrap_model_predict_none_shape():
    """Predict with aggregation='none' returns a 2d array"""
    assert len(bootstrap_model.predict(aggregation='none').shape) == 2

def test_bootstrap_model_predict_none_length():
    """Predict with aggregation='none' returns a 2d array"""
    assert bootstrap_model.predict(aggregation='none').shape[0] == n_bootstraps

def test_bootstrap_model_parameters():
    """Parameters should be the same when loaded again"""
    parameters = bootstrap_model.get_params()
    new_model = models.BootstrapModel(core_model=models.ThermalTime,
                                      num_bootstraps=n_bootstraps,
                                      parameters=parameters)
    assert new_model.get_params() == bootstrap_model.get_params()

def test_bootstrap_save_load():
    """"Save and load a bootstrap model"""
    bootstrap_model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    assert len(loaded_model.predict(obs, predictors)) == len(obs)

def test_do_not_predict_without_data():
    """Should not predict when no fitting was done and no new data passed """
    bootstrap_model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    with pytest.raises(TypeError):
        loaded_model.predict()