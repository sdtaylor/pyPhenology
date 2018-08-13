from pyPhenology import models, utils
import numpy as np
import pytest

obs, predictors = utils.load_test_data(name='vaccinium', phenophase='budburst')


###########################################
# Setup various models
###########################################
########
# This is an ensemble of bootstrap models
alternating_bootstrap = models.BootstrapModel(core_model=models.Alternating, 
                                              num_bootstraps=10)
thermaltime_bootstrap = models.BootstrapModel(core_model=models.ThermalTime, 
                                              parameters={'t1':1,'T':0},
                                              num_bootstraps=10)
uniforc_bootstrap = models.BootstrapModel(core_model=models.Uniforc, 
                                          parameters={'t1':1,},
                                          num_bootstraps=10)

ensemble_bootstrap_model = models.Ensemble(core_models=[alternating_bootstrap,
                                                        thermaltime_bootstrap,
                                                        uniforc_bootstrap])
ensemble_bootstrap_model.fit(obs, predictors, optimizer_params='testing')
##########
# A simple ensemble model
tt1 = models.ThermalTime(parameters={'T':0})
tt2 = models.ThermalTime(parameters={'T':5})
uniforc = models.Uniforc(parameters={'t1':1})
linear = models.Linear()

ensemble_model = models.Ensemble(core_models=[tt1,tt2,uniforc, linear])
ensemble_model.fit(obs, predictors, optimizer_params='testing')

##########
# Bootstrap model
n_bootstraps=5

bootstrap_model = models.BootstrapModel(core_model=models.ThermalTime,
                                        num_bootstraps=n_bootstraps)
bootstrap_model.fit(obs, predictors, optimizer_params='testing')

##########
# Weighted Ensemble model
weighted1 = models.ThermalTime(parameters={'T':0})
weighted2 = models.ThermalTime(parameters={'T':5})
weighted3 = models.Uniforc(parameters={'t1':1})
weighted4 = models.Linear()

stacking_iterations = 10
len_core_models=4

weighted_model = models.WeightedEnsemble(core_models=[weighted1,weighted2,
                                                      weighted3,weighted4])
weighted_model.fit(obs, predictors, optimizer_params='testing', 
                   iterations=stacking_iterations)

##########################
# Combine into a list of tuples for use in pytest.mark.parametrize
test_cases  = [('EnsembleBootstrap',ensemble_bootstrap_model),
                    ('Ensemble',ensemble_model),
                    ('Bootstrap',bootstrap_model),
                    ('WeightedEnsemble',weighted_model)]

# A few of the tests aren't suitable for the  WeightedEnsemble model
test_cases_minus_weighted = [('EnsembleBootstrap',ensemble_bootstrap_model),
                                  ('Ensemble',ensemble_model),
                                  ('Bootstrap',bootstrap_model)]
#########################
# Tests for all ensemble methods

def test_bootstrap_initialize():
    """Bootstrap model requires core_model and num_bootstraps set"""
    with pytest.raises(TypeError):
        models.BootstrapModel()

@pytest.mark.parametrize('model_name, fitted_model', test_cases)
def test_ensemble_predict_default(model_name, fitted_model):
    """Predict with no new data should return 1D array"""
    assert fitted_model.predict().shape == obs.doy.values.shape

@pytest.mark.parametrize('model_name, fitted_model', test_cases)
def test_ensemble_model_predict_default(model_name, fitted_model):
    """Predict with no new data should return 1D array"""
    assert len(fitted_model.predict().shape) == 1
    
@pytest.mark.parametrize('model_name, fitted_model', test_cases_minus_weighted)
def test_ensemble_model_predict_none_shape(model_name, fitted_model):
    """Predict with aggregation='none' returns a 2d array"""
    assert len(fitted_model.predict(aggregation='none').shape) == 2

@pytest.mark.parametrize('model_name, fitted_model', test_cases)
def test_ensemble_score(model_name, fitted_model):
    """The score method should return a number"""
    assert isinstance(fitted_model.score(), float)

@pytest.mark.parametrize('model_name, fitted_model', test_cases)
def test_ensemble_parameters(model_name, fitted_model):
    """Parameters should be the same when loaded again"""
    fitted_model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    assert loaded_model.get_params() == fitted_model.get_params()

@pytest.mark.parametrize('model_name, fitted_model', test_cases)
def test_ensemble_save_load(model_name, fitted_model):
    """"Save and load a model"""
    fitted_model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    assert len(loaded_model.predict(obs, predictors)) == len(obs)

@pytest.mark.parametrize('model_name, fitted_model', test_cases)
def test_ensemble_do_not_predict_without_data(model_name, fitted_model):
    """Should not predict when no fitting was done and no new data passed """
    fitted_model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    with pytest.raises(TypeError):
        loaded_model.predict()

@pytest.mark.parametrize('model_name, fitted_model', test_cases)
def test_ensemble_prediction_is_stable_after_saving(model_name, fitted_model):
    """Predictions shouldn't change after the model is saved and re-loaded"""
    predictions1 = fitted_model.predict()
    fitted_model.save_params('model_params.json', overwrite=True)
    loaded_model = utils.load_saved_model('model_params.json')
    predictions2 = loaded_model.predict(to_predict = obs, predictors=predictors)
    assert np.all(predictions1 == predictions2)

@pytest.mark.parametrize('model_name, fitted_model', test_cases_minus_weighted)
def test_ensemble_aggregation(model_name, fitted_model):
    """It's theoretically possible for *all* median values to equal the mean, 
    but chances are that something is off internally if that happens.
    """
    mean_prediction = fitted_model.predict(aggregation='mean')
    median_prediction = fitted_model.predict(aggregation='median')
    assert np.any(~(mean_prediction==median_prediction))

######################
# More specific tests for the different ensembles
def test_bootstrap_model_predict_none_length():
    """Predict with aggregation='none' returns a 2d array"""
    assert bootstrap_model.predict(aggregation='none').shape[0] == n_bootstraps

def test_WeightedEnsemble_weight_shape():
    """Array of fitted weights should be this shape"""
    assert weighted_model.fitted_weights.shape==(stacking_iterations, len_core_models)

def test_WeightedEnsemble_weight_shape2():
    """Array of mean fitted weights should be this shape"""
    assert weighted_model.weights.shape==(len_core_models,)