from pyPhenology import utils, models
import pytest

"""
Run thru each model and do a simple fit/predict to test
the internals of each one.
"""

obs, temp = utils.load_test_data()

model_names = ['Uniforc','Unichill','ThermalTime','Alternating','MSB',
               'Linear','Sequential','M1','Naive']

@pytest.mark.parametrize('model_name', model_names)
def test_individual_model(model_name):
    """Fit and predict a model, output length should equal input length"""
    Model = utils.load_model(model_name)
    model = Model()
    model.fit(obs, temp, optimizer_params='testing', debug=True)
    assert len(model.predict()) == len(obs)

def test_bootstrap_model():
    """Quick fit/predict on the bootstrap model"""
    model = models.BootstrapModel(core_model=models.ThermalTime,
                                  num_bootstraps=5)
    model.fit(obs, temp, optimizer_params='testing', debug=True)
    assert len(model.predict()) == len(obs)
