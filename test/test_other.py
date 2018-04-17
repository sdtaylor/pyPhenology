from pyPhenology import utils, models
import pytest
import numpy as np

obs, predictors = utils.load_test_data()
Model = utils.load_model('ThermalTime')
model = Model()
#####################################################
# Tests for other things besides the core model fitting/prediction


def test_basinhopping_method():
    model.fit(obs, predictors, method='BH', optimizer_params='testing')
    assert len(model.predict()) == len(obs)
    
def test_bruteforce_method():
    model.fit(obs, predictors, method='BF', optimizer_params='testing')
    assert len(model.predict()) == len(obs)

def test_daylength_util():
    """daylength equation from julian day & latitude.
    
    Numbers confirmed via http://www.solartopo.com/daylength.htm
    """
    d = models.utils.transforms.daylength(np.array([30,90,180]), np.array([20,30,40]))
    np
    assert np.all(np.round(d,1) == np.array([ 11.1 ,  12.3 ,  14.8]))
