import pandas as pd
import pkg_resources
from . import models

def load_test_data(name='vaccinium'):
    """Pre-loaded phenology and associated
    temperature data.

    Available datasets
    ------------------
    'vaccinium'
        Vaccinium corymbosum phenology from Harvard Forest
        Both flowers (phenophase 501) and leaves (phenophase 371)
    
    Parameters
    ----------
    name : str
        Name of the test dataset
    
    Returns
    -------
    obs, temp : tuple
        Pandas dataframes of phenology observations
        and associated temperatures.
    """
    if name=='vaccinium':
        obs_file = 'data/vaccinium_obs.csv'
        temp_file= 'data/vaccinium_temperature.csv'
    else:
        raise ValueError('Uknown dataset name: ' + str(name))
        
    obs_file = pkg_resources.resource_stream(__name__, obs_file)
    temp_file = pkg_resources.resource_stream(__name__, temp_file)
    obs = pd.read_csv(obs_file)
    temp= pd.read_csv(temp_file)
    return obs, temp

def load_model(name):
    if not isinstance(name, str):
        raise TypeError('name must be string, got' + type(name))
    if name=='ThermalTime':
        return models.ThermalTime
    elif name=='Uniforc':
        return models.Uniforc
    elif name=='Unichill':
        return models.Unichill
    elif name=='Alternating':
        return models.Alternating
    elif name=='MSB':
        return models.MSB
    elif name=='Linear':
        return models.Linear
    else:
        raise ValueError('Unknown model name: '+name)