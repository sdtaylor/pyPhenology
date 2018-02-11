import pandas as pd
import pkg_resources
from . import models

def load_test_data(name='vaccinium'):
    """Pre-loaded phenology and associated
    temperature data.

    Available datasets:
        'vaccinium'
            Vaccinium corymbosum phenology from Harvard Forest
            Both flowers (phenophase 501) and leaves (phenophase 371)
    
    Parameters:
        name : str
            Name of the test dataset
    
    Returns:
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
    """Load a model via a string string
    
    Options are ``['ThermalTime','Uniforc','Unichill','Alternating','MSB','Linear']``
    """
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
    

def check_data(observations, temperature, drop_missing=True, for_prediction=False):
    """Make sure observation and temperature data.frames are
    valid before submitting them to models.
    If observations are missing temperature data, optionally return
    a dataframe with those observations dropped.
    """
    models.validation.validate_observations(observations, for_prediction=for_prediction)
    models.validation.validate_temperature(temperature)
    original_obs_columns = observations.columns.values
    
    temperature_pivoted = temperature.pivot_table(index=['site_id','year'], columns='doy', values='temperature').reset_index()
    
    # This first day of temperature data causes NA issues because of leap years
    # TODO: generalize this a bit more
    temperature_pivoted.drop(-67, axis=1, inplace=True)
    
    observations_with_temp = observations.merge(temperature_pivoted, on=['site_id','year'], how='left')
    
    original_sample_size = len(observations_with_temp)
    rows_with_missing_data = observations_with_temp.isnull().any(axis=1)
    missing_info = observations_with_temp[['site_id','year']][rows_with_missing_data].drop_duplicates()
    print(len(missing_info))
    if len(missing_info)>0 and drop_missing:
        observations_with_temp.dropna(axis=0, inplace=True)
        n_dropped = original_sample_size - len(observations_with_temp)
        print('Dropped {n0} of {n1} observations because of missing data'.format(n0=n_dropped, n1=original_sample_size))
        print('\n Missing data from: \n' + str(missing_info))
        return observations_with_temp[original_obs_columns], temperature
    elif len(missing_info)>0:
        print('Missing temperature values detected')
        print('\n Missing data from: \n' + str(missing_info))
        return observations, temperature
    else:
        return observations, temperature