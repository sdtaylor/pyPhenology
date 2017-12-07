import numpy as np
import pandas as pd
from scipy import optimize

def forcing_accumulator(temperature):
    """ The accumulated forcing for each observation
    and doy in the (obs, doy) array.
    """
    return temperature.cumsum(axis=1)

def doy_estimator(forcing, doy_series, threshold, non_prediction=-999):
    """ Find the doy that some forcing threshold is met for a large
    number of sites.
    
    Parameters
    ----------
    forcing : Numpy array
        (obs,doy) array where a is the number of replicates,
        (site, year, individual, etc) and doy corresponds
        to doy. values are the accumulated forcing for 
        each replicate,doy.
    
    doy_series : Numpy array
        1D array as produced by format_temperature(),
        identifying the doy values in forcing[:,b]
        
    threshold : float | int
        Threshold that must be met in forcing
    
    non_prediction : int
        Value to return if the threshold value is not
        met. A large value should be used during fitting
        to ensure unrealistic parameters are not chosen.
    
    Returns
    -------
    doy_final : Numpy array
        1D array of length obs with the doy values which
        first meet the threshold
    """
    
    
    n_samples = forcing.shape[0]

    #If threshold is not met for a particular row, ensure that a large doy
    #gets returned so it produces a large error
    forcing = np.column_stack((forcing, np.repeat(100000, n_samples)))
    doy_series = np.append(doy_series, non_prediction)

    #Repeating the full doy index for each row in forcing array
    doy_series = np.tile(doy_series, (n_samples,1))

    #The doy for each row where F was met
    doy_with_threshold_met = np.argmax(forcing>=threshold, axis=1)
    
    doy_final = doy_series[np.arange(n_samples), doy_with_threshold_met]
    
    return doy_final

def validate_temperature(temperature):
    """ Validate a temperature dataframe to the format used in this package.
    
    Parameters
    ----------
    temperature : Pandas Dataframe
    
    Returns
    -------
    temperature : The same dataframe but with only the valid columns
    """
    assert isinstance(temperature, pd.DataFrame), 'temperature should be a pandas dataframe'
    valid_columns = ['temperature','year','site_id']
    for column in valid_columns:
        assert column in temperature.columns, 'missing required temperature column: '+column
    
    return temperature[valid_columns]

def validate_DOY(DOY, for_prediction=False):
    """ Validate a DOY dataframe to the format used in this package.
    
    Parameters
    ----------
    DOY : Pandas Dataframe
    
    for_prediction : bool
        If being used to in model.predict(), then one less colum is required
    Returns
    -------
    DOY : The same dataframe but with only the valid columns
    """
    assert isinstance(DOY, pd.DataFrame), 'DOY should be a pandas dataframe'
    valid_columns = ['year','site_id']
    if not for_prediction: valid_columns.append('doy')
    
    for column in valid_columns:
        assert column in DOY.columns, 'missing required DOY column: '+column
    
    return DOY[valid_columns]


def format_temperature(DOY, temp_data, drop_missing=True, verbose=True):
    """Create a numpy array of shape (a,b), where a
    is equal to the sample size in DOY, and b is
    equal to the number of days in the yearly time
    series of temperature (ie. Jan 1 - July 30).
    Using a numpy array in this way allows for very 
    efficient processing of phenology mdoels.
    
    Parameters
    ----------
    DOY : Pandas Dataframe
        A data frame with columns ['doy','year','site_id'],
        where every row is an observation for an observed
        phenological event.
    
    temp_data : Pandas Dataframe
        A Dataframe with columns['temperature','year','site_id']
        which matches to the sites and years in DOY.
    
    drop_missing : bool
        Drop observations in DOY which do not have a complete
        temperature time series in temp_data.
    
    verbose : bool
        Show details of processing

    Returns
    -------
    temperature_array : Numpy array
        a 2D array described above
    
    doy_series : Numpy array
        1D array with length equal to the number of columns
        in temperature_array. Represents the doy values.
    
    """
    doy_series = temp_data.doy.dropna().unique()
    doy_series.sort()
    temp_data = temp_data.pivot_table(index=['site_id','year'], columns='doy', values='temperature').reset_index()
    
    # This first day of temperature data causes NA issues because of leap years
    # TODO: generalize this a bit more
    temp_data.drop(-67, axis=1, inplace=True)
    doy_series = doy_series[1:]
    
    DOY_with_temp = DOY.merge(temp_data, on=['site_id','year'], how='left')
    
    if drop_missing:
        DOY_with_temp_n = len(DOY_with_temp)
        DOY_with_temp.dropna(axis=0, inplace=True)
        if verbose: print('Dropped '+str(DOY_with_temp_n - len(DOY_with_temp)) + ' rows')
    
    return DOY_with_temp[doy_series].values, doy_series

class model_optimizer:
    def __init__(self, method, **params):
        assert method in ['DE','BH'], 'Uknown optimizer method: ' + str(method)
        self.method=method
        if method == 'DE':
            self.optimizer = optimize.differential_evolution
            
            optimizer_output = None
    def optimize_parameters(optimzer_func, bounds):
        if method == 'DE':
            optimizer_output = None
            