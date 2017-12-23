import numpy as np
import pandas as pd
from scipy import optimize

def mean_temperature(temperature, doy_series, start_doy, end_doy):
    """Mean temperature of a single time period.
    ie. mean spring temperature.
    
    Parameters
    ----------
    temperature : Numpy array
        (obs,doy) array of daily temperature values
        
    doy_series : Numpy array
        1D array as produced by format_temperature(),
        identifying the doy values in forcing[:,b]
    
    start_doy : int
        The beginning of the time period
        
    end_doy : int
        The end of the time period
    """
    assert start_doy<end_doy, 'start_doy must be < end_doy'
    
    spring_days = np.logical_and(doy_series>=start_doy,doy_series<=end_doy)
    return temperature[:,spring_days].mean(axis=1)

def sigmoid2(temperature, b, c):
    """The two parameter sigmoid function from Chuine 2000
    
    Parameters
    ----------
    temperature : Numpy array
        (obs,doy) array of daily temperature values
    
    b : int
        Sigmoid fitting parameter
    
    c : int
        Sigmoid fitting paramter
    
    Returns
    -------
    temperature : Numpy array
        (obs, doy) array of daily forcings derived from function
    """
    return 1 / (1 + np.exp(b*(temperature-c)))

def sigmoid3(temperature, a, b, c):
    """The three parameter sigmoid function from Chuine 2000
    
    Parameters
    ----------
    temperature : Numpy array
        (obs,doy) array of daily temperature values
    
    a : int
        Sigmoid fitting parameter
    
    b : int
        Sigmoid fitting paramter
    
    b : int
        Sigmoid fitting paramter
        
    Returns
    -------
    temperature : Numpy array
        (obs, doy) array of daily forcings derived from function
    """
    return 1 / (1 + np.exp(a*((temperature - c)**2) + b*(temperature-c)))

def forcing_accumulator(temperature):
    """ The accumulated forcing for each observation
    and doy in the (obs, doy) array.
    """
    return temperature.cumsum(axis=0)

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
    #If threshold is not met for a particular row, ensure that a large doy
    #gets returned so it produces a large error
    non_prediction_buffer = np.expand_dims(np.zeros_like(forcing[0]), axis=0)
    non_prediction_buffer[:] = 10e5
    forcing = np.concatenate((forcing, non_prediction_buffer), axis=0)
    doy_series = np.append(doy_series, non_prediction)

    #The index of the doy for each element where F was met
    doy_with_threshold_met = np.argmax(forcing>=threshold, axis=0)
    
    doy_final = np.take(doy_series, doy_with_threshold_met)
    
    return doy_final

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
    
    return DOY_with_temp[doy_series].values.T, doy_series

def fit_parameters(function_to_minimize, bounds, 
                   method, results_translator, optimizer_params):
    """Internal functions to estimate model parameters. 
    
    Methods
    -------
    'DE', Differential evolution
        Uses a large number of randomly specified parameters which converge
        on a global optimum. 
    
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    
    'BF', Brute force
        Searches for the best parameter set within a confined space. Can take
        an extremely long time if used beyond 2 or 3 parameters.
        
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html
    
    'SA', Simulated annealing
        The most commonly used method in phenology modelling. Not yet implemented
        here as scipy has discontinued it in favor of basin hopping.
        
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.anneal.html
        
    'BH, Basin hopping
        Starts off in a search space randomly, "hopping" around until a suitable
        minimum value is found. Note yet implimented.
        
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
            
    Parameters
    ----------
    funtions_to_minimize : func
        A minimizer function to pass to the optimizer model. Normally
        models._base_model.scipy_error
        
    bounds : list
        List of tuples specifying the upper and lower search space,
        where each tuple represents a model parameter
    
    method : str
        Optimization method to use
    
    results_translator : func
        A function to translate the optimizer output to a dictionary
    
    optimzier_parms : dict
        parameters to pass to the scipy optimizer
        
    Returns
    -------
    fitted_parameters : dict
        Dictionary of fitted parameters
    
    """
    
    assert method in ['DE','BH', 'BF'], 'Uknown optimizer method: ' + str(method)
    if method == 'DE':
        default_params = {'maxiter':None, 
                          'popsize':100, 
                          'mutation':1.5, 
                          'recombination':0.25,
                          'disp':False}
        
        default_params.update(optimizer_params)
        
        optimize_output = optimize.differential_evolution(function_to_minimize,
                                                          bounds=bounds, 
                                                          **default_params)
        fitted_parameters = results_translator(optimize_output['x'])

    elif method == 'BH':
        raise NotImplementedError('Basin Hopping not working yet')
    elif method == 'SE':
        raise NotImplementedError('Simulated Annealing not working yet')
    elif method == 'BF':
        default_params = {'Ns':5,
                          'finish':optimize.fmin_bfgs,
                          'disp':False}
        default_params.update(optimizer_params)
        
        # BF takes a tuple of tuples instead of a list of tuples like DE
        bounds = tuple(bounds)

        optimize_output = optimize.brute(func = function_to_minimize,
                                         ranges = bounds,
                                         **default_params)

        fitted_parameters =  results_translator(optimize_output)
    else:
        raise ValueError('Uknown optimizer method: '+str(method))
    
    return fitted_parameters
            
