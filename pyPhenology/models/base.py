import numpy as np
import pandas as pd
from . import utils, validation
import time
from collections import OrderedDict

class _base_model():
    def __init__(self):
        self._fitted_params = {}
        self.obs_fitting = None
        self.temperature_fitting = None
        self.doy_series = None
        self.debug=False
        
    def fit(self, observations, temperature, method='DE', optimizer_params={}, 
            verbose=False, debug=False):
        """Estimate the parameters of a model.
        
        Parameters
        ----------
        observations : dataframe
            pandas dataframe in the format specific to this package
        
        temperature : dataframe
            pandas dataframe in the format specific to this package
        
        method : str
            Optimization method to use
        
        optimizer_params : dict
            Arguments for the optimizer
        
        verbose : bool
            display progress of the optimizer
        
        debug : bool
            display various internals
        
        """
        
        validation.validate_temperature(temperature)
        validation.validate_observations(observations)
        assert len(self._parameters_to_estimate)>0, 'No parameters to estimate'
    
        self.obs_fitting, self.temperature_fitting, self.doy_series = utils.format_data(observations, temperature, verbose=verbose)
        
        if debug:
            verbose=True
            self.debug=True
            self.model_timings=[]
            print('estimating params:\n {x} \n '.format(x=self._parameters_to_estimate))
            print('array passed to optimizer:\n {x} \n'.format(x=self._scipy_bounds()))
            print('fixed params:\n {x} \n '.format(x=self._fixed_parameters))
        if verbose:
            fitting_start = time.time()

        self._fitted_params = utils.fit_parameters(function_to_minimize = self._scipy_error,
                                                   bounds = self._scipy_bounds(),
                                                   method=method,
                                                   results_translator=self._translate_scipy_parameters,
                                                   optimizer_params = optimizer_params,
                                                   verbose=verbose)
        if verbose:
            total_fit_time = round(time.time() - fitting_start,5)
            print('Total model fitting time: {s} sec.\n'.format(s=total_fit_time))
            
        if debug:
            n_runs = len(self.model_timings)
            mean_time = np.mean(self.model_timings).round(5)
            print('Model iterations: {n}'.format(n=n_runs))
            print('Mean timing: {t} sec/iteration \n\n'.format(t=mean_time))
            self.debug=False
        self._fitted_params.update(self._fixed_parameters)
        
    def predict(self, to_predict=None, temperature=None, doy_series=None):
        """Predict the DOY given temperature data and associated site/year info
        All model parameters must be set either in the initial model call
        or by running fit(). If to_predict and temperature are not set, then
        this will return predictions for the data used in fitting (if available)
        
        Parameters
        ----------
        to_predict : dataframe, optional
            pandas dataframe of site/year combinations to predicte from
            the given temperature data. just like the observations 
            dataframe used in fit() but (optionally) without the doy column
        
        temperature : dataframe, optional
            pandas dataframe in the format specific to this package
            
        Returns
        -------
        predictions : array
            1D array the same length of to_predict. Or if to_predict
            is not used, the same lengh as observations used in fitting.
        
        """
        assert len(self._fitted_params) == len(self.all_required_parameters), 'Not all parameters set'
        """
        valid arg combinations
        {'to_predict':np.ndarray,'temperature':None,'doy_series':np.ndarray}
        {'to_predict':pd.DataFrame,'temperature':pd.DataFrame,doy_series':None}
        {'to_predict':None,'temperature':None,'doy_series':None}
        """
        
        if isinstance(to_predict, np.ndarray) and temperature is None and isinstance(doy_series, np.ndarray):
            # to_predict is a pre-formatted temperature array
            if len(doy_series) != to_predict.shape[0]:
                raise ValueError('to_predict axis 0 does not match doy_series')
            
            # Don't allow any nan values in 2d array
            if len(to_predict.shape)==2:
                if np.any(np.isnan(to_predict)):
                    raise ValueError('Nan values in to_predict array')
                    
            # A 3d array implies spatial data, where nan values are allowed if
            # that location is *only* nan. (ie, somewhere over water)
            elif len(to_predict.shape)==3:
                invalid_entries = np.logical_and(np.isnan(to_predict).any(0),
                                                 ~np.isnan(to_predict).all(0))
                if np.any(invalid_entries):
                    raise ValueError('Nan values in some timeseries of 3d to_predict array')
                
            else:
                raise ValueError('to_predict array is unknown shape')
                
            temp_array = to_predict
            
        elif isinstance(to_predict, pd.DataFrame) and isinstance(temperature, pd.DataFrame) and doy_series is None:
            # New data to predict
            validation.validate_temperature(temperature)
            validation.validate_observations(to_predict, for_prediction=True)
            temp_array, doy_series = utils.format_data(to_predict, temperature, for_prediction=True)
            
        elif to_predict is None and  temperature is None and doy_series is None:
            # Making predictions on data used for fitting
            if self.obs_fitting is not None and self.temperature_fitting is not None:
                temp_array = self.temperature_fitting.copy()
                to_predict = self.obs_fitting.copy()
                doy_series = self.doy_series
            else:
                raise TypeError('No to_predict + temperature passed, and'+ \
                                'no fitting done. Nothing to predict')
        else:
            raise TypeError('Invalid arguments. to_predict and temperature' + \
                            'must both be pandas dataframes of new data to predict,'+\
                            'or set to None to predict the data used for fitting')
        
        predictions = self._apply_model(temp_array.copy(),
                                        doy_series.copy(),
                                        **self._fitted_params)
        
        return predictions
        
    def _organize_parameters(self, passed_parameters):
        """Interpret each passed parameter value to a model.
        They can either be a fixed value, a range to estimate with,
        or, if missing, implying using the default range described
        in the model.
        """
        parameters_to_estimate={}
        fixed_parameters={}

        # Parameter can also be a file to load
        if isinstance(passed_parameters, str):
            passed_parameters = pd.read_csv(passed_parameters).to_dict('records')
            if len(passed_parameters)>1:
                raise Warning('Greater than 1 entry in parameter file. Using the first')
            passed_parameters = passed_parameters[0]

            # all parameters that were saved should be fixed values
            for parameter, value in passed_parameters.items():
                if not isinstance(value*1.0, float):
                    raise TypeError('Expected a set value for parameter {p} in saved file, got {v}'.format(p=parameter, v=value))
        else:
            if not isinstance(passed_parameters, dict):
                raise TypeError('passed_parameters must be either a dictionary or string')

        # This is all the required parameters updated with any
        # passed parameters. This includes any invalid ones, 
        # which will be checked for in a moment.
        params = self.all_required_parameters.copy()
        params.update(passed_parameters)

        for parameter, value in params.items():
            assert parameter in self.all_required_parameters, 'Unknown parameter: '+str(parameter)
            
            if isinstance(value, tuple):
                assert len(value)==2, 'Parameter tuple should have 2 values'
                parameters_to_estimate[parameter]=value
            elif isinstance(value*1.0, float):
                fixed_parameters[parameter]=value
            else:
                raise TypeError('unkown parameter value: '+str(type(value)) + ' for '+parameter)
    
        self._parameters_to_estimate = OrderedDict(parameters_to_estimate)
        self._fixed_parameters = OrderedDict(fixed_parameters)
        
        # If nothing to estimate then all parameters have been
        # passed as fixed values and no fitting is needed
        if len(parameters_to_estimate)==0:
            self._fitted_params = fixed_parameters
    
    def get_params(self):
        #TODO: Put a check here to make sure params are fitted
        return self._fitted_params

    def save_params(self, filename):
        """Save the parameters for a model
        """
        assert len(self._fitted_params)>0, 'Parameters not fit, nothing to save'
        pd.DataFrame([self._fitted_params]).to_csv(filename, index=False)
    
    def get_initial_bounds(self):
        #TODO: Probably just return params to estimate + fixed ones
        raise NotImplementedError()
    
    def get_doy_fitting_estimates(self, **params):
        if self.debug:
            start = time.time()
    
        output =self._apply_model(temperature = self.temperature_fitting.copy(), 
                                  doy_series = self.doy_series.copy(),
                                  **params)
        if self.debug:
            self.model_timings.append(time.time() - start)
        return output
    
    def get_error(self, **kargs):
        doy_estimates = self.get_doy_fitting_estimates(**kargs)
        error = np.sqrt(np.mean((doy_estimates - self.obs_fitting)**2))
        return error
    
    def _translate_scipy_parameters(self, parameters_array):
        """Map parameters from a 1D array to a dictionary for
        use in phenology model functions. Ordering matters
        in unpacking the scipy_array since it isn't labelled. Thus
        it relies on self._parameters_to_estimate being an 
        OrdereddDict
        """
        # If only a single value is being fit, some scipy.
        # optimizer methods will use a single
        # value instead of list of length 1. 
        try:
            _ = parameters_array[0]
        except IndexError:
            parameters_array = [parameters_array]
        labeled_parameters={}
        for i, (param,value) in enumerate(self._parameters_to_estimate.items()):
            labeled_parameters[param]=parameters_array[i]
        return labeled_parameters
    
    def _scipy_error(self,x):
        """Error function for use within scipy.optimize functions.        
        """
        parameters = self._translate_scipy_parameters(x)

        # add any fixed parameters
        parameters.update(self._fixed_parameters)
        
        return self.get_error(**parameters)
    
    def _scipy_bounds(self):
        """Bounds structured for scipy.optimize input"""
        return [bounds for param, bounds  in list(self._parameters_to_estimate.items())]
    
    def score(self, metric='rmse'):
        raise NotImplementedError()
