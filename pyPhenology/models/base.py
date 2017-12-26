import numpy as np
import pandas as pd
from . import utils, validation
import time
from collections import OrderedDict
from copy import deepcopy

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
            total_fit_time = round(time.time() - fitting_start,2)
            print('Total model fitting time: {s} sec.\n'.format(s=total_fit_time))
            
        if debug:
            n_runs = len(self.model_timings)
            mean_time = np.mean(self.model_timings).round(2)
            print('Model iterations: {n}'.format(n=n_runs))
            print('Mean timing: {t} sec/iteration \n\n'.format(t=mean_time))
            self.debug=False
        self._fitted_params.update(self._fixed_parameters)
        
    def predict(self, to_predict=None, temperature=None):
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
        
        # Both of these need to be set, or neither.
        args_are_none = [to_predict is None, temperature is None]
        if any(args_are_none) and not all(args_are_none):
            raise TypeError('Both to_predict and temperature must be set together')
        if all(args_are_none):
            if self.obs_fitting is not None and self.temperature_fitting is not None:
                temp_array = self.temperature_fitting.copy()
                to_predict = self.obs_fitting.copy()
                doy_series = self.doy_series
            else:
                raise TypeError('No to_predict + temperature passed, and'+ \
                                'no fitting done. Nothing to predict')
        else:
            validation.validate_temperature(temperature)
            validation.validate_observations(to_predict, for_prediction=True)
            temp_array, doy_series = utils.format_data(to_predict, temperature, for_prediction=True)
        
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

class Alternating(_base_model):
    """Alternating model, originally defined in Cannell & Smith 1983.
    Phenological event happens the first day that forcing is greater 
    than an exponential curve of number of chill days.
    
    Parameters
    ----------
    a : int | float
        Intercept of chill day curve
    
    b : int | float
        Slope of chill day curve
    
    c : int | float
        scale parameter of chill day curve
        
    threshold : int | flaot
        Degree threshold above which forcing accumulates, and
        below which chilling accumulates. Set to 5 (assuming C)
        by default.
        
    t1 : int
        DOY which forcing and chilling accumulationg starts. Set
        to 1 (Jan 1) by default.
    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'a':(-1000,1000), 'b':(0,5000), 'c':(-5,0),
                                        'threshold':(5,5), 't1':(1,1)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, a, b, c, threshold, t1):
        chill_days = ((temperature < threshold)*1).copy()
        chill_days[doy_series < t1]=0
        chill_days = utils.forcing_accumulator(chill_days)

        # Accumulated growing degree days from Jan 1
        gdd = temperature.copy()
        gdd[gdd < threshold]=0
        gdd[doy_series < t1]=0
        gdd = utils.forcing_accumulator(gdd)

        # Phenological event happens the first day gdd is > chill_day curve
        chill_day_curve = a + b * np.exp( c * chill_days)
        difference = gdd - chill_day_curve

        # The estimate is equal to the first day that
        # gdd - chill_day_curve > 0
        return utils.doy_estimator(difference, doy_series, threshold=0)


class BootstrapModel():
    """Fit a model using bootstrapping of the data.

    """
    def __init__(self, core_model, num_bootstraps, parameters={}):
        validation.validate_model(core_model())
        
        self.model_list=[]
        if isinstance(parameters, str):
            # A filename pointing toward a file from save_params()
            params = pd.read_csv(parameters).to_dict('records')
            for bootstrap_iteration in params:
                bootstrap_iteration.pop('bootstrap_num')
                self.model_list.append(core_model(parameters=bootstrap_iteration))
        elif isinstance(parameters, dict):
            for i in range(num_bootstraps):            
                self.model_list.append(core_model(parameters=parameters))
        elif isinstance(parameters, list):
            # If its the output of BootstrapModel.get_params()
            for bootstrap_iteration in parameters:
                bootstrap_iteration.pop('bootstrap_num')
                self.model_list.append(core_model(parameters=bootstrap_iteration))
        else:
            raise TypeError('parameters must be str or dict, got: ' + str(type(parameters)))
    
    def fit(self,observations, temperature, **kwargs):
        #TODO: do the temperature transform here cause so it doesn't get reapated a bunch
        # need to wait till fit takes arrays directly
        validation.validate_observations(observations)
        validation.validate_temperature(temperature)
        for model in self.model_list:
            obs_shuffled = observations.sample(frac=1, replace=True).copy()
            model.fit(obs_shuffled, temperature, **kwargs)

    def predict(self,to_predict=None, temperature=None, aggregation='mean', **kwargs):
        """Make predictions from the bootstrapped models.
        Predictions will be made using each of the bootstrapped models, with
        the final results being the mean or median (or other) of all bootstraps.
        
        Parameters
        ----------
        aggregation : str
            Either 'mean','median', or 'none'. 'none' return *all* predictions
            in an array of size (num_bootstraps, num_samples)
        
        """
        #TODO: do the temperature transform here cause so it doesn't get reapated a bunch
        # need to wait till predict takes arrays directly
        predictions=[]
        for model in self.model_list:
            predictions.append(model.predict(to_predict=to_predict, 
                                             temperature=temperature,
                                             **kwargs))
        
        predictions = np.array(predictions)
        if aggregation=='mean':
            predictions = np.mean(predictions, 0)
        elif aggregation=='median':
            predictions = np.median(predictions, 0)
        elif aggregation=='none':
            pass
        else:
            raise ValueError('Unknown aggregation: ' + str(aggregation))
        
        return predictions

    def get_params(self):
        all_params=[]
        for i, model in enumerate(self.model_list):
            all_params.append(deepcopy(model.get_params()))
            all_params[-1].update({'bootstrap_num':i})

        return all_params

    def save_params(self, filename):
        assert len(self.model_list[0]._fitted_params)>0, 'Parameters not fit, nothing to save'
        params = self.get_params()
        pd.DataFrame(params).to_csv(filename, index=False)

class Thermal_Time(_base_model):
    """The classic growing degree day model using
    a fixed threshold above which forcing accumulates.
    
    Parameters
    ----------
    t1 : int
        The DOY which forcing accumulating beings
    
    T : int
        The threshold above which forcing accumulates
    
    F : int, > 0
        The total forcing units required
    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'t1':(-67,298),'T':(-25,25),'F':(0,1000)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, t1, T, F):
        #Temperature threshold
        temperature[temperature<T]=0
    
        #Only accumulate forcing after t1
        temperature[doy_series<t1]=0
    
        accumulated_gdd=utils.forcing_accumulator(temperature)
    
        return utils.doy_estimator(forcing = accumulated_gdd, 
                                   doy_series = doy_series, 
                                   threshold = F)

class Uniforc(_base_model):
    """Single phase forcing model using a 
    sigmoid function for forcing units.
    Chuine 2000
    
    Parameters
    ----------
    t1 : int
        The DOY which forcing accumulating beings
    
    F : int, > 0
        The total forcing units required
        
    b : int
        Sigmoid function parameter
    
    c : int
        Sigmoid function parameter
    """
    def __init__(self, parameters={} ):
        _base_model.__init__(self)
        self.all_required_parameters = {'t1':(-67,298),'F':(0,200),'b':(-20,0),'c':(-50,50)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, t1, F, b, c):
        temperature = utils.sigmoid2(temperature, b=b, c=c)
    
        #Only accumulate forcing after t1
        temperature[doy_series<t1]=0
    
        accumulateed_forcing=utils.forcing_accumulator(temperature)
    
        return utils.doy_estimator(forcing = accumulateed_forcing,
                                   doy_series=doy_series,
                                   threshold=F)

class Unichill(_base_model):
    """Two phase forcing model using a 
    sigmoid function for forcing units 
    and chilling. 
    Chuine 2000
    
    Parameters
    ----------
    t0 : int
        The DOY which chilling accumulating beings
    
    C : int, > 0
        The total chilling units required

    F : int, > 0
        The total forcing units required
        
    b_f : int
        Sigmoid function parameter for forcing
    
    c_f : int
        Sigmoid function parameter for forcing
        
    a_c : int
        Sigmoid funcion parameter for chilling
        
    b_c : int
        Sigmoid funcion parameter for chilling
        
    c_c : int
        Sigmoid funcion parameter for chilling
    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'t0':(-67,298),'C':(0,300),'F':(0,200),
                                        'b_f':(-20,0),'c_f':(-50,50),
                                        'a_c':(0,20),'b_c':(-20,20),'c_c':(-50,50)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, t0, C, F, b_f, c_f, a_c, b_c, c_c):
        assert len(temperature.shape)==2, 'Unichill model currently only supports 2d temperature arrays'

        temp_chilling = temperature.copy()
        temp_forcing  = temperature.copy()
        
        temp_forcing = utils.sigmoid2(temp_forcing, b=b_f, c=c_f)
        temp_chilling =utils.sigmoid3(temp_chilling, a=a_c, b=b_c, c=c_c) 
    
        #Only accumulate chilling after t0
        temp_chilling[doy_series<t0]=0
        accumulated_chill=utils.forcing_accumulator(temp_chilling)
        
        # Heat forcing accumulation starts when the chilling
        # requirement, C, has been met(t1 in the equation). 
        # Enforce this by setting everything prior to that date to 0
        # TODO: optimize this so it doesn't use a for loop
        t1_values = utils.doy_estimator(forcing = accumulated_chill,
                                        doy_series=doy_series,
                                        threshold=C)
        for col, t1 in enumerate(t1_values):
            temp_forcing[doy_series<t1, col]=0
    
        accumulated_forcing = utils.forcing_accumulator(temp_forcing)
        
        return utils.doy_estimator(forcing = accumulated_forcing,
                                   doy_series=doy_series,
                                   threshold=F)

class MSB(_base_model):
    """Macroscale Species-specific Budburst model. Jeong et al. 2013
    Extension of the Alternating model which add a correction (d)
    using the mean spring temperature 
    
    Parameters
    ----------
    a : int | float
        Intercept of chill day curve
    
    b : int | float
        Slope of chill day curve
    
    c : int | float
        scale parameter of chill day curve
    
    d : int | float
        Correction factor
        
    threshold : int | flaot
        Degree threshold above which forcing accumulates, and
        below which chilling accumulates. Set to 5 (assuming C)
        by default.
        
    t1 : int
        DOY which forcing and chilling accumulationg starts. Set
        to 1 (Jan 1) by default.
    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'a':(-1000,1000), 'b':(0,5000), 'c':(-5,0),
                                        'd':(-100,100), 'threshold':(5,5), 't1':(1,1)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, a, b, c, d, threshold, t1):
        chill_days = ((temperature < threshold)*1).copy()
        chill_days[doy_series < t1]=0
        chill_days = utils.forcing_accumulator(chill_days)

        # Accumulated growing degree days from Jan 1
        gdd = temperature.copy()
        gdd[gdd < threshold]=0
        gdd[doy_series < t1]=0
        gdd = utils.forcing_accumulator(gdd)

        chill_day_curve = a + b * np.exp( c * chill_days)
        
        # Make the spring temps the same shape as chill_day_curve
        # for easy addition.
        mean_spring_temp = utils.mean_temperature(temperature, doy_series,
                                                  start_doy=0, end_doy=60)
        mean_spring_temp *= d
        # Add in correction based on per site spring temperature
        chill_day_curve += mean_spring_temp

        # Phenological event happens the first day gdd is > chill_day curve
        difference = gdd - chill_day_curve

        # The estimate is equal to the first day that
        # gdd - chill_day_curve > 0
        return utils.doy_estimator(difference, doy_series, threshold=0)

class Linear(_base_model):
    """A linear regression where DOY ~ mean_spring_tempearture
    
    Parameters
    ----------
    intercept : int | float
        y intercept of the model
    
    slope : int | float
        Slope of the model
        
    spring_start : int
        The start day of spring, defaults to Jan 1 (DOY 0)
    
    spring_end : int
        The last day of spring, defaults to March 30 (DOY 90)

    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'intercept':(-67,298),'slope':(-25,25),
                                        'spring_start':(0,0), 'spring_end':(90,90)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, intercept, slope, 
                     spring_start, spring_end):
        mean_spring_temp = utils.mean_temperature(temperature, doy_series,
                                                  start_doy = spring_start,
                                                  end_doy = spring_end)
        return mean_spring_temp * slope + intercept
